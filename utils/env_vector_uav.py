import signal
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from threading import Thread
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
import numpy as np
import attr
import copy

from src.common.param import args

from utils.pickle5_multiprocessing import ConnectionWrapper
from utils.env_utils_uav import ENV
from utils.logger import logger


COMMAND_CLOSE = "close"
COMMAND_SET_BATCH = "set_batch"
COMMAND_GET_OBS = "get_obs_at"


try:
    # Use torch.multiprocessing if we can.
    # We have yet to find a reason to not use it and
    # you are required to use it when sending a torch.Tensor
    # between processes
    import torch
    from torch import multiprocessing as mp  # type:ignore
except ImportError:
    torch = None
    import multiprocessing as mp  # type:ignore


@attr.s(auto_attribs=True, slots=True)
class _ReadWrapper:
    r"""Convenience wrapper to track if a connection to a worker process
    should have something to read.
    """
    read_fn: Callable[[], Any]
    rank: int
    is_waiting: bool = False

    def __call__(self) -> Any:
        if not self.is_waiting:
            raise RuntimeError(
                f"Tried to read from process {self.rank}"
                " but there is nothing waiting to be read"
            )
        res = self.read_fn()
        self.is_waiting = False

        return res


@attr.s(auto_attribs=True, slots=True)
class _WriteWrapper:
    r"""Convenience wrapper to track if a connection to a worker process
    can be written to safely.  In other words, checks to make sure the
    result returned from the last write was read.
    """
    write_fn: Callable[[Any], None]
    read_wrapper: _ReadWrapper

    def __call__(self, data: Any) -> None:
        if self.read_wrapper.is_waiting:
            raise RuntimeError(
                f"Tried to write to process {self.read_wrapper.rank}"
                " but the last write has not been read"
            )
        self.write_fn(data)
        self.read_wrapper.is_waiting = True


class VectorEnvUtil:

    _num_envs: int
    _mp_ctx: BaseContext
    _workers: List[Union[mp.Process, Thread]]
    _connection_read_fns: List[_ReadWrapper]
    _connection_write_fns: List[_WriteWrapper]

    def __init__(
        self,
        load_scenes,
        num_envs: int = 1,
        multiprocessing_start_method: str = "forkserver",
        workers_ignore_signals: bool = False,
    ) -> None:
        """..

        :param make_env_fn: function which creates a single environment. An
            environment can be of type :ref:`env.Env` or :ref:`env.RLEnv`
        :param env_fn_args: tuple of tuple of args to pass to the
            :ref:`_make_env_fn`.
        :param auto_reset_done: automatically reset the environment when
            done. This functionality is provided for seamless training
            of vectorized environments.
        :param multiprocessing_start_method: the multiprocessing method used to
            spawn worker processes. Valid methods are
            :py:`{'spawn', 'forkserver', 'fork'}`; :py:`'forkserver'` is the
            recommended method as it works well with CUDA. If :py:`'fork'` is
            used, the subproccess  must be started before any other GPU useage.
        :param workers_ignore_signals: Whether or not workers will ignore SIGINT and SIGTERM
            and instead will only exit when :ref:`close` is called
        """
        self._is_closed = True

        self.load_scenes = load_scenes
        self._num_envs = int(num_envs)

        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            env_fn_args={
                'load_scenes': load_scenes,
            },
            workers_ignore_signals=workers_ignore_signals,
        )

        self._is_closed = False

    @staticmethod
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn_args: dict,
        mask_signals: bool = False,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        r"""process worker for creating and interacting with the environment."""
        if mask_signals:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            signal.signal(signal.SIGUSR2, signal.SIG_IGN)

        env = ENV(
            load_scenes=env_fn_args['load_scenes'],
        )

        if parent_pipe is not None:
            parent_pipe.close()

        try:
            command, data = connection_read_fn()
            while command != COMMAND_CLOSE:
                if command == COMMAND_SET_BATCH:
                    env.set_batch(data)
                    connection_write_fn(True)

                elif command == COMMAND_GET_OBS:
                    index, state = data
                    (done, oracle_success), state = env.get_obs_at(index, state)
                    connection_write_fn(
                        ((done, oracle_success), state)
                    )

                else:
                    raise NotImplementedError(f"Unknown command {command}")

                command, data = connection_read_fn()
        except KeyboardInterrupt:
            print("Worker KeyboardInterrupt")
        except Exception as e:
            logger.error(e)
            try:
                logger.error('command is: {} \t data is: {}'.format(command, data))
            except:
                pass
        finally:
            if child_pipe is not None:
                child_pipe.close()

    def _spawn_workers(
        self,
        env_fn_args,
        workers_ignore_signals: bool = False,
    ) -> Tuple[List[_ReadWrapper], List[_WriteWrapper]]:
        parent_connections, worker_connections = zip(
            *[
                [ConnectionWrapper(c) for c in self._mp_ctx.Pipe(duplex=True)]
                for _ in range(self._num_envs)
            ]
        )
        self._workers = []
        for worker_conn, parent_conn in zip(worker_connections, parent_connections):
            ps = self._mp_ctx.Process(
                target=self._worker_env,
                args=(
                    worker_conn.recv,
                    worker_conn.send,
                    env_fn_args,
                    workers_ignore_signals,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(cast(mp.Process, ps))
            ps.daemon = True
            ps.start()
            worker_conn.close()

        read_fns = [
            _ReadWrapper(p.recv, rank)
            for rank, p in enumerate(parent_connections)
        ]
        write_fns = [
            _WriteWrapper(p.send, read_fn)
            for p, read_fn in zip(parent_connections, read_fns)
        ]

        return read_fns, write_fns

    def close(self) -> None:
        if self._is_closed:
            return

        for read_fn in self._connection_read_fns:
            if read_fn.is_waiting:
                read_fn()
                
        for write_fn in self._connection_write_fns:
            write_fn((COMMAND_CLOSE, ''))

        self._is_closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


    def set_batch(self, batch):
        self.batch = copy.deepcopy(batch)

        for index in range(self._num_envs):
            self._connection_write_fns[index](
                (COMMAND_SET_BATCH, copy.deepcopy(batch))
            )

        results = [
            self._connection_read_fns[index]() for index in range(self._num_envs)
        ]

        return


    def get_obs(self, obs_states) -> Tuple[List[Any], List[Any]]:
        self.obs_states = obs_states

        for index in range(len(obs_states)):
            _, _, state = obs_states[index]
            self._connection_write_fns[index](
                (COMMAND_GET_OBS, (index, state))
            )

        results = [
            self._connection_read_fns[index]() for index in range(len(obs_states))
        ]

        obs = []
        sim_states = []
        for index in range(len(obs_states)):
            (done, oracle_success), sim_state = results[index]
            self.obs_states[index] = (obs_states[index][0], obs_states[index][1], sim_state)

            obs.append(
                self._format_obs_at(index, done, oracle_success)
            )
            sim_states.append(sim_state)

        return obs, sim_states

    def _format_obs_at(self, index: int, done, oracle_success):
        rgb_images, depth_images, sim_state = self.obs_states[index]

        observations = [info for info in sim_state.trajectory[-5:]]
        ###need fix
        observations[-1]['description']=sim_state.task_info['description']
        observations[-1]['object_name']=sim_state.task_info['object_name']
        observations[-1]['object_size']=sim_state.task_info['object_size']
        observations[-1]['rgb']=rgb_images
        observations[-1]['depth']=depth_images
        observations[-1]['pre_poses'] = [item['sensors']['state'] for item in sim_state.trajectory[-10:]]
        observations[-1]['step'] = sim_state.step
        observations[-1]['move_distance'] = sim_state.move_distance
        observations[-1]['start_position'] = sim_state.start_pose['start_position']
        observations[-1]['start_quaternionr'] = sim_state.start_pose['start_quaternionr']

        if len(sim_state.heading_changes)>0:
            avg_heading = sum(sim_state.heading_changes)/len(sim_state.heading_changes)
        else:
            avg_heading = 0.0
        observations[-1]['avg_heading_changes'] = round(avg_heading, 2)

        oracle_success = sim_state.oracle_success
        collision = sim_state.is_collisioned

        return observations, done, collision, oracle_success

