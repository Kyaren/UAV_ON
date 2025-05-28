import torch
import tqdm
import os
from pathlib import Path
import sys
import time
import random
import numpy as np

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from common.param import args
from env_uav import AirVLNENV
from src.closeloop_util import BatchIterator, EvalBatchState, initialize_env_eval
from model_wrapper.base_model import BaseModelWrapper
from utils.logger import logger



def eval(modelWrapper: BaseModelWrapper, env: AirVLNENV ,is_fixed, save_eval_path):
    with torch.no_grad():
        data = BatchIterator(env)
        data_len = len(data)
        pbar = tqdm.tqdm(total=data_len,desc="batch")
        cnt=0
        while True:
            env_batch = env.next_minibatch(skip_scenes=[])
            if env_batch is None:
                break

            batch_state = EvalBatchState(batch_size=env.batch_size, env_batchs=env_batch, env=env, save_eval_path= save_eval_path)
            pbar.update(n = env.batch_size)
            cnt+= env.batch_size
            for t in range(args.maxActions):
                logger.info('Step: {} \t Completed: {} / {}'.format(t, cnt-batch_state.skips.count(False), data_len))
                
                if t < 10:
                    possible_actions = [
                    'forward', 
                    'left', 'right',
                    'ascend', 'descend',
                    'rotl', 'rotr'
                    ]
                else:
                    possible_actions = [
                    'forward', 
                    'left', 'right',
                    'ascend', 'descend',
                    'rotl', 'rotr',
                    'stop'
                    ]

                actions = [random.choice(possible_actions) for _ in range(env.batch_size)]
                steps_size = np.zeros(env.batch_size)
                dones = [act =='stop' for act in actions]

                batch_state.dones = dones
                print(actions)

                env.makeActions(actions, steps_size, is_fixed)
                obs = env.get_obs()
                user_prompts = [[] for _ in range(env.batch_size)]
                batch_state.update_from_env_output(obs,user_prompts,actions, steps_size, True)
                batch_state.update_metric()

                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break
        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":
    seed = 42  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    env = initialize_env_eval(dataset_path=args.dataset_path, save_path=args.eval_save_path)
    fixed = args.is_fixed

    save_eval_path = os.path.join(args.eval_save_path, args.name)
    if not os.path.exists(args.eval_save_path):
        os.makedirs(args.eval_save_path)

    modelWrapper = BaseModelWrapper

    eval(modelWrapper=modelWrapper, env=env, is_fixed=fixed, save_eval_path=save_eval_path)

    env.delete_VectorEnvUtil()