from collections import OrderedDict
import copy
import random
import sys
import time
import numpy as np
import math
import os
import json
from pathlib import Path
import airsim
import random
from typing import Dict, List, Optional

import tqdm
from src.common.param import args
from utils.logger import logger
from airsim_plugin.airsim_settings import AirsimActions
sys.path.append(str(Path(str(os.getcwd())).resolve()))
from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool
from utils.env_utils_uav import SimState, getNextPosition
from utils.env_vector_uav import VectorEnvUtil


def prepare_object_map():
    with open(args.map_spawn_area_json_path, 'r') as f:
        map_dict = json.load(f)
    return map_dict


class AirVLNENV:
    def __init__(self, batch_size=8, 
                 dataset_path=None,
                 save_path=None,
                 eval_json_path=None,
                 seed=1,
                 activate_maps=[]
                 ):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.eval_json_path = eval_json_path
        self.seed = seed
        self.collected_keys = set()
        #self.dataset_group_by_scene = dataset_group_by_scene
        self.activate_maps = set(activate_maps)
        self.map_area_dict = prepare_object_map()
        self.exist_save_path = save_path
        load_data = self.load_my_datasets()
        self.data = load_data
        logger.info('Loaded dataset {}.'.format(len(self.eval_json_path)))
        self.index_data = 0
        
        self.data = self._group_scenes()
        logger.warning('dataset grouped by scene, ')

        scenes = [item['map_name'] for item in self.data]
        self.scenes = set(scenes)
        self.sim_states: Optional[List[SimState]] = [None for _ in range(batch_size)]
        self.last_using_map_list = []
        self.one_scene_could_use_num = 5e3
        self.this_scene_used_cnt = 0
        self.init_VectorEnvUtil()

    def load_my_datasets(self):
        """
            load object location json file, reconstruct a json file with every infomation
            
            return: object_info (contains position, rotation, scale, object name, instruction )
        """
        object_list= json.load(open(self.dataset_path, 'r'))

        return 
    
    def _group_scenes(self):
        """
            group all objects with their scene name, choose objects which
        """
        scene_sort_keys: OrderedDict[str, int] = {}
        for item in self.data:
            if str(item['map_name']) not in scene_sort_keys:
                scene_sort_keys[str(item['map_name'])] = len(scene_sort_keys)
        return sorted(self.data, key=lambda e: (scene_sort_keys[str(e['map_name'])]))
        

    def init_VectorEnvUtil(self):
        self.delete_VectorEnvUtil()
        self.VectorEnvUtil = VectorEnvUtil(self.scenes, self.batch_size)

    def delete_VectorEnvUtil(self):
        if hasattr(self, 'VectorEnvUtil'):
            del self.VectorEnvUtil
        import gc
        gc.collect()

    ###load_json后需要修改
    def next_minibatch(self, skip_scenes=[], data_it=0):
        batch = []
        while True:
            if self.index_data >= len(self.data):
                random.shuffle(self.data)
                logger.warning('random shuffle data')
                if self.dataset_group_by_scene:
                    self.data = self._group_scenes()
                    logger.warning('dataset grouped by scene')

                if len(batch) == 0:
                    self.index_data = 0
                    self.batch = None
                    return

                self.index_data = self.batch_size - len(batch)
                batch += self.data[:self.index_data]
                break

            task = self.data[self.index_data]

            if task['map_name'] in skip_scenes:
                self.index_data += 1
                continue

            if args.run_type in ['collect', 'train'] and args.collect_type in ['dagger', 'SF']:
                
                _key = '{}_{}'.format(new_trajectory['seq_name'], data_it)
                if _key in self.collected_keys:
                    self.index_data += 1
                    continue
                else:
                    batch.append(new_trajectory)
                    self.index_data += 1
            else:
                batch.append(new_trajectory)
                self.index_data += 1

            if len(batch) == self.batch_size:
                break 

        self.batch = copy.deepcopy(batch)
        assert len(self.batch) == self.batch_size, 'next_minibatch error'
        self.VectorEnvUtil.set_batch(self.batch)
        return self.batch

    def getObjectList(self):
        pass
    
    def changeToNewTask(self):
        self._changeEnv(need_change=False)
        
        self._setDrone()

        self.update_measurements()

    
    def _setDrone(self,):
        drone_info = [item['drone'] for item in self.batch]
        poses = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        x_val=drone_info[cnt]['position'][0],
                        y_val=drone_info[cnt]['position'][1],
                        z_val=drone_info[cnt]['position'][2],
                    ),
                    orientation_val=airsim.Quaternionr(
                        x_val=drone_info[cnt]['orientation'][0],
                        y_val=drone_info[cnt]['orientation'][1],
                        z_val=drone_info[cnt]['orientation'][2],
                        w_val=drone_info[cnt]['orientation'][3],
                    ),
                )
                poses[index_1].append(pose)
                cnt += 1
                self.simulator_tool.setPoses(poses=drone_info)
                state_info_results = self.simulator_tool.getSensorInfo()
    
                self.sim_states[cnt] = SimState(index=cnt, step=0, raw_trajectory_info=self.batch[cnt])
                self.sim_states[cnt].sensorInfo = [state_info_results[index_1][index_2]]


    def _changeEnv(self, need_change: bool = True):
        using_map_list = [item['map_name'] for item in self.batch]
        
        assert len(using_map_list) == self.batch_size, '错误'

        machines_info_template = copy.deepcopy(args.machines_info)
        total_max_scene_num = 0
        for item in machines_info_template:
            total_max_scene_num += item['MAX_SCENE_NUM']
        assert self.batch_size <= total_max_scene_num, 'error args param: batch_size'

        machines_info = []
        ix = 0
        for index, item in enumerate(machines_info_template):
            machines_info.append(item)
            delta = min(self.batch_size, item['MAX_SCENE_NUM'], len(using_map_list)-ix)
            machines_info[index]['open_scenes'] = using_map_list[ix : ix + delta]
            machines_info[index]['gpus'] = [args.gpu_id] * 8
            ix += delta

        cnt = 0
        for item in machines_info:
            cnt += len(item['open_scenes'])
        assert self.batch_size == cnt, 'error create machines_info'

        #
        if self.this_scene_used_cnt < self.one_scene_could_use_num and \
            len(set(using_map_list)) == 1 and len(set(self.last_using_map_list)) == 1 and \
            using_map_list[0] is not None and self.last_using_map_list[0] is not None and \
            using_map_list[0] == self.last_using_map_list[0] and \
            need_change == False:
            self.this_scene_used_cnt += 1
            logger.warning('no need to change env: {}'.format(using_map_list))
            # use the current environments
            return
        else:
            logger.warning('to change env: {}'.format(using_map_list))
 
        #
        while True:
            try:
                self.machines_info = copy.deepcopy(machines_info)
                print('machines_info:', self.machines_info)
                self.simulator_tool = AirVLNSimulatorClientTool(machines_info=self.machines_info)
                self.simulator_tool.run_call()
                break
            except Exception as e:
                logger.error("启动场景失败 {}".format(e))
                time.sleep(3)
            except:
                logger.error('启动场景失败')
                time.sleep(3)

        self.last_using_map_list = using_map_list.copy()
        self.this_scene_used_cnt = 1


    def get_obs(self):
        obs_states = self._getStates()
        obs, states = self.VectorEnvUtil.get_obs(obs_states)
        self.sim_states = states
        return obs

    def _getStates(self):
        responses = self.simulator_tool.getImageResponses()
        cnt = 0
        for item in responses:
            cnt += len(item)
        assert len(responses) == len(self.machines_info), 'error'
        assert cnt == self.batch_size, 'error'

        states = [None for _ in range(self.batch_size)]
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2 in range(len(item['open_scenes'])):
                rgb_images = responses[index_1][index_2][0]
                depth_images = responses[index_1][index_2][1]
                state = self.sim_states[cnt]
                states[cnt] = (rgb_images, depth_images, state)
                cnt += 1
        return states
    
    def _get_current_state(self) -> list:
        states = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            states.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                s = self.sim_states[cnt].state
                state = airsim.KinematicsState()
                state.position = airsim.Vector3r(*s['position'])
                state.orientation = airsim.Quaternionr(*s['orientation'])
                state.linear_velocity = airsim.Vector3r(*s['linear_velocity'])
                state.angular_velocity = airsim.Vector3r(*s['angular_velocity'])
                states[index_1].append(state)
                cnt += 1
        return states

    def _get_current_pose(self) -> list:
        poses = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                poses[index_1].append(
                    self.sim_states[cnt].pose
                )
                cnt += 1
        return poses

    def reset(self):
        self.changeToNewTrajectorys()
        return self.get_obs()

    def revert2frame(self, index):
        self.sim_states[index].revert2frames()
        
    def makeActions(self, action_list, steps_size):
        poses = []
        fly_types = []
        for index, action in enumerate(action_list):
            if self.sim_states[index].is_end == True:
                action = AirsimActions.STOP
                # continue
            if action == AirsimActions.STOP or self.sim_states[index].step >= int(args.maxAction):
                self.sim_states[index].is_end = True

            current_pose = self.sim_states[index].pose
            (new_pose, fly_type) = getNextPosition(current_pose, action, steps_size[index])
            poses.append(new_pose)
            fly_types.append(fly_type)

        result = self.simulator_tool.move_to_next_pose(poses_list=poses, fly_types=fly_types)
        if not result:
            logger.error('move_to_next_pose error')

        for index, action in enumerate(action_list):
            if self.sim_states[index].is_end == True:
                continue

            if action == AirsimActions.STOP or self.sim_states[index].step >= int(args.maxAction):
                self.sim_states[index].is_end = True

            self.sim_states[index].step += 1
            self.sim_states[index].pose = poses[index]
            self.sim_states[index].trajectory.append([
                poses[index].position.x_val, poses[index].position.y_val, poses[index].position.z_val, # xyz
                poses[index].orientation.x_val, poses[index].orientation.y_val, poses[index].orientation.z_val, poses[index].orientation.w_val,
            ])
            

    def update_measurements(self):
        self._update_distance_to_target()
        
    def _update_distance_to_target(self):
        target_positions = [item['object_position'] for item in self.batch]
        for idx, target_position in enumerate(target_positions):
            current_position = self.sim_states[idx].pose[0:3]
            distance = np.linalg.norm(np.array(current_position) - np.array(target_position))
            print(f'batch[{idx}/{len(self.batch)}]| distance: {round(distance, 2)}, position: {current_position[0]}, {current_position[1]}, {current_position[2]}, target: {target_position[0]}, {target_position[1]}, {target_position[2]}')
     