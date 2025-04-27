
import json
import random
import shutil

import cv2
import numpy as np
from utils.utils import *
from src.common.param import args
import torch.backends.cudnn as cudnn
from src.env_uav import AirVLNENV


def setup(dagger_it=0, manual_init_distributed_mode=False):
    if not manual_init_distributed_mode:
        init_distributed_mode()

    seed = 100 + get_rank() + dagger_it
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False

def CheckPort():
    pid = FromPortGetPid(int(args.DDP_MASTER_PORT))
    if pid is not None:
        print('DDP_MASTER_PORT ({}) is being used'.format(args.DDP_MASTER_PORT))
        return False

    return True

def initialize_env(dataset_path, save_path, train_json_path, activate_maps=[]):
    train_env = AirVLNENV(batch_size=args.batchSize, dataset_path=dataset_path, save_path=save_path, eval_json_path=train_json_path, activate_maps=activate_maps)
    return train_env

def initialize_env_eval(dataset_path, save_path, eval_json_path):
    train_env = AirVLNENV(batch_size=args.batchSize, dataset_path=dataset_path, save_path=save_path, eval_json_path=eval_json_path)
    return train_env

def save_to_dataset_dagger(episodes, path, dagger_it, teacher_after_collision_steps):
    ori_path = path
    path_parts = ori_path.strip('/').split('/')
    map_name, seq_name = path_parts[-2], path_parts[-1]
    root_path = os.path.join(args.dagger_save_path, seq_name)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    folder_names = ['log']
    for folder_name in folder_names:
        os.makedirs(os.path.join(root_path, folder_name), exist_ok=True)
    save_logs(episodes, root_path)


    ori_obj = os.path.join(ori_path, 'object_description.json')
    target_obj = os.path.join(root_path, 'object_description.json')
    shutil.copy2(ori_obj, target_obj)
    with open(os.path.join(root_path, 'dagger_info.json'), 'w') as f:
        json.dump({'teacher_after_collision_steps': teacher_after_collision_steps,
                   'map_name': map_name,
                   'seq_name': seq_name}, f)
        
def save_to_dataset_eval(episodes, path, ori_traj_dir):
    root_path = os.path.join(path)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    folder_names = ['log']
    for folder_name in folder_names:
        os.makedirs(os.path.join(root_path, folder_name), exist_ok=True)
    print(root_path)
    save_logs(episodes, root_path)

    ori_obj = os.path.join(ori_traj_dir, 'object_description.json')
    target_obj = os.path.join(root_path, 'object_description.json')
    shutil.copy2(ori_obj, target_obj)
    with open(os.path.join(path, 'ori_info.json'), 'w') as f:
        json.dump({'ori_traj_dir': ori_traj_dir}, f)

def save_logs(episodes, trajectory_dir):
    save_dir = os.path.join(trajectory_dir, 'log')
    for idx, episode in enumerate(episodes):
        info = {'frame': idx, 'sensors': episode['sensors']}
        with open(os.path.join(save_dir, str(idx).zfill(6) + '.json'), 'w') as f:
            json.dump(info, f)


def load_object_description():
    object_desc_dict = dict()
    with open(args.object_name_json_path, 'r') as f:
        file = json.load(f)
        for item in file:
            object_desc_dict[item['object_name']] = item['object_desc']
    return object_desc_dict

def target_distance_increasing_for_10frames(lst):
    if len(lst) < 10:
        return False
    sublist = lst[-10:]
    for i in range(1, len(sublist)):
        if sublist[i] < sublist[i - 1]:
            return False
    return True

class BatchIterator:
    def __init__(self, env: AirVLNENV):
        self.env = env
    
    def __len__(self):
        return len(self.env.data)
    
    def __next__(self):
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch
    
    def __iter__(self):
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch

class DaggerBatchState:
    def __init__(self, bs, env_batchs, train_env):
        self.bs = bs
        self.episodes = [[] for _ in range(bs)]
        self.train_env = train_env
        self.skips = [False] * bs
        self.dones = [False] * bs
        self.oracle_success = [False] * bs
        self.collisions = [False] * bs
        self.back_count = [dict() for _ in range(bs)] 
        self.envs_to_pause = []
        self.paths = [b['trajectory_dir'] for b in env_batchs]
        self.target_positions = [b['object_position'] for b in env_batchs]
        object_desc_dict = load_object_description()
        self.object_infos = [object_desc_dict.get(b['object']['asset_name'].replace("AA", "")) for b in env_batchs]
        self.trajs = [b['trajectory'] for b in env_batchs]
        
    def update_from_env_output(self, outputs, check_collision_function=None):
        observations, dones, collisions, oracle_success = [list(x) for x in zip(*outputs)]
        if check_collision_function is not None:
            collisions, dones = check_collision_function(self.episodes, observations, collisions, dones)
        for i in range(self.bs):
            if i in self.envs_to_pause:
                continue
            self.episodes[i].append(observations[i][-1])
            if oracle_success[i]:
                dones[i] = True
        self.oracle_success = oracle_success
        self.dones = dones
        self.collisions = collisions
        return
    
    
    def check_dagger_batch_termination(self, dagger_it):
        for i in range(self.bs):
            ep = self.episodes[i]
            if not self.skips[i] and ((self.dones[i] and not self.collisions[i]) or (len(self.episodes[i]) >= args.maxWaypoints * 5 // 10 and self.collisions[i])):
                ori_path = self.paths[i]
                self.skips[i] = True
                if self.collisions[i]:
                    ep = ep[:-25]
                save_to_dataset_dagger(ep, ori_path, dagger_it, self.teacher_after_collision_steps[i])
            elif len(ep) < args.maxWaypoints * 5 // 10 and self.collisions[i] and not self.skips[i]: # the dagger is not long enough, so we don't save this data
                self.skips[i] = True
        if all(self.dones):
            return True # terminate
        return False 
    
    def dagger_step_back(self):
        # if collisions without teacher action, return to last 2 frame and move with teacher action
        for i in range(self.bs):
            if self.dones[i] or i in self.envs_to_pause:
                continue
            # If no collision occurs or no teacher intervention is required, apply ModelWrapper control.
            # If a collision occurs and teacher intervention is required, the DAgger trajectory fails, and the training ends.
            # If current step is using teacher action, disable the teacher flag and apply ModelWrapper control.
            if not self.collisions[i] and self.need_teacher[i]:
                self.need_teacher[i] = False
            elif self.collisions[i] and not self.need_teacher[i]:
                if (len(self.episodes[i]) in self.back_count[i] and self.back_count[i][len(self.episodes[i])] > 3) or sum(self.back_count[i].values()) > 30:
                    continue
                else:
                    self.back_count[i][len(self.episodes[i])] = self.back_count[i].get(len(self.episodes[i]), 0) + 1
                    self.train_env.revert2frame(i)
                    self.need_teacher[i] = True
                    self.collisions[i] = False
                    # reset the done flag caused by collision
                    self.dones[i] = False
                    if len(self.episodes[i]) > 10:
                        self.episodes[i] = self.episodes[i][0:-10]
                    else:
                        self.episodes[i] = self.episodes[i][0:1]
                    assert len(self.episodes[i]) == len(self.train_env.sim_states[i].trajectory)
                    remove_index = 0
                    for teacher_after_collision_step in self.teacher_after_collision_steps[i][::-1]:
                        if teacher_after_collision_step >= len(self.episodes[i]):
                            remove_index -= 1
                    self.teacher_after_collision_steps[i] = self.teacher_after_collision_steps[i][0: (None if remove_index==0 else remove_index)]
                    self.teacher_after_collision_steps[i].append(len(self.episodes[i]))
                    
                    
class EvalBatchState:
    def __init__(self, batch_size, env_batchs, env):
        self.batch_size = batch_size
        self.eval_env = env
        self.episodes = [[] for _ in range(batch_size)]
        self.target_positions = [b['object_position'] for b in env_batchs]
        self.object_infos = [self._get_object_info(b) for b in env_batchs]
        self.trajs = [b['trajectory'] for b in env_batchs]
        self.ori_data_dirs = [b['trajectory_dir'] for b in env_batchs]
        self.dones = [False] * batch_size
        self.predict_dones = [False] * batch_size
        self.collisions = [False] * batch_size
        self.success = [False] * batch_size
        self.oracle_success = [False] * batch_size
        self.early_end = [False] * batch_size
        self.skips = [False] * batch_size
        self.distance_to_ends = [[] for _ in range(batch_size)]
        self.envs_to_pause = []
        
        self._initialize_batch_data()

    def _get_object_info(self, batch):
        object_desc_dict = self._load_object_description()
        return object_desc_dict.get(batch['object']['asset_name'].replace("AA", ""))

    def _load_object_description(self):
        with open(args.object_name_json_path, 'r') as f:
            return {item['object_name']: item['object_desc'] for item in json.load(f)}

    def _initialize_batch_data(self):
        outputs = self.eval_env.reset()
        observations, self.dones, self.collisions, self.oracle_success = [list(x) for x in zip(*outputs)]
        
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            self.episodes[i].append(observations[i][-1])
            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))

    def _calculate_distance(self, observation, target_position):
        return np.linalg.norm(np.array(observation['sensors']['state']['position']) - np.array(target_position))

    def update_from_env_output(self, outputs):
        observations, self.dones, self.collisions, self.oracle_success = [list(x) for x in zip(*outputs)]
        self.collisions, self.dones = self.assist.check_collision_by_depth(self.episodes, observations, self.collisions, self.dones)
        
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            for j in range(len(observations[i])):
                self.episodes[i].append(observations[i][j])
            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))
            if target_distance_increasing_for_10frames(self.distance_to_ends[i]):
                self.collisions[i] = True
                self.dones[i] = True


    def update_metric(self):
        for i in range(self.batch_size):
            if self.dones[i]:
                continue
            if self.predict_dones[i] and not self.skips[i]:
                if self.distance_to_ends[i][-1] <= 20 and not self.early_end[i]:
                    self.success[i] = True
                elif self.distance_to_ends[i][-1] > 20:
                    self.early_end[i] = True
                if self.oracle_success[i] and self.early_end[i]:
                    self.dones[i] = True
                elif self.success[i]:
                    self.dones[i] = True
                    
    def check_batch_termination(self, t):
        for i in range(self.batch_size):
            if t == args.maxWaypoints:
                self.dones[i] = True
            if self.dones[i] and not self.skips[i]:
                self.envs_to_pause.append(i)
                prex = ''
                if self.success[i]:
                    prex = 'success_'
                    print(i, " has succeed!")
                elif self.oracle_success[i]:
                    prex = "oracle_"
                    print(i, " has oracle succeed!")
                new_traj_name = prex +  self.ori_data_dirs[i].split('/')[-1]
                new_traj_dir = os.path.join(args.eval_save_path, new_traj_name)
                save_to_dataset_eval(self.episodes[i], new_traj_dir, self.ori_data_dirs[i])
                self.skips[i] = True
                print(i, " has finished!")
        return np.array(self.skips).all()