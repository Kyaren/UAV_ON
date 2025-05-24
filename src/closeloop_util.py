
import json
import random
import shutil

import cv2
import copy
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


def initialize_env(dataset_path, save_path, train_json_path, activate_maps=[]):
    train_env = AirVLNENV(batch_size=args.batchSize, dataset_path=dataset_path, save_path=save_path, activate_maps=activate_maps)
    return train_env

def initialize_env_eval(dataset_path, save_path):
    train_env = AirVLNENV(batch_size=args.batchSize, dataset_path=dataset_path, save_path=save_path)
    return train_env

def get_episode_by_id(episodes, target_id):
    """
    episodes: List[dict]，每个 dict 都有 'episode_id' 键
    target_id: str 或 int，要查找的 ID
    返回：匹配的那个 dict，如果找不到返回 None
    """
    return next((ep for ep in episodes if ep.get('episode_id') == target_id), None)
        
def save_to_dataset_eval(episodes, path, ori_traj_dir, task_id, collision, prompts, action_list, steps_size):
    
    root_path = os.path.join(path)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    
    os.makedirs(os.path.join(root_path, 'log'), exist_ok=True)
    print(root_path)
    save_logs(episodes, root_path, collision, action_list,steps_size)

    prompt_info = os.path.join(path, 'prompt_info.txt')
    with open(prompt_info, 'w') as f:
        for i, prompt in enumerate(prompts):
            f.write(f"\n[STEP {i} PROMPT]\n")
            f.write(f"{prompt}\n\n")

    with open(os.path.join(args.dataset_path),'r') as f:
        infos = json.load(f)
    episode_info = get_episode_by_id(infos, task_id)
    target_obj = os.path.join(path, 'object_description.json')
    with open(target_obj, 'w') as f:
        json.dump(episode_info, f, indent=2, ensure_ascii=False)
    

def save_logs(episodes, trajectory_dir, collision, action_list, steps_size):
    save_dir = os.path.join(trajectory_dir, 'log')
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'trajectory.jsonl')
    with open(log_path, 'w') as f:
        for idx, episode in enumerate(episodes):
            md = episode['move_distance']
            dt = episode['distance_to_target']

            md_2 = round(md, 2)
            dt_2 = round(dt, 2)

            info = {
                'frame': idx,
                'is_collision': collision[idx],
                'action': action_list[idx],
                'steps_size': steps_size[idx],
                'move_distance': md_2,
                'distance_to_end': dt_2,
                'sensors': episode['sensors'],
                
            }
            f.write(json.dumps(info) + '\n')


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
                    
class EvalBatchState:
    def __init__(self, batch_size, env_batchs, env, save_eval_path):
        self.batch_size = batch_size
        self.DISTANCE_TO_SUCCESS = 20
        self.eval_env = env
        self.episodes = [[] for _ in range(batch_size)]
        self.target_positions = [b['object_position'] for b in env_batchs]
        self.save_eval_path = save_eval_path
        self.ori_data_dirs = [b['trajectory_dir'] for b in env_batchs]
        self.task_id = [b['task_id'] for b in env_batchs]
        self.dones = [False] * batch_size
        self.predict_dones = [False] * batch_size
        self.collisions = [[] for _ in range(batch_size)]
        self.success = [False] * batch_size
        self.oracle_success = [False] * batch_size
        self.early_end = [False] * batch_size
        self.skips = [False] * batch_size
        self.distance_to_ends = [[] for _ in range(batch_size)]
        self.envs_to_pause = []
        self.trajectory = [[] for _ in range(batch_size)]
        self.user_prompt = [[] for _ in range(batch_size)]
        self.action_list = [[None] for _ in range(batch_size)]
        self.steps_size = [[0] for _ in range(batch_size)]
        self._initialize_batch_data()
    

    def _initialize_batch_data(self):
        outputs = self.eval_env.reset()
        observations, self.dones, collisions, self.oracle_success = [list(x) for x in zip(*outputs)]
        
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            self.collisions[i].append(collisions[i])
            self.episodes[i].append(observations[i][-1])
            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))


    def _calculate_distance(self, observation, target_position):
        obs_pos = np.array(observation['sensors']['state']['position'])
        coords = np.array(target_position)
        # 如果 coords.shape = (N,3)，说明是多点
        if coords.ndim == 2 and coords.shape[1] == 3:
            dists = np.linalg.norm(coords - obs_pos[None, :], axis=1)
            return float(dists.min())
        else:
            return float(np.linalg.norm(obs_pos - coords))

    def update_from_env_output(self, outputs, user_prompts, actions, steps_size, is_fixed):
        observations, self.dones, collisions, self.oracle_success = [list(x) for x in zip(*outputs)]
       
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            for j in range(len(observations[i])):
                self.episodes[i].append(observations[i][j])
            self.collisions[i].append(collisions[i])
            self.user_prompt[i].append(user_prompts[i])
            self.action_list[i].append(actions[i])
            
            if is_fixed:
                if actions[i]=='rotr' or actions[i]=='rotl':
                    self.steps_size[i].append(args.rotateAngle)
                elif actions[i]=='ascend' or actions[i]=='descend':
                    self.steps_size[i].append(args.z_step_size)
                else:
                    self.steps_size[i].append(args.xOy_step_size)
            else:
                self.steps_size[i].append(steps_size[i])

            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))
          


    def update_metric(self):
        for i in range(self.batch_size):
            if self.dones[i] and not self.skips[i]:
                if self.distance_to_ends[i][-1] <= self.DISTANCE_TO_SUCCESS:
                    self.success[i] = True

            if self.collisions[i][-1]:
                self.dones[i] = True
                print(f"episode {i} has collision!")
                self.success[i] = False
                self.oracle_success[i] = False
                    
    def check_batch_termination(self, t):
        for i in range(self.batch_size):
            if t == args.maxActions - 1:
                self.dones[i] = True
                if self.distance_to_ends[i][-1] <= self.DISTANCE_TO_SUCCESS:
                    self.oracle_success[i] = True
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
                new_traj_dir = os.path.join(args.eval_save_path, new_traj_name, f"task_{self.task_id[i]}")
                
                full_traj = self.eval_env.sim_states[i].trajectory
                self.trajectory[i] = copy.deepcopy(full_traj)
                
                save_to_dataset_eval(self.trajectory[i], new_traj_dir, self.ori_data_dirs[i], self.task_id[i], self.collisions[i], self.user_prompt[i], self.action_list[i], self.steps_size[i])
                self.skips[i] = True
                print(i, " has finished!")
        return np.array(self.skips).all()
    

        