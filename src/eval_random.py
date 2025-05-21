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
from src.closeloop_util import BatchIterator, EvalBatchState, initialize_env_eval, CheckPort
from utils.logger import logger
from model_wrapper.base_model import BaseModelWrapper
from model_wrapper.ON_Air import ONAir



def eval(modelWrapper: BaseModelWrapper, env: AirVLNENV ,is_fixed, save_eval_path):
    with torch.no_grad():
        data = BatchIterator(env)
        data_len = len(data)
        pbar = tqdm.tqdm(total=data_len,desc="batch")
        
        while True:
            env_batch = env.next_minibatch(skip_scenes=[])
            
            if env_batch is None:
                break

            batch_state = EvalBatchState(batch_size=env.batch_size, env_batchs=env_batch, env=env, save_eval_path= save_eval_path)
            
            pbar.update(n = env.batch_size)
            

            for t in range(args.maxActions):
                logger.info('Step: {} \t Completed: {} / {}'.format(t, int(env.index_data)-int(env.batch_size), data_len))
                
                
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
                batch_state.update_from_env_output(obs)
                batch_state.update_metric()


                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break
                
                time.sleep(0.3)
        
        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":
    

    env = initialize_env_eval(dataset_path=args.dataset_path, save_path=args.save_path)
    fixed = args.is_fixed

    save_eval_path = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    modelWrapper = BaseModelWrapper

    eval(modelWrapper=modelWrapper, env=env, is_fixed=fixed, save_eval_path=save_eval_path)

    env.delete_VectorEnvUtil()