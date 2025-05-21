import torch
import tqdm
import os
from pathlib import Path
import sys
import time
sys.path.append(str(Path(str(os.getcwd())).resolve()))
from common.param import args
from env_uav import AirVLNENV
from src.closeloop_util import BatchIterator, EvalBatchState, initialize_env_eval
from utils.logger import logger
from model_wrapper.base_model import BaseModelWrapper
from model_wrapper.ON_Air_2 import ONAir

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
           
            inputs ,user_prompts = modelWrapper.prepare_inputs(batch_state.episodes,is_fixed)
            cnt+= env.batch_size
            for t in range(args.maxActions):
                
                

                logger.info('Step: {} \t Completed: {} / {}'.format(t, cnt-batch_state.skips.count(False), data_len))

                
                start1 = time.time()
                actions, steps_size, dones = modelWrapper.run(inputs, is_fixed)
                print("get actions time:",time.time()-start1)
                
                for i in range(env.batch_size):
                    if dones[i]:
                        batch_state.dones[i] = True
                        
                for i in range(len(actions)):
                    print(actions[i],":",steps_size[i])
                env.makeActions(actions, steps_size, is_fixed)
                
                ###get next step observations
                obs = env.get_obs()   
                batch_state.update_from_env_output(obs,user_prompts,actions, steps_size, is_fixed)     
                batch_state.update_metric()
                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break
                
                start2 = time.time()
                inputs, user_prompts = modelWrapper.prepare_inputs(batch_state.episodes, is_fixed)
                #print("prepare inputs time:",time.time()-start2)
        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":
    

    env = initialize_env_eval(dataset_path=args.dataset_path, save_path=args.eval_save_path)
    fixed = args.is_fixed

    save_eval_path = os.path.join(args.eval_save_path, args.name)
    if not os.path.exists(args.eval_save_path):
        os.makedirs(args.eval_save_path)

    modelWrapper = ONAir(fixed=fixed, batch_size=args.batchSize)

    eval(modelWrapper=modelWrapper, env=env, is_fixed=fixed, save_eval_path=save_eval_path)

    env.delete_VectorEnvUtil()