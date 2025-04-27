import torch
import tqdm

from src.common.param import args
from env_uav import AirVLNENV
from src.closeloop_util import BatchIterator, EvalBatchState,




def eval(env: AirVLNENV ):
    with torch.no_grad():
        data = BatchIterator(env)
        data_len = len(data)
        pbar = tqdm.tqdm(total=data_len)
        
        while True:
            batch = env.next_minibatch(skip_scenes=[])
            
            pbar.update(args.batchSize)
            