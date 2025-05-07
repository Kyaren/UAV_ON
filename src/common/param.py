import argparse
import os
import datetime
from pathlib import Path
from utils.CN import CN

import transformers
from dataclasses import dataclass, field
from typing import List, Optional


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="UAV_Search")

        project_prefix = str(Path(str(os.getcwd())).parent.resolve())
        self.parser.add_argument("--project_prefix", type=str, default=str(project_prefix))
        self.parser.add_argument("--name", type=str, default="ON-Air")
        self.parser.add_argument("--maxActions",type=int, default=150)
        self.parser.add_argument("--xOy_step_size", type=int, default=5)
        self.parser.add_argument("--z_step_size", type=int, default=2)
        self.parser.add_argument("--rotateAngle", type=int, default=45)
        self.parser.add_argument("--batchSize",type=int, default=2)
        self.parser.add_argument("--simulator_tool_port", type=int, default=30000, help="simulator_tool port")
        self.parser.add_argument("--save_path", type=str, default='./logs/eval', help="path to save the results")
        self.parser.add_argument("--dataset_path", type=str, default='/home/syx/Desktop/ModularNeighborhood/TestEpisode/UnSeenThings.json', help="path to the dataset")
        self.parser.add_argument("--is_fixed", type=bool, default=True, help="whether to use fixed action step size")
        self.parser.add_argument("--gpu_id", type=int, default=0)
        self.parser.add_argument("--generation_model_path", type=str, default='THUDM/cogvlm2-llama3-chat-19B')
        self.parser.add_argument("--eval_save_path", type=str, default='./logs/eval', help="path to save the results")

        self.args = self.parser.parse_args()

parm = Param()
args = parm.args

args.make_dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
args.logger_file_name = '{}/DATA/output/{}/logs/{}_{}.log'.format(args.project_prefix, args.name, args.name, args.make_dir_time)



args.machines_info = [
    {
        'MACHINE_IP': '127.0.0.1',
        'SOCKET_PORT': int(args.simulator_tool_port),
        'MAX_SCENE_NUM': 16,
        'open_scenes': [],
        
    },
]

default_config = CN.clone()
default_config.make_dir_time = args.make_dir_time
default_config.freeze()