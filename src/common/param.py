'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-05-19 21:10:39
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-05-21 18:52:11
FilePath: /UAV_Search/src/common/param.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
import os
import datetime
from pathlib import Path
from utils.CN import CN

import transformers
from dataclasses import dataclass, field
from typing import List, Optional


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="UAV_Search")

        project_prefix = str(Path(str(os.getcwd())).parent.resolve())
        self.parser.add_argument("--project_prefix", type=str, default=str(project_prefix))
        self.parser.add_argument("--name", type=str, default="ON-Air")
        self.parser.add_argument("--maxActions",type=int, default=150)
        self.parser.add_argument("--xOy_step_size", type=int, default=5)
        self.parser.add_argument("--z_step_size", type=int, default=2)
        self.parser.add_argument("--rotateAngle", type=int, default=15)
        self.parser.add_argument("--batchSize",type=int, default=6)
        self.parser.add_argument("--simulator_tool_port", type=int, default=31000, help="simulator_tool port")
        self.parser.add_argument("--dataset_path", type=str, default='../DATA/SeenThings.json', help="path to the dataset")
        self.parser.add_argument("--is_fixed", type=str2bool, default=True, help="whether to use fixed action step size")
        self.parser.add_argument("--gpu_id", type=int, default=0)
        self.parser.add_argument("--generation_model_path", type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
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