#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 18:20:37 2025

@author: shawnjx
"""

import os
import json

# 数据目录
data_dir = "../DATA_test"

# 初始化总计数
total_episodes = 0

# 遍历所有 json 文件
for filename in os.listdir(data_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

            if len(data) == 0:
                print(f"⚠️ 文件 {filename} 是空的，跳过")
                continue

            last_episode = data[-1]
            episode_id = int(last_episode.get("episode_id", -1))
            count = episode_id + 1
            total_episodes += count

            print(f"{filename}: {count} episodes")

print(f"\n✅ Total episodes across all files: {total_episodes}")