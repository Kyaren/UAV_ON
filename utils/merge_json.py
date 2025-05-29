import os
import json
from glob import glob

# 设置你的 JSON 文件路径，这里假设都放在 ./jsons 目录下
input_folder = "../DATA"
output_file = "../DATA/merged_episodes.json"

# 读取所有 .json 文件（可根据实际后缀修改）
json_files = sorted(glob(os.path.join(input_folder, "*.json")))

merged_data = []
episode_counter = 0

for file_path in json_files:
    with open(file_path, "r") as f:
        try:
            episodes = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding {file_path}: {e}")
            continue
        
        for episode in episodes:
            episode["episode_id"] = str(episode_counter)
            merged_data.append(episode)
            episode_counter += 1

# 写入合并后的结果
with open(output_file, "w") as out_f:
    json.dump(merged_data, out_f, indent=4)

print(f"Merged {episode_counter} episodes into {output_file}")