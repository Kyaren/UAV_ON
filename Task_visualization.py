# -*- coding: utf-8 -*-
"""
Created on Sat May 10 16:37:32 2025

@author: ShawnJX
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# 统一路径前缀
base_path = "./logs/WinterTown/WinterTown.json/task_6"
# base_path = "./eval01/oracle_SeenThings.json/task_4"
# base_path = "./eval_fixed/success_SeenThings.json/task_4"


# 组合完整路径
object_description_path = os.path.join(base_path, "object_description.json")
trajectory_path = os.path.join(base_path, "log", "trajectory.jsonl")

# === 读取起点和目标点 ===
with open(object_description_path, "r") as f:
    obj_desc = json.load(f)

start_pos = obj_desc["start_pose"]["start_position"]     # 起点
target_pos = obj_desc["pose"][0]                         # 目标点

# === 读取轨迹和方向（单位向量） ===
trajectory = []
orientations = []

with open(trajectory_path, "r") as f:
    for line in f:
        data = json.loads(line)
        pos = data["sensors"]["state"]["position"]
        quat = data["sensors"]["state"]["quaternionr"]
        trajectory.append(pos)
        direction = R.from_quat(quat).apply([1, 0, 0])  # x轴朝向
        orientations.append(direction)

trajectory_df = pd.DataFrame(trajectory, columns=["x", "y", "z"])

# === 坐标平移（起点作为原点） ===
offset_x, offset_y = start_pos[0], start_pos[1]
trajectory_df["x"] -= offset_x
trajectory_df["y"] -= offset_y

# === 翻转 Z 轴（可视化习惯，Z 朝上） ===
trajectory_df["z"] = -trajectory_df["z"]

# === 起点和目标点也进行平移 & 翻转 Z ===
start_pos_trans = [0.0, 0.0, -start_pos[2]]
target_pos_trans = [
    target_pos[0] - offset_x,
    target_pos[1] - offset_y,
    -target_pos[2]
]

# === 朝向箭头 ===
arrow_origins = trajectory_df[["x", "y", "z"]].values
arrow_dirs = np.array(orientations)

# === 可视化 ===
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# UAV轨迹线
ax.plot(
    trajectory_df["x"],
    trajectory_df["y"],
    trajectory_df["z"],
    label="UAV Trajectory",
    color="blue"
)

# 起点和目标点
ax.scatter(*start_pos_trans, color='green', s=100, label='Start Pose')
ax.text(start_pos_trans[0] + 0.5, start_pos_trans[1] + 0.5, start_pos_trans[2] + 0.5, "Start", color='green')

ax.scatter(*target_pos_trans, color='red', s=100, label='Target Object')
ax.text(target_pos_trans[0] + 0.5, target_pos_trans[1] + 0.5, target_pos_trans[2] + 0.5, "Target", color='red')

# 朝向箭头（Z方向也反转）
scale = 2.0
for origin, direction in zip(arrow_origins, arrow_dirs):
    ax.quiver(
        origin[0], origin[1], origin[2],
        direction[0], direction[1], -direction[2],
        color='orange', length=scale, normalize=True
    )

# 坐标轴和标题
ax.set_xlabel("X (relative to start)")
ax.set_ylabel("Y (relative to start)")
ax.set_zlabel("Z (flipped)")
ax.set_title("UAV Trajectory and Orientation (Start at Origin)")
ax.legend()
# 固定探索区域范围
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])
ax.set_xticks(np.arange(-50, 51, 10))
ax.set_yticks(np.arange(-50, 51, 10))
plt.tight_layout()
plt.show()

# === 可选：计算终点到目标点的距离 ===
final_pos = trajectory_df.iloc[-1][["x", "y"]].values
target_2d = np.array(target_pos_trans[:2])
distance_to_target = np.linalg.norm(final_pos - target_2d)
print(f"Distance to target (XY plane): {distance_to_target:.2f} units")