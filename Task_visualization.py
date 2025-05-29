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
# base_path = "./Logs/unfixed_logs/BrushifyUrban/success_BrushifyUrban.json/task_26"

base_path = "./Logs/eval_fixed/BrushifyUrban/BrushifyUrban.json/task_4"



# base_path = "./logs/eval_fixed/CabinLake/success_CabinLake.json/task_50"
# base_path = "./Logs/eval_fixed/CabinLake/oracle_CabinLake.json/task_40"
# base_path = "./logs/eval_fixed/Neighborhood/Neighborhood.json/task_0"


# base_path = "./eval_unfixed/WinterTown/WinterTown.json/task_1"
# base_path = "./logs/unfixed_logs/Slum/Slum.json/task_1"


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

# 全局字体大小设置
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

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
ax.text(start_pos_trans[0] + 0.5, start_pos_trans[1] + 0.5, start_pos_trans[2] + 0.5, "Start", color='green', fontweight='bold',       # 加粗
bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=2))

ax.scatter(*target_pos_trans, color='red', s=100, label='Target Object')
ax.text(target_pos_trans[0] + 0.5, target_pos_trans[1] + 0.5, target_pos_trans[2] + 0.5, "Target", color='red', fontweight='bold',       # 加粗
bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=2))

# 最终位置
final_pos = trajectory_df.iloc[-1][["x", "y", "z"]].values
ax.scatter(*final_pos, color='purple', s=100, label='Final Position')
ax.text(final_pos[0] + 0.5, final_pos[1] + 0.5, final_pos[2] + 0.5, "Final", color='purple', fontweight='bold',       # 加粗
bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=2))

# === 朝向箭头（Z方向也反转） ===
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
ax.legend()
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])
ax.set_xticks(np.arange(-50, 51, 10))
ax.set_yticks(np.arange(-50, 51, 10))

# === 添加距离文字（XY 平面） ===
distance_to_target = np.linalg.norm(final_pos - target_pos_trans)
distance_text = f"Dist to Target: {distance_to_target:.2f} units"
# === 使用 3D 中点 ===
midpoint_3d = (final_pos + np.array(target_pos_trans)) / 2
ax.text(
    midpoint_3d[0], midpoint_3d[1], midpoint_3d[2],
    distance_text,
    color='darkred',         # 字体颜色
    fontsize=18,             # 字体大小
    fontweight='bold',       # 加粗
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=2)  # 白底红框
)
plt.tight_layout()
plt.show()

# 控制台也输出
print(f"Distance to target (XY plane): {distance_to_target:.2f} units")
