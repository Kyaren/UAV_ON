#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 16:55:30 2025

@author: shawnjx
"""

import os
import json
import sys
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument('--base_root', type=str, default='./random_logs')
args = argparser.parse_args()

base_root = args.base_root

# ===== Utility Functions =====
def get_last_distance(file_path):
    """Read the 'distance_to_end' from the last line of trajectory.jsonl"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                last = json.loads(lines[-1])
                return last.get("distance_to_end", None)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
    return None

def get_final_move_distance(traj_path):
    """Read the 'move_distance' from the last line of trajectory.jsonl"""
    try:
        with open(traj_path, 'r') as f:
            lines = f.readlines()
            if lines:
                last = json.loads(lines[-1])
                return last.get("move_distance", None)
    except Exception as e:
        print(f"Error reading {traj_path}: {e}")
    return None

def count_task_folder(folder_path):
    """Safely count number of task folders in a directory. Return 0 if not exists."""
    if os.path.exists(folder_path):
        return len([f for f in os.listdir(folder_path) if f.startswith("task_")])
    else:
        return 0

def get_geodesic_distance(desc_path):
    try:
        with open(desc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data["info"]["geodesic_distance"]
    except Exception as e:
        print(f"Error reading {desc_path}: {e}")
    return None


def count_actions(traj_path):
    """Count the number of lines (actions) in trajectory.jsonl"""
    try:
        with open(traj_path, 'r') as f:
            return len(f.readlines()) - 1
    except Exception as e:
        print(f"Error reading {traj_path}: {e}")
        return None

# ===== DTS Calculation =====
def compute_dts(folders):
    """
    Compute Distance to Success (DTS) over all tasks.
    Supports one or multiple folders.
    """
    if isinstance(folders, str):
        folders = [folders]

    distances = []

    for folder_path in folders:
        task_dirs = [d for d in os.listdir(folder_path) if d.startswith("task_")]
        for task in task_dirs:
            traj_path = os.path.join(folder_path, task, "log", "trajectory.jsonl")
            if os.path.exists(traj_path):
                d = get_last_distance(traj_path)
                if d is not None:
                    distances.append(d)

    if distances:
        return sum(distances) / len(distances), len(distances)
    else:
        return None, 0

# ===== SPL Calculation (success + failure) =====
def compute_spl(fail_folders, success_folders, oracle_folders):
    all_tasks = set()
    success_tasks = set()

    # 收集所有 task 文件夹路径（带完整路径）
    for folder in fail_folders + success_folders + oracle_folders:
        if not os.path.exists(folder):
            continue
        all_tasks.update(
            os.path.join(folder, t)
            for t in os.listdir(folder)
            if t.startswith("task_")
        )

    for folder in success_folders:
        if not os.path.exists(folder):
            continue
        success_tasks.update(
            os.path.join(folder, t)
            for t in os.listdir(folder)
            if t.startswith("task_")
        )

    spl_sum = 0.0
    total_count = 0

    for task_path in all_tasks:
        total_count += 1
        if task_path in success_tasks:
            traj_path = os.path.join(task_path, "log", "trajectory.jsonl")
            desc_path = os.path.join(task_path, "object_description.json")
            p_i = get_final_move_distance(traj_path)
            l_i = get_geodesic_distance(desc_path)

            if p_i and l_i and p_i > 0:
                spl_sum += l_i / max(p_i, l_i)

    return spl_sum / total_count if total_count else None, total_count


def classify_tasks_by_size(folders):
    size_groups = {"small": [], "mid": [], "big": []}
    for folder in folders:
        if not os.path.exists(folder):
            continue
        for task_name in os.listdir(folder):
            if not task_name.startswith("task_"):
                continue
            task_path = os.path.join(folder, task_name)
            desc_path = os.path.join(task_path, "object_description.json")
            if not os.path.exists(desc_path):
                continue
            try:
                with open(desc_path, 'r', encoding='utf-8') as f:
                    desc = json.load(f)
                    size_str = desc.get("size", "")
                    if "small" in size_str:
                        size_groups["small"].append(task_path)
                    elif "mid" in size_str:
                        size_groups["mid"].append(task_path)
                    elif "big" in size_str:
                        size_groups["big"].append(task_path)
                    else:
                        print(f"[WARN] Unknown size format in {desc_path}: {size_str}")
            except Exception as e:
                print(f"Error reading {desc_path}: {e}")
    return size_groups

def compute_dts_from_tasks(task_paths):
    """
    Compute DTS given a list of task_x folders.
    """
    distances = []
    for task_path in task_paths:
        traj_path = os.path.join(task_path, "log", "trajectory.jsonl")
        if os.path.exists(traj_path):
            d = get_last_distance(traj_path)
            if d is not None:
                distances.append(d)

    if distances:
        return sum(distances) / len(distances), len(distances)
    else:
        return None, 0

def compute_spl_from_tasks(all_tasks, success_tasks):
    spl_sum = 0.0
    total_count = len(all_tasks)

    for task_path in all_tasks:
        if task_path in success_tasks:
            traj_path = os.path.join(task_path, "log", "trajectory.jsonl")
            desc_path = os.path.join(task_path, "object_description.json")
            p_i = get_final_move_distance(traj_path)
            l_i = get_geodesic_distance(desc_path)

            if p_i and l_i and p_i > 0:
                spl_sum += l_i / max(p_i, l_i)

    return spl_sum / total_count if total_count else None, total_count

def get_termination_type(task_folder):
    traj_path = os.path.join(task_folder, "log", "trajectory.jsonl")
    if os.path.exists(traj_path):
        try:
            with open(traj_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    return "unknown"
                last = json.loads(lines[-1])
                frame = last.get("frame", -1)
                is_collision = last.get("is_collision", False)
                if is_collision:
                    return "collision"
                elif frame == 150 or len(lines) == 151:
                    return "step_limit"
                else:
                    return "stop"
        except Exception as e:
            print(f"[ERROR] reading {traj_path}: {e}")
            return "error"
    return "missing"

all_success_dirs = []
all_oracle_dirs = []
all_fail_dirs = []

for scene in os.listdir(base_root):
    scene_path = os.path.join(base_root, scene)
    if not os.path.isdir(scene_path):
        continue

    success_dir = os.path.join(scene_path, f'success_{scene}.json')
    oracle_dir = os.path.join(scene_path, f'oracle_{scene}.json')
    fail_dir = os.path.join(scene_path, f'{scene}.json')

    if os.path.exists(success_dir):
        all_success_dirs.append(success_dir)
    if os.path.exists(oracle_dir):
        all_oracle_dirs.append(oracle_dir)
    if os.path.exists(fail_dir):
        all_fail_dirs.append(fail_dir)

# ===== Task Count Statistics =====
success_tasks = sum(count_task_folder(d) for d in all_success_dirs)
oracle_tasks = sum(count_task_folder(d) for d in all_oracle_dirs)
fail_tasks = sum(count_task_folder(d) for d in all_fail_dirs)
total_tasks = success_tasks + oracle_tasks + fail_tasks

# ===== SR & OSR =====
sr = success_tasks / total_tasks if total_tasks > 0 else 0
osr = (oracle_tasks + success_tasks) / total_tasks if total_tasks > 0 else 0

# ===== DTS =====
dts_total, _ = compute_dts(all_success_dirs + all_oracle_dirs + all_fail_dirs)

# ===== SPL  =====
spl_value, _ = compute_spl(
    fail_folders=all_fail_dirs,
    success_folders=all_success_dirs,
    oracle_folders=all_oracle_dirs
)

# ===== Print All Results =====
print("===== Evaluation Results =====")
print(f"Total Tasks      : {total_tasks}")
print(f"Success Rate (SR): {sr:.2%} ({success_tasks})")
print(f"Oracle  Rate (OSR): {osr:.2%} ({oracle_tasks})\n")

print("Distance to Success (DTS):")
print(f"- All     : {dts_total:.3f}")

print("Success-weighted Path Length (SPL):")
print(f"- SPL (Success + Failure): {spl_value * 100:.3f}%")

termination_stats = {
    "collision": 0,
    "step_limit": 0,
    "stop": 0,
    "error": 0,
    "missing": 0,
    "unknown": 0
}

# 收集所有任务（成功、失败、oracle）
all_task_folders = []
for folders in [all_success_dirs, all_oracle_dirs, all_fail_dirs]:
    for folder in folders:
        if not os.path.exists(folder):
            continue
        task_dirs = [os.path.join(folder, t) for t in os.listdir(folder) if t.startswith("task_")]
        all_task_folders.extend(task_dirs)

# 统计每个任务的结束类型
for task_path in all_task_folders:
    term_type = get_termination_type(task_path)
    termination_stats[term_type] += 1

# ===== Print Termination Statistics =====
print("\n===== Termination Type Statistics =====")
for k, v in termination_stats.items():
    print(f"{k:<12}: {v}")
    
# ===== Termination Type Ratio =====
print("\n===== Termination Type Ratios =====")
total_known = (
    termination_stats["collision"]
    + termination_stats["step_limit"]
    + termination_stats["stop"]
)

if total_known > 0:
    for k in ["collision", "step_limit", "stop"]:
        ratio = termination_stats[k] / total_known
        print(f"{k:<12}: {ratio:.2%} ({termination_stats[k]})")
else:
    print("No valid termination types to calculate ratios.")

    
total_frame = 0
total_distance = 0.0
valid_frame_count = 0
valid_distance_count = 0

for task_path in all_task_folders:
    term_type = get_termination_type(task_path)
    termination_stats[term_type] += 1

    traj_path = os.path.join(task_path, "log", "trajectory.jsonl")
    if os.path.exists(traj_path):
        try:
            with open(traj_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last = json.loads(lines[-1])
                    # 统计 frame
                    if "frame" in last:
                        total_frame += last["frame"]
                        valid_frame_count += 1
                    # 统计 move_distance
                    if "move_distance" in last:
                        total_distance += last["move_distance"]
                        valid_distance_count += 1
        except Exception as e:
            print(f"[ERROR] reading {traj_path}: {e}")
            
# ===== Average Frame and Move Distance =====
avg_frame = total_frame / valid_frame_count if valid_frame_count > 0 else 0
avg_move_distance = total_distance / valid_distance_count if valid_distance_count > 0 else 0

print("\n===== Additional Statistics =====")
print(f"Average Final Frame Number     : {avg_frame:.2f}")
print(f"Average Final Move Distance    : {avg_move_distance:.2f}")





# ===== 按 size 分类任务 =====
success_by_size = classify_tasks_by_size(all_success_dirs)
oracle_by_size = classify_tasks_by_size(all_oracle_dirs)
fail_by_size = classify_tasks_by_size(all_fail_dirs)

# ===== 遍历 small / mid / big 分类统计指标 =====
for size in ["small", "mid", "big"]:
    print(f"\n===== Size Category: {size.upper()} =====")
    
    all_tasks = success_by_size[size] + oracle_by_size[size] + fail_by_size[size]
    total = len(all_tasks)
    sr = len(success_by_size[size]) / total if total else 0
    osr = (len(success_by_size[size]) + len(oracle_by_size[size])) / total if total else 0

    # 使用新版本指标计算函数（基于 task 路径）
    dts, _ = compute_dts_from_tasks(all_tasks)
    spl, _ = compute_spl_from_tasks(all_tasks, success_by_size[size])

    print(f"Total Tasks      : {total}")
    print(f"Success Rate (SR): {sr:.2%} ({len(success_by_size[size])})")
    print(f"Oracle  Rate (OSR): {osr:.2%} ({len(oracle_by_size[size])})")
    print(f"Distance to Success (DTS): {dts:.3f}" if dts is not None else "DTS: N/A")
    print(f"SPL: {spl * 100:.3f}%" if spl is not None else "SPL: N/A")

