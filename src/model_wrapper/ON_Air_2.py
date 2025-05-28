from model_wrapper.base_model import BaseModelWrapper
from airsim_plugin.airsim_settings import AirsimActionSettings
#from model_wrapper.Qwen_api_captions_2 import generate_caption, encode_image
from model_wrapper.Qwen_api_captions import generate_caption, encode_image
from openai import AsyncClient
from io import BytesIO
from src.common.param import args
from common.prompts import fixed_system_prompt, fixed_user_prompt_template, unfixed_system_prompt, unfixed_user_prompt_template

import numpy as np
import asyncio
import math
import airsim
import copy
import time
import torch
import torch.nn.functional as F
import os
import json

class ONAir(BaseModelWrapper):
    def __init__(self, fixed, batch_size):
        super().__init__()
        self.fixed = fixed
        self.gpt_client = AsyncClient()
        self.start_position = [[] for _ in range(batch_size)]
        self.start_yaw = [0 for _ in range(batch_size)]
        self.current_poses = [[] for _ in range(batch_size)]

        self.unfixed_system_prompt = unfixed_system_prompt
        self.fixed_system_prompt = fixed_system_prompt


    def prepare_inputs(self, episodes, fixed):
        inputs = []
        user_prompts = []
        images = []
        depth_images = []

        for i in range(len(episodes)):
            sources = episodes[i]
            for src in sources[::-1]:
                if 'rgb' in src and 'depth' in src:
                    for img in src['rgb']:
                        images.append(img)
                    depth_images.extend(src['depth'])
                    break
  
        b64_imgs = encode_image(images)

        GROUP = 4
        GROUP_PER_BATCH = 2 
        BATCH_IMG = GROUP * GROUP_PER_BATCH
        
        def iterate_batches(img_list):
            n = len(img_list)
            full_batches = n // BATCH_IMG          # 完整批次数
            tail        = n %  BATCH_IMG           # 残余张数

            for b in range(full_batches):
                yield img_list[b*BATCH_IMG : (b+1)*BATCH_IMG]

            if tail:                               # 处理最后不足 8 张
                yield img_list[-tail:] 
        captions = []
        print("start generate caption")
        start=time.time()
        for imgs in iterate_batches(b64_imgs):
            # raw = generate_caption_qwen_api(imgs)
            raw = generate_caption(imgs)
            
            if len(raw) != len(imgs):
                raise ValueError(f"Expected {len(imgs)} captions, got {len(raw)}")
            captions.extend(raw)

        print("generation captions time:", time.time()-start)
      

        for i in range(len(episodes)):
            
            captions4 = captions[4*i:4*i+4]
            
            self.start_position[i] = episodes[i][-1]['start_position']
            
            quaternionr = airsim.Quaternionr(x_val=episodes[i][-1]['start_quaternionr'][0],
                                             y_val=episodes[i][-1]['start_quaternionr'][1],
                                             z_val=episodes[i][-1]['start_quaternionr'][2],
                                             w_val=episodes[i][-1]['start_quaternionr'][3])
            pitch, roll, yaw = airsim.to_eularian_angles(quaternionr)
            self.start_yaw[i] = math.degrees(yaw)
            
            step_num = episodes[i][-1]['step']
            description = episodes[i][-1]['description']
            object_name = episodes[i][-1]['object_name']
            object_size = episodes[i][-1]['object_size']
            depth_info = self.process_depth(depth_images=depth_images)
            
            previous_position = episodes[i][-1]['pre_poses']
            move_distance = episodes[i][-1]['move_distance']
            AvgHeadingChange = episodes[i][-1]['avg_heading_changes']

            raw_poses = self.process_poses(poses=previous_position)

            
            if len(raw_poses) < 10 and len(raw_poses) > 0:
                last_pose = raw_poses[-1]
                raw_poses += [last_pose] * (10 - len(raw_poses))

            elif len(raw_poses) == 0:
                last_pose = [(self.start_position[i][0], self.start_position[i][1], self.start_position[i][2]), self.start_yaw[i]]
                raw_poses = [last_pose] * 10

            # 格式化为 prompt 字符串
            format_previous_position = "{\n" + "\n".join([f"    {p}," for p in raw_poses]) + "\n}"
            
            # 提取最后一个 pose 并存为结构化数据，便于后续使用
            if len(raw_poses) > 0:
                last_pose = raw_poses[-1]
                xyz = last_pose[0]  # (x, y, z)
                yaw = last_pose[1]
                self.current_poses[i] = [xyz[0], xyz[1], xyz[2], yaw]
            else:
                # fallback
                self.current_poses[i] = [self.start_position[i][0], self.start_position[i][1], 
                                         self.start_position[i][2], self.start_yaw[i]]

            x_min = int(math.floor(self.start_position[i][0] - 50))
            x_max = int(math.ceil(self.start_position[i][0] + 50))
            y_min = int(math.floor(self.start_position[i][1] - 50))
            y_max = int(math.ceil(self.start_position[i][1] + 50))
            if not fixed:
                conversation = [
                    {"role": "system", "content": self.unfixed_system_prompt},
                    {
                        "role": "user", 
                        "content": unfixed_user_prompt_template.format(
                            object_name=object_name, object_size=object_size, description=description,
                            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                            captions4=captions4, depth_info=depth_info,
                            format_previous_position=format_previous_position, step_num=step_num,
                            move_distance=move_distance, AvgHeadingChange=AvgHeadingChange
                        )
                    }
                ]
            else:
                conversation = [
                    {"role": "system", "content": self.fixed_system_prompt},
                    {
                        "role": "user", 
                        "content": fixed_user_prompt_template.format(
                            object_name=object_name, object_size=object_size, description=description,
                            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                            captions4=captions4, depth_info=depth_info,
                            format_previous_position=format_previous_position, step_num=step_num,
                            move_distance=move_distance, AvgHeadingChange=AvgHeadingChange
                        ) 
                    }
                ]
            prompt_info = conversation[1]["content"]
            user_prompts.append(prompt_info)
            inputs.append(conversation)

        return inputs, user_prompts
    
    async def unfixed_single_call(self, conversation):
        resp = await self.gpt_client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=conversation
        )
        text = resp.choices[0].message.content.strip()
        text = text.strip("[]`\"'")
        parts = [p.strip().strip("[]`\"'") for p in text.split(",")]
        action = parts[0].strip('\'"')
        # 解析 value
        value = float(parts[1]) if "." in parts[1] else int(parts[1])
        done = (action == 'stop')
        return action, value, done 
    
    async def fixed_single_call(self, conversation):
        
        resp = await self.gpt_client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=conversation
        )
        action = resp.choices[0].message.content.strip().strip('\'"')
        # 解析 value
        
        done = (action == 'stop')
        return action, 0, done              

    async def batch_calls(self, conversations, fixed):
        if fixed:
            tasks = [self.fixed_single_call(conv) for conv in conversations]
        else:
            tasks = [self.unfixed_single_call(conv) for conv in conversations]
        return await asyncio.gather(*tasks)

    def run(self, inputs, fixed, prompt_info_list=None):
        results = asyncio.run(self.batch_calls(inputs, fixed))
        actions, steps_size, predict_dones = zip(*results)

        new_actions, new_step_size = self.redirect_action(actions,steps_size, fixed)

        return list(new_actions), list(new_step_size), list(predict_dones)

    def process_depth(self, depth_images):
        depth_info = []
        for depth_image in depth_images:
            distance_image = np.array(depth_image) / 255.0 * 100
            x = torch.from_numpy(distance_image).unsqueeze(0).unsqueeze(0).float()
            y = -F.adaptive_max_pool2d(-x, (3, 3))
            y_np = y.squeeze().cpu().numpy()
            y_int = np.round(y_np).astype(int).tolist()
            depth_info.append(y_int)

        return depth_info 

    def process_poses(self, poses):
        pre_poses_xyzYaw = []
        for pose in poses:
            pos = pose['position']
            raw_quaternionr = pose['quaternionr']
            quaternionr = airsim.Quaternionr(
                x_val=raw_quaternionr[0], y_val=raw_quaternionr[1], 
                z_val=raw_quaternionr[2], w_val=raw_quaternionr[3]
            )
            pitch, roll, yaw = airsim.to_eularian_angles(quaternionr)
            yaw_degree = round(math.degrees(yaw), 2)

            # 结构化格式 [(x, y, z), yaw]
            formatted = [
                (round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)),
                yaw_degree
            ]
            pre_poses_xyzYaw.append(formatted)

        return pre_poses_xyzYaw

    def redirect_action(self, actions, step_size, fixed):
        new_actions = [None] * len(actions)
        new_step_size = list(step_size)
        for i, action in enumerate(actions):
            new_actions[i] = action
            try:
                start_position = self.start_position[i]
                x_min = round(start_position[0] - 50, 2)
                x_max = round(start_position[0] + 50, 2)
                y_min = round(start_position[1] - 50, 2)
                y_max = round(start_position[1] + 50, 2)

                current_pose = self.current_poses[i]
                x, y, z, yaw = current_pose

                if action == 'forward':
                    dx = math.cos(math.radians(yaw))
                    dy = math.sin(math.radians(yaw))
                    dz = 0

                    vector = np.array([dx, dy, dz])
                    norm = np.linalg.norm(vector)
                    if norm > 1e-6:
                        unit_vector = vector / norm
                    else:
                        unit_vector = np.array([0, 0, 0])
                    
                    if fixed:
                        new_position = np.array([x, y, z]) + unit_vector * AirsimActionSettings.FORWARD_STEP_SIZE
                    else:
                        new_position = np.array([x, y, z]) + unit_vector * step_size

                    if new_position[0] > x_max or new_position[0] < x_min or new_position[1] > y_max or new_position[1] < y_min:
                        new_actions[i] = 'rotl'
                        new_step_size[i] = 15
                        print(f"[INFO] Episode {i}: '{action}' would go out of bounds → replaced with '{new_actions[i]}'")


                elif action == "left":
                    unit_x = 1.0 * math.cos(math.radians(yaw + 90))
                    unit_y = 1.0 * math.sin(math.radians(yaw + 90))
                    vector = np.array([unit_x, unit_y, 0])

                    norm = np.linalg.norm(vector)
                    if norm > 1e-6:
                        unit_vector = vector / norm
                    else:
                        unit_vector = np.array([0, 0, 0])
                    
                    if fixed:
                        new_position = np.array([x, y, z]) - unit_vector * AirsimActionSettings.LEFT_RIGHT_STEP_SIZE
                    else:
                        new_position = np.array([x, y, z]) - unit_vector * step_size
                    
                    if new_position[0] > x_max or new_position[0] < x_min or new_position[1] > y_max or new_position[1] < y_min:
                        new_actions[i] = 'rotl'
                        new_step_size[i] = 15
                        print(f"[INFO] Episode {i}: '{action}' would go out of bounds → replaced with '{new_actions[i]}'")

                elif action == "right":
                    unit_x = 1.0 * math.cos(math.radians(yaw + 90))
                    unit_y = 1.0 * math.sin(math.radians(yaw + 90))
                    vector = np.array([unit_x, unit_y, 0])

                    norm = np.linalg.norm(vector)
                    if norm > 1e-6:
                        unit_vector = vector / norm
                    else:
                        unit_vector = np.array([0, 0, 0])
                    
                    if fixed:
                        new_position = np.array([x, y, z]) + unit_vector * AirsimActionSettings.LEFT_RIGHT_STEP_SIZE
                    else:
                        new_position = np.array([x, y, z]) + unit_vector * step_size

                    if new_position[0] > x_max or new_position[0] < x_min or new_position[1] > y_max or new_position[1] < y_min:
                        new_actions[i] = 'rotl'
                        new_step_size[i] = 15
                        print(f"[INFO] Episode {i}: '{action}' would go out of bounds → replaced with '{new_actions[i]}'")


                else:
                    new_actions[i] = action
                    new_step_size[i] = step_size[i]
                    continue  # 跳过不检查 ascend/descend/rotl/rotr/stop

            except Exception as e:
                print(f"[WARNING] run() failed to check bounds for episode {i}: {e}")
                # 不变更动作
                new_actions[i] = actions[i]
                new_step_size[i] = step_size[i]
        
        return new_actions, new_step_size
           
    
    # def turn_to_nearest_axis(self, dx, dy, yaw):
    #     def closest_signed_xy_axis(dx: float, dy: float):
    #         """
    #         找出 (dx,dy) 在 XY 平面里最接近的有向轴方向，并返回该方向和向该轴的最小夹角（度）。
    #         """
    #         L = math.hypot(dx, dy)
    #         if L == 0:
    #             raise ValueError("零向量没有方向")
    #         # 计算与四个方向的夹角（弧度）
    #         angles = {
    #             '+X': math.acos( dx / L),
    #             '-X': math.acos(-dx / L),
    #             '+Y': math.acos( dy / L),
    #             '-Y': math.acos(-dy / L),
    #         }
    #         # 选最小的
    #         axis, angle_rad = min(angles.items(), key=lambda kv: kv[1])
    #         return axis, math.degrees(angle_rad)
        
    #     axis, _ = closest_signed_xy_axis(dx, dy)
    #     target_yaws = { '+X':   0,
    #                     '+Y':  90,
    #                     '-X': 180,
    #                     '-Y': 270}
    #     target = target_yaws[axis]

    #     delta_r = (target - yaw + 360) % 360
    #     delta_l = (yaw - target + 360) % 360

    #     if delta_r <= delta_l:
    #         return 'rotr'
    #     else:
    #         return 'rotl'