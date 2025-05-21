from model_wrapper.base_model import BaseModelWrapper
#from model_wrapper.generate_caption import generate_caption
#from model_wrapper.cogvlm import generate_caption, load_model
from model_wrapper.qwenVL import generate_caption, load_model
from openai import AsyncClient

from src.common.param import args

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
    def __init__(self, fixed):
        super().__init__()
        # self.caption_model = load_cogvlm()
        self.fixed = fixed
        self.model, self.tokenizer = load_model(args.generation_model_path)
        self.client = AsyncClient()
        self.unfixed_system_prompt ="""# Prompt Header: Role & Rules  
            You are a UAV navigating a 3D outdoor environment. Follow the given task goal and interpret the multimodal inputs to decide the next action.  

            # Task Constraints
            At the beginning of each episode, you are deployed at a random initial pose \(P_0 = [x, y, z, \psi]\), where \(x, y, z\) are your 3D coordinates and \(\psi\) is your yaw (horizontal rotation).  
            You are equipped with RGB-D sensors facing four directions: **front**, **left**, **right**, and **down**. No GPS or global map is available—only egocentric observations.

            Your search is constrained to a **horizontal radius of 50 units** from the starting point.  
            You can execute **at most 150 actions**, including movements, rotations, and the `stop` command.

            You should search in a square area with a side length of 50 units, centered at the starting point. before your navigation you should estimate the next step position to avoid out of the search area.
            If next position will out of range, you should reject that action and give another action.
            Your navigation is considered **successful** if you stop within **20 units** of the target object.
            
            Following is input example:

            # Target Information  
            Target = [Name: Quercus robur; Size: mid(2*2=4 squares); Description: Organic irregular crown shape with lobate dark green leaves, fissured grey bark texture, and acorn fruits; trunk diameter suggesting mature growth stage.]

            # Search Area
            All positions are represented as 3D coordinates in the format: **(x, y, z)**  
            You must strictly stay within the following 2D navigation boundary (horizontal plane):
            **X Range**: [min_x, max_x] = [xx.x, xx.x]  
            **Y Range**: [min_y, max_y] = [yy.y, yy.y]

            # RGB Captions
            Front = "a narrow alley between buildings, a white fence on the side"  
            Left = "a low-rise building with red bricks"  
            Right = "a tree next to a small yard"  
            Down = "a tiled ground and a shadow of the UAV"  

            # Depth Information
            FrontDepth:
            [[88.3, 90.1, 92.7],    # upper-left, upper-center, upper-right
            [65.0, 70.2, 75.4],    # center-left, center, center-right
            [32.0, 40.8, 45.1]]    # lower-left, lower-center, lower-right

            LeftDepth:
            [[80.1, 85.3, 89.0],
            [60.2, 66.7, 71.9],
            [30.0, 38.5, 43.0]]

            RightDepth:
            [[82.7, 86.5, 91.2],
            [61.3, 69.8, 76.4],
            [34.1, 42.7, 49.0]]

            DownDepth:
            [[5.8, 6.2, 6.7],
            [7.1, 7.4, 7.9],
            [8.5, 9.0, 9.3]]

            # Previous UAV Poses (last 10 steps)
            [
            [10.0, 20.0, 5.0, 90],
            [10.5, 20.0, 5.0, 90],
            [11.0, 20.0, 5.0, 90],
            [11.5, 20.0, 5.0, 90],
            [12.0, 20.0, 5.0, 90],
            [12.5, 20.5, 5.0, 105],
            [13.0, 21.0, 5.0, 120],
            [13.5, 21.5, 5.0, 135],
            [14.0, 22.0, 5.0, 150],
            [14.5, 22.5, 5.0, 165]
            ]

            # Trajectory Summary
            StepsSoFar = 120
            DistanceTraveled = 45.8
            AvgHeadingChange = 7.3

            # Action Format Instruction  
            Based on the information above, decide the **next action** for the UAV.  

            The output should be in the format: `[action_type, value]`  
            Only return **one action** in **one line**, and **do not include any explanation**.

            ## 1. Movement:  
            - Format: `[<direction>, <distance>]`  
            - Valid directions: `forward`, `left`, `right`, `ascend`, `descend`  
            - Distance is a positive number in **units**

            ## 2. Rotation:  
            - Format: `[<rotation>, <angle>]`  
            - Valid rotations: `rotl` (rotate left), `rotr` (rotate right)  
            - Angle is a positive number in **degrees**

            ## 3. Stop Condition:  
            - Use `[stop, 0]` when the target is **within 20 units** of the current UAV position.

            Only return one line with the chosen action. Do not include explanation or extra words.
            """
        
        self.fixed_system_prompt = """
            ## Prompt Header: Role & Rules  
            You are a UAV navigating a 3D outdoor environment. Follow the given task goal and interpret the multimodal inputs to decide the next action.  

            # Coordinate System  
            All positions are represented in the format: (x, y, z)  
            - x: East-West axis  
            - y: North-South axis  
            - z: Altitude (vertical height)  
            - yaw: Horizontal heading angle in degrees (0 degrees = facing east)

            # Navigation Constraints  
            At the beginning of each episode, you are deployed at a random initial pose  
            P0 = (x0, y0, z0, yaw0)

            You must strictly stay within a fixed 2D horizontal search area:  
            - X Range: [-10.0, 40.0]  
            - Y Range: [15.0, 65.0]  
            - Z: no restriction

            Before taking any action, estimate your next position:  
            P_next = (x_current + delta_x, y_current + delta_y, z_current + delta_z)  
            If the next x or y position is outside the above range, do not execute the action.  
            Exceeding this boundary will result in navigation failure.

            You can execute at most 150 actions, including movement, rotation, and "stop".  
            Your navigation is considered successful only if you stop within 20 units of the target.

            # Navigation Strategy Guidance
            - Use "forward", "left", or "right" to move in the current heading direction. Moving forward is generally preferred when the front is not blocked.
            - Only use "rotl" or "rotr" when forward movement is clearly blocked or unsafe. Rotating without moving should be avoided unless absolutely necessary.
            - Use multiple rotations (e.g., 12 x "rotl") only when a complete turn is required to change direction after being blocked from all sides.
            - Use "ascend" early in the search to gain a better view of the environment.
            - Use "descend" when you are near the ground or need to inspect low-level objects.
            - Do not rotate repeatedly in place when movement is possible.

            # Exploration Strategy
            - Your goal is to explore as much of the environment as possible to find the target.
            - Prioritize using "forward" to cover new ground.
            - Do not rotate more than 2 times in a row unless you are clearly surrounded or blocked.
            - Avoid spinning or staying in the same place. Keep moving in open directions.
            - If the path ahead is visible (depth values > 10), move forward. Use rotation only if all paths are blocked or exploration is stuck.
            - Failure to explore new areas (e.g., rotating in circles without moving) may lead to mission failure.

            # Target Information  
            Target = [Name: Quercus robur; Size: mid(2x2=4 squares); Description: Organic irregular crown shape with lobate dark green leaves, fissured grey bark texture, and acorn fruits; trunk diameter suggesting mature growth stage.]

            # RGB Captions  
            Front = "a narrow alley between buildings, a white fence on the side"  
            Left = "a low-rise building with red bricks"  
            Right = "a tree next to a small yard"  
            Down = "a tiled ground and a shadow of the UAV"  

            # Depth Information  
            FrontDepth:  
            [[88.3, 90.1, 92.7],  
            [65.0, 70.2, 75.4],  
            [32.0, 40.8, 45.1]]  

            LeftDepth:  
            [[80.1, 85.3, 89.0],  
            [60.2, 66.7, 71.9],  
            [30.0, 38.5, 43.0]]  

            RightDepth:  
            [[82.7, 86.5, 91.2],  
            [61.3, 69.8, 76.4],  
            [34.1, 42.7, 49.0]]  

            DownDepth:  
            [[5.8, 6.2, 6.7],  
            [7.1, 7.4, 7.9],  
            [8.5, 9.0, 9.3]]  

            # Previous UAV Poses (last 10 steps)  
            # Format: [(x, y, z), yaw_angle_in_degrees]
            {
            [(10.0, 20.0, 5.0), 90],  
            [(10.0, 25.0, 5.0), 90],  
            [(10.0, 30.0, 5.0), 90],  
            [(15.0, 30.0, 5.0), 90],  
            [(15.0, 30.0, 7.0), 90],  
            [(15.0, 30.0, 7.0), 105],  
            [(15.0, 30.0, 7.0), 120],  
            [(15.0, 30.0, 7.0), 135],  
            [(15.0, 30.0, 7.0), 150],  
            [(20.0, 30.0, 7.0), 150]  
            }

            # Trajectory Summary  
            StepsSoFar = 120  
            DistanceTraveled = 45.8  
            AvgHeadingChange = 7.3

            # Action Format Instruction  
            Based on the information above, return only one valid action.

            You can only choose from the following action strings:

            1. Movement (horizontal = 5 units, vertical = 2 units):  
            - "forward"  
            - "left"  
            - "right"  
            - "ascend"  
            - "descend"  

            2. Rotation (15 degrees per step):  
            - "rotl"  
            - "rotr"  

            3. Stop (only if within 20 units of the target):  
            - "stop"

            Only return exactly one quoted string from the above list.  
            Do not output explanations, JSON, or natural language.

            """

    def prepare_inputs(self, episodes, fixed):
        inputs=[]
        images=[]
        depth_images=[]
        prompt_info_list = []  # 新增：记录 prompt 和 metadata
        for i in range(len(episodes)):
            sources=episodes[i]
            for src in sources[::-1]:
                if 'rgb' and 'depth' in src:
                    images.extend(src['rgb'])
                    depth_images.extend(src['depth'])
                    break
            
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
        start=time.time()
        for imgs in iterate_batches(images):
            caps = generate_caption(self.model, self.tokenizer, imgs)
            captions.extend(caps)
        print("generation captions time:", time.time()-start)

        current_pose = []

        for i in range(len(episodes)):
            
            captions4 = captions[4*i:4*i+4]
            #captions = generate_caption(image_files=images)
            #print(captions)
            
            start_position = episodes[i][-1]['start_position']

            # start_quaternionr = episodes[i][-1]['start_quaternionr']
            # pitch,roll,yaw = airsim.to_eularian_angles(start_quaternionr)

            step_num = episodes[i][-1]['step']
            description = episodes[i][-1]['description']
            object_name = episodes[i][-1]['object_name']
            object_size = episodes[i][-1]['object_size']
            depth_info = self.process_depth(depth_images=depth_images)
            previous_position = episodes[i][-1]['pre_poses']
            move_distance = episodes[i][-1]['move_distance']
            AvgHeadingChange = episodes[i][-1]['avg_heading_changes']
            #-------------------- old version--------------------------------#
            # raw_poses = self.process_poses(poses=previous_position)
            # format_previous_position = "{\n" + "\n".join([f"    {p}," for p in raw_poses]) + "\n}"
            # ========== 提取轨迹格式 ==========
            raw_poses = self.process_poses(poses=previous_position)
            # 格式化为 prompt 字符串
            format_previous_position = "{\n" + "\n".join([f"    {p}," for p in raw_poses]) + "\n}"

            # 提取最后一个 pose 并存为结构化数据，便于后续使用
            if len(raw_poses) > 0:
                last_pose = raw_poses[-1]
                xyz = last_pose[0]  # (x, y, z)
                yaw = last_pose[1]
                # 存入 episode 对象中，用于 run() 内部判断越界
                current_pose.append([xyz[0], xyz[1], xyz[2], yaw])
            else:
                # fallback
                episodes[i][-1]['current_pose'] = [start_position[0], start_position[1], start_position[2], 0.0]
                current_pose.append([start_position[0], start_position[1], start_position[2], 0.0])
            # --------------------------------------------------------------#

            x_min = round(start_position[0] - 50, 2)
            x_max = round(start_position[0] + 50, 2)
            y_min = round(start_position[1] - 50, 2)
            y_max = round(start_position[1] + 50, 2)
            if not fixed:
                conversation = [
                    {"role": "system", "content": self.unfixed_system_prompt},
                    {
                        "role": "user", 
                        "content": f"""# Target Information 
                                        Target = [Name:{object_name},
                                        Size:{object_size},
                                        Description:{description}]

                                    # RGB Captions
                                        Front = {captions4[0]}  
                                        Left = {captions4[1]}  
                                        Right = {captions4[2]}  
                                        Down = {captions4[3]}  

                                    # Search Area
                                    All positions are represented as 3D coordinates in the format: **(x, y, z)**  
                                    You must strictly stay within the following 2D navigation boundary (horizontal plane):
                                    **X Range**: [min_x, max_x] = [{x_min}, {x_max}]  
                                    **Y Range**: [min_y, max_y] = [{y_min}, {y_max}]

                                    # Depth Information
                                        FrontDepth:
                                        {depth_info[0]}

                                        LeftDepth:
                                        {depth_info[1]}

                                        RightDepth:
                                        {depth_info[2]}

                                        DownDepth:
                                        {depth_info[3]}

                                    # Previous UAV Poses (last 10 steps)
                                        {format_previous_position}

                                    # Trajectory Summary
                                        StepsSoFar = {step_num}
                                        DistanceTraveled = {move_distance}
                                        AvgHeadingChange = {AvgHeadingChange}
                                    """
                    }
                ]
            else:
                conversation = [
                    {"role": "system", "content": self.fixed_system_prompt},
                    {
                        "role": "user", 
                        "content": f"""# Target Information 
                                        Target = [Name:{object_name},
                                        Size:{object_size},
                                        Description:{description}]

                                        # Search Area
                                            All positions are represented as 3D coordinates in the format: **(x, y, z)**  
                                            You must strictly stay within the following 2D navigation boundary (horizontal plane):
                                            **X Range**: [min_x, max_x] = [{x_min}, {x_max}]  
                                            **Y Range**: [min_y, max_y] = [{y_min}, {y_max}]

                                        # RGB Captions
                                            Front = {captions4[0]}  
                                            Left = {captions4[1]}  
                                            Right = {captions4[2]}  
                                            Down = {captions4[3]}  

                                        # Depth Information
                                            FrontDepth:
                                            {depth_info[0]}

                                            LeftDepth:
                                            {depth_info[1]}

                                            RightDepth:
                                            {depth_info[2]}

                                            DownDepth:
                                            {depth_info[3]}

                                        # Previous UAV Poses (last 10 steps)
                                            {format_previous_position}

                                        # Trajectory Summary
                                            StepsSoFar = {step_num}
                                            DistanceTraveled = {move_distance}
                                            AvgHeadingChange = {AvgHeadingChange}
                                        """
                    }
                ]
            # 保存 prompt 和相关信息，动作结果后面再补充
            prompt_info = {
                "episode_id": i,
                "system": conversation[0]["content"],
                "user": conversation[1]["content"]
            }

            # 保存路径
            save_dir = "./prompt_debug_txt"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"episode_{i}.txt")

            with open(save_path, "a") as f:  # ✅ 用 a 模式追加，不覆盖
                f.write(f"\n========== Step {step_num} ==========\n\n")
                f.write("[SYSTEM PROMPT]\n")
                f.write(conversation[0]["content"] + "\n\n")
                f.write("[USER PROMPT]\n")
                f.write(conversation[1]["content"] + "\n\n")

            self.current_poses = current_pose  # 🔧 保存到类属性中
            inputs.append(conversation)
        return inputs
    
    async def unfixed_single_call(self, conversation):
        resp = await self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=conversation
        )
        text = resp.choices[0].message.content.strip().strip("[]")
        parts = [p.strip() for p in text.split(",")]
        action = parts[0].strip('\'"')
        # 解析 value
        value = float(parts[1]) if "." in parts[1] else int(parts[1])
        done = (action == 'stop')
        return action, value, done 
    
    async def fixed_single_call(self, conversation):
        
        resp = await self.client.chat.completions.create(
            model='gpt-4o-mini',
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
        return list(actions), list(steps_size), list(predict_dones)
    #------------------------------- out of bound adjust-----------------------#
    # def run(self, inputs, fixed, prompt_info_list=None):
    #     results = asyncio.run(self.batch_calls(inputs, fixed))
    #     actions, steps_size, predict_dones = zip(*results)

    #     new_actions = []

    #     # 从 prepare_inputs() 中保存的 current_pose 中读取
    #     for i, action in enumerate(actions):
    #         try:
    #             # 当前 pose：[x, y, z, yaw]，假设你之前 prepare_inputs 中 return 了它们
    #             current_pose = self.current_poses[i]
    #             x, y, z, yaw = current_pose

    #             # 只考虑水平动作：forward/left/right
    #             if action == "forward":
    #                 dx = 5 * math.cos(math.radians(yaw))
    #                 dy = 5 * math.sin(math.radians(yaw))
    #             elif action == "left":
    #                 dx = 5 * math.cos(math.radians(yaw + 90))
    #                 dy = 5 * math.sin(math.radians(yaw + 90))
    #             elif action == "right":
    #                 dx = 5 * math.cos(math.radians(yaw - 90))
    #                 dy = 5 * math.sin(math.radians(yaw - 90))
    #             else:
    #                 new_actions.append(action)
    #                 continue  # 跳过不检查 ascend/descend/rotl/rotr/stop

    #             x_next = x + dx
    #             y_next = y + dy

    #             # 从 prompt 中提取边界范围
    #             user_prompt = inputs[i][1]['content']
    #             x_min_str = user_prompt.split("**X Range**")[1].split("]")[0].split("[")[-1]
    #             y_min_str = user_prompt.split("**Y Range**")[1].split("]")[0].split("[")[-1]
    #             x_min, x_max = [float(v) for v in x_min_str.split(",")]
    #             y_min, y_max = [float(v) for v in y_min_str.split(",")]

    #             # 越界检查
    #             out_of_bounds = (x_next < x_min or x_next > x_max or
    #                             y_next < y_min or y_next > y_max)

    #             if out_of_bounds:
    #                 print(f"[INFO] Episode {i}: '{action}' would go out of bounds → replaced with 'rotl'")
    #                 action = "rotl"

    #         except Exception as e:
    #             print(f"[WARNING] run() failed to check bounds for episode {i}: {e}")
    #             # 不变更动作

    #         new_actions.append(action)

    #     return list(new_actions), list(steps_size), list(predict_dones)
    #-----------------------------------------------------------------------#

    # def run_fixed(self, inputs):
    #     actions=[]
    #     predict_dones = []
    #     for conversation in inputs:
    #         response = self.client.chat.completions.create(
    #             model='gpt-4o-mini',
    #             messages=conversation
    #         )
    #         action = response.choices[0].message.content.strip().strip('\'"')
    #         print("拿到的action:", action)
    #         actions.append(action)
            
    #         if action == 'stop':
    #             predict_dones.append(True)
    #         else:
    #             predict_dones.append(False)
            
    #     step_sizes = np.zeros(len(actions))
        
    #     return actions,step_sizes, predict_dones
    # def process_depth(self, depth_images):
    #     depth_info = []
    #     for depth_image in depth_images:
    #         distance_image = np.array(depth_image)/255.0 * 100
    #         x = torch.from_numpy(distance_image).unsqueeze(0).unsqueeze(0).float()
    #         y = F.adaptive_max_pool2d(x, (3,3))
    #         depth_info.append(y)

    #     return depth_info    
    
    def process_depth(self, depth_images):
        depth_info = []
        for depth_image in depth_images:
            distance_image = np.array(depth_image) / 255.0 * 100
            x = torch.from_numpy(distance_image).unsqueeze(0).unsqueeze(0).float()
            y = F.adaptive_max_pool2d(x, (3, 3))
            y_np = y.squeeze().cpu().numpy()
            y_int = np.round(y_np).astype(int).tolist()
            depth_info.append(y_int)

        return depth_info  # ✅ 注意：return 在循环外部

    def process_poses(self, poses):
        pre_poses_xyzYaw = []
        for pose in poses:
            pos = pose['position']
            raw_quaternionr = pose['quaternionr']
            quaternionr = airsim.Quaternionr(
                x_val=raw_quaternionr[0], y_val=raw_quaternionr[1], 
                z_val=raw_quaternionr[2], w_val=raw_quaternionr[3]
            )
            roll, pitch, yaw = airsim.to_eularian_angles(quaternionr)
            yaw_degree = round(math.degrees(yaw), 2)

            # ✅ 结构化格式 [(x, y, z), yaw]
            formatted = [
                (round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)),
                yaw_degree
            ]
            pre_poses_xyzYaw.append(formatted)

        return pre_poses_xyzYaw
