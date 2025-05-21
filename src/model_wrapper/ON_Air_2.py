from model_wrapper.base_model import BaseModelWrapper
from airsim_plugin.airsim_settings import AirsimActionSettings
#from model_wrapper.Qwen_api_captions_2 import generate_caption, encode_image
from model_wrapper.Qwen_api_captions import generate_caption, encode_image
from openai import AsyncClient
from io import BytesIO
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
    def __init__(self, fixed, batch_size):
        super().__init__()
        self.fixed = fixed
        self.gpt_client = AsyncClient()
        self.start_position = [[] for _ in range(batch_size)]
        self.start_yaw = [0 for _ in range(batch_size)]
        self.current_poses = [[] for _ in range(batch_size)]

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

            - Each horizontal movement (e.g., "forward", "left", "right") advances the UAV by exactly **5 units** in the heading direction.
            - If the **depth value** in the corresponding direction is **less than 5**, it indicates an obstacle is too close. Executing the movement will result in a **collision** and the navigation will be considered **failed**.

            You can execute at most 150 actions, including movement, rotation, and "stop".  
            Your navigation is considered successful only if you stop within 20 units of the target.

            # Altitude Adjustment Strategy  
            Your flying height (z) should be adapted based on the target size:  
            - For **small** targets, keep average DownDepth around **5.5** (i.e., fly at ~5–6m height).  
            - For **mid** targets, keep DownDepth around **7.5** (~7–8m height).  
            - For **large** targets, keep DownDepth around **9.5** (~9–10m height).  

            Use "ascend" or "descend" to adjust altitude when the average DownDepth is too high or too low compared to the desired value.  
            You must adjust your altitude **before searching for small targets**, or you might miss them.  
            Avoid staying at very high altitudes (DownDepth > 10) when looking for small or mid-sized targets.

            # Navigation Strategy Guidance  
            - Use "forward", "left", or "right" to move in the current heading direction. Moving forward is preferred when the front is not blocked.  
            - **Before executing any movement**, examine the **Depth map**. If any value in the direction is **less than 5**, do not move that way to avoid crashing.  
            - **If any value in the front direction is greater than 10, you must choose "forward"**. Not doing so is considered an invalid decision.  
            - Only use "rotl" or "rotr" when all directions are clearly blocked or unsafe.  
            - **Do not rotate more than 2 times in a row. If you rotate 3 times without moving, the mission will be judged as a failure.**  
            - Repeatedly rotating without exploration is strictly prohibited and will lead to navigation failure.  
            - Rotating when movement is possible is penalized. Avoid rotating just to wait.  
            - Use multiple rotations (e.g., 12 x "rotl") only when a full turn is needed due to blockage.  
            - Use "ascend" early to obtain better visual information; use "descend" when inspecting low objects.
            - Compare the **Target Name** and **Description** with the scene captions from each direction (Front, Left, Right, Down).  
            - If any direction contains visual elements that strongly match the Target (e.g., object type, material, color, function), prioritize moving toward that direction.  
            - For example, if the Target is a "Picnic Table", and the Front caption mentions a "picnic table", you must choose "forward".  
            - Continue to move in the matching direction until you are close to the target. Only stop when confident that you are within 20 units of it.
            - Failure to stop when the target is nearby will lead to **mission failure**.

            # Exploration Strategy  
            - Your goal is to **actively explore new areas** of the environment to find the target.  
            - Prioritize moving forward whenever possible.  
            - Do not rotate more than 2 times in a row.  
            - If you rotate 3 times in a row without exploring a new location, it will be judged as a **critical navigation failure**.  
            - Avoid staying in the same place. Always seek directions that allow forward movement.  
            - If the front is open (depth > 10), **you must move forward** instead of rotating.  
            - Use rotation only when **all directions are unsafe or blocked**.  
            - Failure to move when it is possible will be penalized.

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

            3. Stop (only if the target is clearly mentioned or visually present in the current scene):  
            - "stop" ← Use this as soon as you recognize the target object based on the RGB captions or semantic match with the target description.
            - Do not stop too early. **Stopping at a wrong object will lead to mission failure.**  
            - Remember: if any caption clearly mentions the target object (based on name or description), the correct action is to move toward that direction.

            Only return exactly one quoted string from the above list.  
            Do not output explanations, JSON, or natural language.
            """
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
            #-------------------- old version--------------------------------#
            # raw_poses = self.process_poses(poses=previous_position)
            # format_previous_position = "{\n" + "\n".join([f"    {p}," for p in raw_poses]) + "\n}"
            # ========== 提取轨迹格式 ==========
            raw_poses = self.process_poses(poses=previous_position)

            # ✅ 补齐不足 10 条的轨迹（复制最后一个）
            if len(raw_poses) < 10 and len(raw_poses) > 0:
                last_pose = raw_poses[-1]
                raw_poses += [last_pose] * (10 - len(raw_poses))

            # ✅ 防御性处理：如果为空，也补一个默认 pose
            elif len(raw_poses) == 0:
                last_pose = [(self.start_position[i][0], self.start_position[i][1], self.start_position[i][2]), self.start_yaw[i]]
                raw_poses = [last_pose] * 10

            # 格式化为 prompt 字符串
            format_previous_position = "{\n" + "\n".join([f"    {p}," for p in raw_poses]) + "\n}"
            # --------------------------------------------------------------#

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
            # --------------------------------------------------------------#

            x_min = int(math.floor(self.start_position[i][0] - 50))
            x_max = int(math.ceil(self.start_position[i][0] + 50))
            y_min = int(math.floor(self.start_position[i][1] - 50))
            y_max = int(math.ceil(self.start_position[i][1] + 50))
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
                        "content": f"""
                        # Target Information 
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

                        # Reminder  
                        - The target is **{object_name}**, described as: {description}  
                        - Compare the target description with all RGB captions (Front, Left, Right, Down).  
                        - The goal is to **explore the environment** and get **visually close to the target** before deciding to stop.  
                        - Do not rotate repeatedly. If you rotated in the last step, prioritize moving instead.  
                        - If any caption strongly matches the target name or description (e.g., object type, shape, color, function), and the visual evidence suggests that the object is **close and clearly visible**, you may consider stopping.  
                        - Avoid stopping immediately after a weak or partial match. Keep moving until the evidence is **strong and consistent**.  
                        - Stopping too early on a vague match is a **mission failure**. It's better to get closer and confirm than to stop prematurely.  
                        - **If StepsSoFar:{step_num} is more than 100 steps**, and the target seems to match any current caption reasonably well, you may relax your stopping condition and consider stopping based on a **single strong match**.
                        """
                    }
                ]
            # 保存 prompt 和相关信息，动作结果后面再补充
            
            # # 保存路径
            # save_dir = "./logs/prompt_debug_txt"
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = os.path.join(save_dir, f"episode_{i}.txt")

            # with open(save_path, "a") as f:  # 使用 "a" 模式打开文件以追加内容
            #     f.write(f"\n========== Step {step_num} ==========\n\n")
            #     f.write("[SYSTEM PROMPT]\n")
            #     f.write(conversation[0]["content"] + "\n\n")
            #     f.write("[USER PROMPT]\n")
            #     f.write(conversation[1]["content"] + "\n\n")
            
            prompt_info = conversation[1]["content"]
            user_prompts.append(prompt_info)
             #  保存到类属性中
            inputs.append(conversation)
        return inputs, user_prompts
    
    async def unfixed_single_call(self, conversation):
        resp = await self.gpt_client.chat.completions.create(
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
        
        resp = await self.gpt_client.chat.completions.create(
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

    # def run(self, inputs, fixed, prompt_info_list=None):
    #     results = asyncio.run(self.batch_calls(inputs, fixed))
    #     actions, steps_size, predict_dones = zip(*results)
    #     return list(actions), list(steps_size), list(predict_dones)

    #------------------------------- out of bound adjust-----------------------#
    def run(self, inputs, fixed, prompt_info_list=None):
        results = asyncio.run(self.batch_calls(inputs, fixed))
        actions, steps_size, predict_dones = zip(*results)

        new_actions = self.redirect_action(actions,steps_size, fixed)

        return list(new_actions), list(steps_size), list(predict_dones)
    #-----------------------------------------------------------------------#

    def process_depth(self, depth_images):
        depth_info = []
        for depth_image in depth_images:
            distance_image = np.array(depth_image) / 255.0 * 100
            x = torch.from_numpy(distance_image).unsqueeze(0).unsqueeze(0).float()
            y = F.adaptive_max_pool2d(x, (3, 3))
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

            # ✅ 结构化格式 [(x, y, z), yaw]
            formatted = [
                (round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)),
                yaw_degree
            ]
            pre_poses_xyzYaw.append(formatted)

        return pre_poses_xyzYaw

    def redirect_action(self, actions, step_size, fixed):
        new_actions = [None] * len(actions)

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
                        print(f"[INFO] Episode {i}: '{action}' would go out of bounds → replaced with '{new_actions[i]}'")


                else:
                    new_actions[i] = action
                    continue  # 跳过不检查 ascend/descend/rotl/rotr/stop

            except Exception as e:
                print(f"[WARNING] run() failed to check bounds for episode {i}: {e}")
                # 不变更动作
                new_actions[i] = actions[i]
        
        return new_actions
           
    
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