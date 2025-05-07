from model_wrapper.base_model import BaseModelWrapper
from model_wrapper.generate_caption import generate_caption
#from model_wrapper.cogvlm import generate_caption
from openai import AsyncClient
from src.common.param import args

import numpy as np
import asyncio
import math
import airsim
import copy
import torch
import torch.nn.functional as F

class ONAir(BaseModelWrapper):
    def __init__(self, fixed):
        super().__init__()
        # self.caption_model = load_cogvlm()
        self.fixed = fixed
        self.client = AsyncClient()
        self.unfixed_system_prompt ="""# Prompt Header: Role & Rules  
            You are a UAV navigating a 3D outdoor environment. Follow the given task goal and interpret the multimodal inputs to decide the next action.  

            # Task Constraints
            At the beginning of each episode, you are deployed at a random initial pose \(P_0 = [x, y, z, \psi]\), where \(x, y, z\) are your 3D coordinates and \(\psi\) is your yaw (horizontal rotation).  
            You are equipped with RGB-D sensors facing four directions: **front**, **left**, **right**, and **down**. No GPS or global map is available—only egocentric observations.

            Your search is constrained to a **horizontal radius of 50 units** from the starting point.  
            You can execute **at most 150 actions**, including movements, rotations, and the `stop` command.

            Your navigation is considered **successful** if you stop within **20 units** of the target object.
            
            Following is input example:

            # Target Information  
            Target = [Name: Quercus robur; Size: mid(2*2=4 squares); Description: Organic irregular crown shape with lobate dark green leaves, fissured grey bark texture, and acorn fruits; trunk diameter suggesting mature growth stage.]

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
            - Valid directions: `forward`, `backward`, `left`, `right`, `ascend`, `descend`  
            - Distance is a positive number in **units**

            ## 2. Rotation:  
            - Format: `[<rotation>, <angle>]`  
            - Valid rotations: `rotl` (rotate left), `rotr` (rotate right)  
            - Angle is a positive number in **degrees**

            ## 3. Stop Condition:  
            - Use `[stop, 0]` when the target is **within 20 units** of the current UAV position.

            Only return one line with the chosen action. Do not include explanation or extra words.
            """
        
        self.fixed_system_prompt = """## Prompt Header: Role & Rules  
            You are a UAV navigating a 3D outdoor environment. Follow the given task goal and interpret the multimodal inputs to decide the next action.  

            # Task Constraints
            At the beginning of each episode, you are deployed at a random initial pose \(P_0 = [x, y, z, \psi]\), where \(x, y, z\) are your 3D coordinates and \(\psi\) is your yaw (horizontal rotation).  
            You are equipped with RGB-D sensors facing four directions: **front**, **left**, **right**, and **down**. No GPS or global map is available—only egocentric observations.

            Your search is constrained to a **horizontal radius of 50 units** from the starting point.  
            You can execute **at most 150 actions**, including movements, rotations, and the `stop` command.

            Your navigation is considered **successful** if you stop within **20 units** of the target object.

            # Target Information  
            Target = [Name: Quercus robur; Size: mid(2*2=4 squares); Description: Organic irregular crown shape with lobate dark green leaves, fissured grey bark texture, and acorn fruits; trunk diameter suggesting mature growth stage.]

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
            [10.0, 20.0, 5.0, 90],
            [11.0, 20.0, 5.0, 90],
            [11.0, 20.0, 5.0, 90],
            [12.0, 20.0, 5.0, 90],
            [12.0, 20.5, 5.0, 105],
            [13.0, 21.0, 5.0, 120],
            [13.0, 21.5, 5.0, 135],
            [14.0, 22.0, 5.0, 150],
            [14.0, 22.0, 5.0, 165]
            ]

            # Trajectory Summary
            StepsSoFar = 120
            DistanceTraveled = 45.8
            AvgHeadingChange = 7.3

            # Action Format Instruction  
            Based on the information above, decide the **next action** for the UAV.  

            You can only choose from the following **fixed action set**, divided into three categories:

            ## 1. Movement (Horizontal step size = 5 units, Vertical step size = 2 units):  
            - `"forward"`  
            - `"backward"`  
            - `"left"`  
            - `"right"`  
            - `"ascend"`  
            - `"descend"`  

            ## 2. Rotation (Each step rotates 15 degrees):  
            - `"rotl"` (rotate left 15°)  
            - `"rotr"` (rotate right 15°)  

            ## 3. Stop (Only choose when the target is within 20 units range):  
            - `"stop"`  

            Only return one line with the chosen action. Do not include explanation or extra words.
            """

    def prepare_inputs(self, episodes, fixed):
        inputs=[]
        for i in range(len(episodes)):
            sources=copy.deepcopy(episodes[i])
            images=[]
            depth_images=[]
            for src in sources[::-1]:
                if 'rgb' and 'depth' in src:
                    images.extend(src['rgb'])
                    depth_images.extend(src['depth'])
                    break
            #captions = generate_caption(images, args.generation_model_path)
            captions = generate_caption(image_files=images)
            #print(captions)
            step_num = episodes[i][-1]['step']
            description = episodes[i][-1]['description']
            depth_info = self.process_depth(depth_images=depth_images)
            previous_position = episodes[i][-1]['pre_poses']
            move_distance = episodes[i][-1]['move_distance']
            AvgHeadingChange = episodes[i][-1]['avg_heading_changes']
            format_previous_position = self.process_poses(poses=previous_position)
            if not fixed:
                conversation = [
                    {"role": "system", "content": self.unfixed_system_prompt},
                    {
                        "role": "user", 
                        "content": f"""# Target Information 
                                        Target = {description}

                                    # RGB Captions
                                        Front = {captions[0]}  
                                        Left = {captions[1]}  
                                        Right = {captions[2]}  
                                        Down = {captions[3]}  

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
                                        Target = {description}

                                    # RGB Captions
                                        Front = {captions[0]}  
                                        Left = {captions[1]}  
                                        Right = {captions[2]}  
                                        Down = {captions[3]}  

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

    def run(self, inputs, fixed):
        results = asyncio.run(self.batch_calls(inputs, fixed))
        actions, steps_size, predict_dones = zip(*results)
        return list(actions), list(steps_size), list(predict_dones)

    # def run_unfixed(self, inputs):
    #     actions=[]
    #     steps_size=[]
    #     predict_dones = []
    #     for conversation in inputs:
    #         response = self.client.chat.completions.create(
    #             model='gpt-4o-mini',
    #             messages=conversation
    #         )
    #         output = response.choices[0].message.content.strip().strip("[]")
    #         parts = [p.strip() for p in output.split(",")]
    #         action = parts[0].strip('\'"')
    #         if "." in parts[1]:
    #             value = float(parts[1])
    #         else:
    #             value = int(parts[1])
            
    #         step_size = value

    #         #predict done
    #         if action =='stop':
    #             predict_dones.append(True)
    #         else:
    #             predict_dones.append(False)

    #         actions.append(action)
    #         steps_size.append(step_size)

    #     return actions, steps_size, predict_dones
    

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
    
    def process_depth(self, depth_images):
        depth_info = []
        for depth_image in depth_images:
            distance_image = np.array(depth_image)/255.0 * 100
            x = torch.from_numpy(distance_image).unsqueeze(0).unsqueeze(0).float()
            y = F.adaptive_max_pool2d(x, (3,3))
            depth_info.append(y)

        return depth_info
    
    def process_poses(self, poses):
        pre_poses_xyzYaw = []
        for pose in poses:
            pos = pose['position']
            raw_quaternionr = pose['quaternionr']
            quaternionr = airsim.Quaternionr(x_val= raw_quaternionr[0], y_val=raw_quaternionr[1], 
                                             z_val=raw_quaternionr[2], w_val=raw_quaternionr[3])
            roll, pitch, yaw = airsim.to_eularian_angles(quaternionr)

            yaw_degree = math.degrees(yaw)
            pre_poses_xyzYaw.append([pos[0], pos[1], pos[2], yaw_degree])

        return pre_poses_xyzYaw