from model_wrapper.base_model import BaseModelWrapper
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from airsim_plugin.airsim_settings import AirsimActionSettings

import airsim
import io
import torch
import numpy as np
import math

class CLIP_H(BaseModelWrapper):
    def __init__(self,batch_size):
        self.device = 'cuda:0'
        self.model, self.processor = self.load_clip()
        self.action_mapping ={
            0 : 'forward',
            1 : 'left',
            2 : 'right',
            3 : 'descend',
        }
        self.threshold = 0.24
        self.start_position = [[] for _ in range(batch_size)]
        self.start_yaw = [0 for _ in range(batch_size)]
        self.current_poses = [[] for _ in range(batch_size)]
        self.prev_action = [ None for _ in range(batch_size)]
    
    def load_clip(self,):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        return model.to(self.device), processor

    def prepare_inputs(self, episodes):
        images = []
        depths = []
        inputs = []
        user_prompts = [[] for _ in range(len(episodes))]
        for i in range(len(episodes)):
            sources = episodes[i]
            for src in sources[::-1]:
                if 'rgb' in src and 'depth' in src:
                    for img in src['rgb']:
                        images.append(img)
                    depths.append(src['depth'])        
                    break 
        
        for i in range(len(episodes)):
            self.start_position[i] = episodes[i][-1]['start_position']

            previous_position = episodes[i][-1]['pre_poses']
            raw_poses = self.process_poses(poses=previous_position)

            
            if len(raw_poses) < 10 and len(raw_poses) > 0:
                last_pose = raw_poses[-1]
                raw_poses += [last_pose] * (10 - len(raw_poses))

            
            elif len(raw_poses) == 0:
                last_pose = [(self.start_position[i][0], self.start_position[i][1], self.start_position[i][2]), self.start_yaw[i]]
                raw_poses = [last_pose] * 10

            if len(raw_poses) > 0:
                last_pose = raw_poses[-1]
                xyz = last_pose[0]  # (x, y, z)
                yaw = last_pose[1]
                self.current_poses[i] = [xyz[0], xyz[1], xyz[2], yaw]
            else:
                # fallback
                self.current_poses[i] = [self.start_position[i][0], self.start_position[i][1], 
                                         self.start_position[i][2], self.start_yaw[i]]


            pil_images = [Image.open(io.BytesIO(img)) for img in images[4*i:4*i+4]]   
            descriptions=episodes[i][-1]['description']
            proc_input = self.processor(text=descriptions, images=pil_images, return_tensors="pt", padding=True)
            proc_inputs = {k: v.to(self.device) for k, v in proc_input.items()}        
            inputs.append(proc_inputs)
        return inputs, user_prompts,depths

    def run(self, input, depth):
        actions = []
        conflict = [False] * len(input)
        dones = []
        processed_depth = self.process_depth(depth)
        for i in range(len(input)):
            
            with torch.no_grad():
                outputs = self.model(**input[i])
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.squeeze(1)

                img_feats = outputs.image_embeds      # shape [n_images, dim]
                txt_feats = outputs.text_embeds       # shape [n_texts, dim]
                # 先做 L2 归一化
                img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)
                txt_feats = txt_feats / txt_feats.norm(dim=1, keepdim=True)
                # 然后点乘
                cos_sim = img_feats @ txt_feats.T    # shape [n_images, n_texts], in [-1,1]

                print("probs:",cos_sim)
                val, idx = cos_sim.topk(4, dim=0)
                action = self.action_mapping.get(int(idx[0].item()), None)
                sim_value = val[0].item()
                if sim_value >= self.threshold:
                    action = 'stop'

                prev_action = self.prev_action[i]

                if(prev_action == 'left' and action == 'right') or (prev_action == 'right' and action == 'left'):
                    action = self.action_mapping.get(int(idx[1].item()), None)
                    sim_value = val[1].item()
                    conflict[i]=True
                if processed_depth[i]<=5 and action == 'descend':
                    if conflict[i]:                    
                        action = self.action_mapping.get(int(idx[2].item()), None)
                        sim_value = val[2].item()
                    else:
                        action = self.action_mapping.get(int(idx[1].item()), None)
                        sim_value = val[1].item()
                    
                if sim_value >= self.threshold:
                    action = 'stop'
                
                new_action = self.redirect_action(action, i)
                self.prev_action[i] = new_action
                actions.append(new_action)

                done = (action == 'stop')
                dones.append(done)
        return actions, np.zeros(len(actions)), dones
    
    def process_depth(self, depth_images):
        depth_info = []
        for depth_image in depth_images:
            distance_image = np.array(depth_image) / 255.0 * 100.0
            nearest_dist = np.min(distance_image).astype(int)
            depth_info.append(nearest_dist)

        return depth_info
    
    def redirect_action(self, action, i):
        
        new_action = action
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
                
                new_position = np.array([x, y, z]) + unit_vector * AirsimActionSettings.FORWARD_STEP_SIZE
                    
                if new_position[0] > x_max or new_position[0] < x_min or new_position[1] > y_max or new_position[1] < y_min:
                    new_action = 'rotl'
                    print(f"[INFO] Episode {i}: '{action}' would go out of bounds → replaced with '{new_action}'")


            elif action == "left":
                unit_x = 1.0 * math.cos(math.radians(yaw + 90))
                unit_y = 1.0 * math.sin(math.radians(yaw + 90))
                vector = np.array([unit_x, unit_y, 0])

                norm = np.linalg.norm(vector)
                if norm > 1e-6:
                    unit_vector = vector / norm
                else:
                    unit_vector = np.array([0, 0, 0])
                    
                    
                new_position = np.array([x, y, z]) - unit_vector * AirsimActionSettings.LEFT_RIGHT_STEP_SIZE                    
                    
                if new_position[0] > x_max or new_position[0] < x_min or new_position[1] > y_max or new_position[1] < y_min:
                    new_action = 'rotl'
                    print(f"[INFO] Episode {i}: '{action}' would go out of bounds → replaced with '{new_action}'")

            elif action == "right":
                unit_x = 1.0 * math.cos(math.radians(yaw + 90))
                unit_y = 1.0 * math.sin(math.radians(yaw + 90))
                vector = np.array([unit_x, unit_y, 0])

                norm = np.linalg.norm(vector)
                if norm > 1e-6:
                    unit_vector = vector / norm
                else:
                    unit_vector = np.array([0, 0, 0])
                    
                    
                new_position = np.array([x, y, z]) + unit_vector * AirsimActionSettings.LEFT_RIGHT_STEP_SIZE
                   
                if new_position[0] > x_max or new_position[0] < x_min or new_position[1] > y_max or new_position[1] < y_min:
                    new_action = 'rotl'
                    
                    print(f"[INFO] Episode {i}: '{action}' would go out of bounds → replaced with '{new_action}'")

            else:
                new_action = action

        except Exception as e:
            print(f"[WARNING] run() failed to check bounds for episode {i}: {e}")
            # 不变更动作
            new_action = action
                   
        return new_action
    
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