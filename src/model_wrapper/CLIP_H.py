from model_wrapper.base_model import BaseModelWrapper
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

import io
import torch
import numpy as np

class CLIP_H(BaseModelWrapper):
    def __init__(self):
        self.device = 'cuda:0'
        self.model, self.processor = self.load_clip()
        self.action_mapping ={
            0 : 'forward',
            1 : 'left',
            2 : 'right',
            3 : 'descend',
        }
        self.threshold = 0.225


    
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
            pil_images = [Image.open(io.BytesIO(img)) for img in images[4*i:4*i+4]]   
            descriptions=episodes[i][-1]['description']
            proc_input = self.processor(text=descriptions, images=pil_images, return_tensors="pt", padding=True)
            proc_inputs = {k: v.to(self.device) for k, v in proc_input.items()}        
            inputs.append(proc_inputs)
        return inputs, user_prompts,depths

    def run(self, input, depth):
        actions = []
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
                max_val, max_idx = torch.max(cos_sim, dim=0)
                action = self.action_mapping.get(int(max_idx.item()), None)
                sim_value = max_val.item()
                if sim_value >= self.threshold:
                    action = "stop"
                if processed_depth[i]<=5 and action == "descend":
                    secend_val, secend_idx = cos_sim.topk(2, dim=0)
                    action = self.action_mapping.get(int(secend_idx[1].item()), None)
                    sim_value = secend_val[1].item()
                
                actions.append(action)

                done = (action == "stop")
                dones.append(done)
        return actions, np.zeros(len(actions)), dones
    
    def process_depth(self, depth_images):
        depth_info = []
        for depth_image in depth_images:
            distance_image = np.array(depth_image) / 255.0 * 100.0
            nearest_dist = np.min(distance_image).astype(int)
            depth_info.append(nearest_dist)

        return depth_info