import argparse
import cv2
import torch
import copy
import numpy as np

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer 

device = 'cuda:0'
torch_type = torch.bfloat16
query = """System:
            You are an image understanding assistant. Your task is to generate concise, factual scene descriptions for each of four input images. For each image, you must:

            1. List the main objects present, using simple descriptors (e.g., “yellow slide, Springer Spaniel dog, gas station building”).
            2. Specify the overall scene type (e.g., “playground, street, park, gas station”).
            3. Mention basic spatial arrangement if helpful (e.g., “the slide is in the foreground; the dog stands to its right”).
            4. Convey any clear semantic context (e.g., “looks like a children’s playground next to a service station”).

            Use plain, unembellished language. Do not describe fine-grained object parts or add opinions.

            User:
            I will provide four images. For each one, return exactly one description string. Output your response as a JSON-style list of four strings, like:

            [
                "Description of image 1.",
                "Description of image 2.",
                "Description of image 3.",
                "Description of image 4."
            ]

            Images:  
            Image 1: <IMAGE_1>  
            Image 2: <IMAGE_2>  
            Image 3: <IMAGE_3>  
            Image 4: <IMAGE_4>  

            Assistant:
        """
gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
    }



def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_type,
        trust_remote_code=True,
        device_map=device,
        # load_in_4bit=True,
        # low_cpu_mem_usage=True
    ).eval()
    return model, tokenizer

def recur_move_to(item, tgt, criterion_func):
    if criterion_func(item):
        device_copy = item.to(tgt)
        return device_copy
    elif isinstance(item, list):
        return [recur_move_to(v, tgt, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
    else:
        return item

def collate_fn(features, tokenizer) -> dict:
    images = [feature.pop('images', None) for feature in features if 'images' in feature]
    tokenizer.pad_token = tokenizer.eos_token
    max_length = max(len(feature['input_ids']) for feature in features)

    def pad_to_max_length(feature, max_length):
        padding_length = max_length - len(feature['input_ids'])
        feature['input_ids'] = torch.cat([feature['input_ids'], torch.full((padding_length,), tokenizer.pad_token_id)])
        feature['token_type_ids'] = torch.cat([feature['token_type_ids'], torch.zeros(padding_length, dtype=torch.long)])
        feature['attention_mask'] = torch.cat([feature['attention_mask'], torch.zeros(padding_length, dtype=torch.long)])
        if feature['labels'] is not None:
            feature['labels'] = torch.cat([feature['labels'], torch.full((padding_length,), tokenizer.pad_token_id)])
        else:
            feature['labels'] = torch.full((max_length,), tokenizer.pad_token_id)
        return feature

    features = [pad_to_max_length(feature, max_length) for feature in features]
    batch = {
        key: torch.stack([feature[key] for feature in features])
        for key in features[0].keys()
    }

    if images:
        batch['images'] = images

    return batch

def generate_caption(responses, model_path):
    model, tokenizer=load_model(model_path=model_path)
    pil_imgs = []
    for resp in responses:
        arr = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        img = arr.reshape(resp.height, resp.width, 3)

        img = np.flipud(img)
        pil_imgs.append(Image.fromarray(img))

    # 2) 构造单样本输入
    sample = model.build_conversation_input_ids(
        tokenizer,
        query=query,
        history=[],
        images=pil_imgs,
        template_version='chat'
    )

    batch = collate_fn([sample], tokenizer)
    batch = recur_move_to(batch, 'cuda:0', lambda x: isinstance(x, torch.Tensor))
    batch = recur_move_to(batch, torch.bfloat16, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))

    with torch.no_grad():
        out = model.generate(**batch, **gen_kwargs)
        out = out[:, batch['input_ids'].shape[1]:]
        decoded = tokenizer.batch_decode(out)
        text = decoded[0].split("<|end_of_text|>")[0].strip()

    return text