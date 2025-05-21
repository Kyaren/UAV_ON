import torch
import numpy as np

from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer 

device = 'cuda:2'
torch_type = torch.bfloat16
gen_kwargs = {
            "max_new_tokens": 64,
            "pad_token_id": 128002, # 使用tokenizer中定义的pad_token_id
            "top_k": 1,
    }
query = """<s>[INST] <<SYS>>
            You are an image understanding assistant. Your task is to generate one concise description per image that focuses on:
            1. The main objects present, using simple nouns .
            2. Each object’s approximate position or relationship .
            3. The overall scene type .
            4. Any clear semantic context.

            Use plain, factual language. Do not add opinions, judgments, or fine-grained part details.  
            User will supply an image; you must return exactly one description string (no quotes).  

            <</SYS>>
            User:
            Here is the image:
            <IMAGE>

            Assistant:
            [/INST]
        """




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
        load_in_4bit=True,
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

def generate_caption(model,tokenizer,responses):
    samples = []
    
        
    for resp in responses:
        pil_image = Image.open(BytesIO(resp)).convert("RGB")
        sample = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=[],
            images=[pil_image],
            template_version='chat',
        )
        samples.append(sample)

       
       
    batch = collate_fn(samples, tokenizer)
    batch = recur_move_to(batch, device, lambda x: isinstance(x, torch.Tensor))
    batch = recur_move_to(batch, torch.bfloat16, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))

    with torch.no_grad():
        out = model.generate(**batch, **gen_kwargs)
        out = out[:, batch['input_ids'].shape[1]:]
        decoded = tokenizer.batch_decode(out)
        # text = decoded[0].split("<|end_of_text|>")[0].strip("[]").strip('\'"')

    return [text.split("<|end_of_text|>")[0].strip("[]\"'") for text in decoded]