import base64
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

device = 'cuda:0'
torch_type = torch.bfloat16

def load_model(model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype = torch_type,
        device_map = device
    )

    processer = AutoProcessor.from_pretrained(model_path)

    return model, processer

def base64_to_image(img_file):
    base64_imgs=[]
    for file in img_file:
        encode_string = base64.b64encode(file).decode('utf-8')
        base64_imgs.append(encode_string)
    return base64_imgs

def generate_caption(model,processor,responses):

    image_data = base64_to_image(responses)
    system_prompt = """
            You are an image understanding assistant. Your task is to generate one concise but detailed description per image that focuses on:

            1. **Key objects and their core attributes**  
            - List each main object with simple nouns **and** one or two attributes (e.g., “yellow slide, medium-sized, plastic”; “Springer Spaniel dog, tricolor coat”; “gas station building, metal canopy”).  
            
            2. **Object quantities and groupings**  
            - Specify number if more than one or if it’s a cluster (e.g., “three children”, “a row of parked cars”).  
            
            3. **Precise spatial relationships**  
            - Describe relative positions, distances or directions (e.g., “the dog sits immediately to the right of the slide”, “the building stands in the distant background, slightly left of center”).  
            
            4. **Object states or actions**  
            - Note any visible activity or condition (e.g., “children sliding down”, “pump nozzles hanging idle”, “car doors open”).  
                
            5. **Avoid opinions or irrelevancies**  
            - Use plain factual language. Do not include judgments, emotional tone words, or fine-grained internal part details.  

            User will supply an image; you must return exactly one well-structured description string (no quotation marks).

    """
    messages=[]
    for i in range(len(image_data)):
        message = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "data:image;base64," + image_data[i]}
                ]
            }
        ]
        messages.append(message)

    texts = [
        processor.apply_chat_template(msg, tokenize = False, add_generation_prompt=True) for msg in messages
    ]

    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(output_texts)
    return output_texts