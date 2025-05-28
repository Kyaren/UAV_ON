import base64
from io import BytesIO
import traceback
from PIL import Image
import json
import time
from openai import AsyncOpenAI
from dashscope import MultiModalConversation
import asyncio


def encode_image(image_files):

    base_img = []
    for file in image_files:
        encoded_string = base64.b64encode(file).decode("utf-8")
        base_img.append(encoded_string)
    return base_img


def generate_caption(image_file, temperature=0.7):
    n = len(image_file)
    system_prompt = f"""
            You are an image understanding assistant. Your task is to generate one concise and detailed description per image that focuses on:

            1. **Key objects and their core attributes**  
            - List every object with simple nouns **and** one or two attributes (e.g., “yellow slide, medium-sized, plastic”; “Springer Spaniel dog, tricolor coat”; “gas station building, metal canopy”).  
            
            2. **Object quantities and groupings**  
            - Specify number if more than one or if it’s a cluster (e.g., “three children”, “a row of parked cars”).  
            
            3. **Precise spatial relationships**  
            - Describe relative positions, distances or directions (e.g., “the dog sits immediately to the right of the slide”, “the building stands in the distant background, slightly left of center”).  
            
            4. **Object states or actions**  
            - Note any visible activity or condition (e.g., “children sliding down”, “pump nozzles hanging idle”, “car doors open”).  
                
            5. **Avoid opinions or irrelevancies**  
            - Use plain factual language. Do not include judgments, emotional tone words, or fine-grained internal part details.  
        
            User will supply {n} images; you must return exactly one well-structured description string for every image(no quotation marks).
            For each image, return exactly only one description item. 
            You are not allowed to return a blank caption string or more than {n} caption for {n} images.
            Return your answer **only** as a JSON array of caption strings (no prose outside the array), like:
            Example output exactly:
            [
                "yellow slide, medium-sized, plastic; slide is in the foreground; children sliding down",
                "......",
                "gas station building, metal canopy; two fuel pumps; pump nozzles hanging idle; canopy in the background"
            ]
            Do not write anything else.
            IMPORTANT INSTRUCTIONS:
            - You must respond **only in valid JSON format**.
            - Do **not** include any extra explanation, commentary, markdown formatting, or newlines.
            - The response must be a **single-line JSON object**, not a stringified object or list.
    """
    
    
    captions = []
    user_content = []
    for img_b64 in image_file:
            user_content.append({'image': 'data:image/png;base64,' + img_b64})
    messages = [{"role": "system",
                "content": [system_prompt]},
                {'role':'user','content': user_content}]
    
    try:
        response = MultiModalConversation.call(
            api_key="sk-c728ec5a2c98441282a8742a21d0cec6",
            model='qwen-vl-max',
            messages=messages
        )

        # 检查响应是否为空或结构异常
        if not response or "output" not in response:
            raise RuntimeError("DashScope API response is None or missing 'output'")

        # 尝试提取文本内容
        raw = response["output"]["choices"][0]["message"].content[0]["text"].strip()

        # 解析 JSON 格式
        captions = json.loads(raw)

        if len(captions) != len(image_file):
            raise ValueError(
                f"For batch of {len(image_file)} images, expected {len(image_file)} captions but got {len(captions)}:\n{captions}"
            )

        return captions

    except Exception as e:
        print("⚠️ [generate_caption] Failed to get or parse caption:")
        traceback.print_exc()
        # 返回默认描述（防止程序崩溃）
        return ["Description time out. [ServerError]"] * len(image_file)
    