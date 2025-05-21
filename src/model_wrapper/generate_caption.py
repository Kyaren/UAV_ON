from openai import OpenAI 
import json
import airsim
import base64
import asyncio



def encode_image(image_file):
    base_img=[]
    for file in image_file:
        encoded_string = base64.b64encode(file).decode('utf-8')
        base_img.append(encoded_string)
    return base_img



async def single_generate_caption(image_files, client,temperature=0.7):
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

            User will supply four image; you must return exactly one well-structured description string for every image(no quotation marks).
            
            Output your response as a JSON-style list of four strings, like:
            [
                "Description of image 1.",
                "Description of image 2.",
                "Description of image 3.",
                "Description of image 4."
            ]
            Do not write anything else.

        """
    
    image_messages = []
    for b64 in image_files:
        image_messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
            }
        })

    
    response =await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":system_prompt },
            {"role": "user", "content":  image_messages}
        ],
        temperature=temperature
    )

    raw = response.choices[0].message.content.strip() 

    return raw

async def batch_generate_caption(image_files, client):
    tasks = []
    for image_file in image_files:
        tasks.append(single_generate_caption(image_file, client))

    return await asyncio.gather(*tasks)

def generate_caption(image_files, client):
    return asyncio.run(batch_generate_caption(image_files, client))

