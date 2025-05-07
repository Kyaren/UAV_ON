from openai import OpenAI 
import json
import airsim
import base64



def encode_image(image_file):
    base_img=[]
    for file in image_file:
        encoded_string = base64.b64encode(file).decode('utf-8')
        base_img.append(encoded_string)
    return base_img



def generate_caption(image_files,temperature=0.8):
    gptclient = OpenAI()
    system_prompt = """
            System:
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
    image_data = encode_image(image_files)
    image_messages = []
    for b64 in image_data:
        image_messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
            }
        })

    
    response =gptclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":system_prompt },
            {"role": "user", "content": "请描述下面这些图片：", **{"content_type":"application/vnd.openai.multimodal", "content": image_messages}}  
        ],
        temperature=temperature
    )

    raw = response.choices[0].message.content.strip() 

    try:
        descriptions = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to decode JSON response: {raw}")
    
    return descriptions




# if __name__ == "__main__":
#     gptclient = OpenAI()
#     client= airsim.MultirotorClient()
#     client.confirmConnection()
#     client.armDisarm(True)
#     client.enableApiControl(True)


#     import numpy as np
#     Image=[]
#     response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
#     depth_img_in_meters = airsim.list_to_2d_float_array(response[0].image_data_float, response[0].width, response[0].height)
#     depth_image = (np.clip(depth_img_in_meters, 0, 100) / 100 * 255).astype(np.uint8)
    
#     Image.append(depth_image)
#     caption = process_depth(Image)
#     print(caption)