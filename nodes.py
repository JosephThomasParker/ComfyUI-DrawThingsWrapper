import base64
import numpy as np
import math
import requests
from PIL import Image
import io
import torch 

class DrawThingsWrapper:
    def __init__(self):
        pass


    CATEGORY = "DrawThingsWrapper"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 42}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "guidance_scale": ("FLOAT", {"default": 3.5}),
                "steps": ("INT", {"default": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate_image"

    def generate_image(self, prompt, seed, width, height, guidance_scale, steps):
        # Call the Draw Things API
        api_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"

        payload = {
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "steps": steps
        }

        response = requests.post(api_url, json=payload)

        # Raise an error if the request failed
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        # Process the images (assuming they are base64 encoded or raw binary data)
        images = []
        for img_data in data['images']:
            image_bytes = base64.b64decode(img_data)
            # Convert the image data to a Pillow Image object
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            # Convert to float32 tensor and normalize
            tensor_image = torch.from_numpy(image_np.astype(np.float32) / 255.0)
            images.append(tensor_image)
        return(torch.stack(images),)
    
NODE_CLASS_MAPPINGS = {
    "DrawThingsWrapper": DrawThingsWrapper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DrawThingsWrapper": "Draw Things Wrapper"
}
