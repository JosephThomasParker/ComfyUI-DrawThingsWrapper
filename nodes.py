#!/usr/bin/env python3
"""
Wrapper nodes for calling Draw Things from ComfyUI
"""

import base64
import numpy as np
import requests
from PIL import Image
import io
import torch


class DrawThingsTxt2Img:
    def __init__(self):
        pass

    CATEGORY = "DrawThingsWrapper"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "flux_1_dev_q8p.ckpt"}),
                "prompt": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 42}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "guidance_scale": ("FLOAT", {"default": 3.5}),
                "sampler": (["UniPC","DPM++ 2M Karras","Euler Ancestral", "DPM++ SDE Karras", "PLMS", "DDIM", "LCM", "Euler A Substep", "DPM++ SDE Substep", "TCD", "DPM++ 2M Trailing", "Euler A Trailing", "DPM++ SDE Trailing", "DDIM Trailing", "DPM++ 2M AYS", "Euler A AYS", "DPM++ SDE AYS"], {"default": "Euler A Trailing"}),
                "steps": ("INT", {"default": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate_image"

    def generate_image(self, model, prompt, seed, width, height, guidance_scale, sampler, steps):
        # Call the Draw Things API
        api_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"

        payload = {
            "model": model,
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "sampler": sampler,
            "steps": steps,
        }

        response = requests.post(api_url, json=payload)

        # Raise an error if the request failed
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        # Process the images (assuming they are base64 encoded or raw binary data)
        images = []
        for img_data in data["images"]:
            image_bytes = base64.b64decode(img_data)
            # Convert the image data to a Pillow Image object
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            # Convert to float32 tensor and normalize
            tensor_image = torch.from_numpy(image_np.astype(np.float32) / 255.0)
            images.append(tensor_image)
        return (torch.stack(images),)

class DrawThingsImg2Img:
    def __init__(self):
        pass

    CATEGORY = "DrawThingsWrapper"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "flux_1_dev_q8p.ckpt"}),
                "prompt": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 42}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "guidance_scale": ("FLOAT", {"default": 3.5}),
                "sampler": (["UniPC","DPM++ 2M Karras","Euler Ancestral", "DPM++ SDE Karras", "PLMS", "DDIM", "LCM", "Euler A Substep", "DPM++ SDE Substep", "TCD", "DPM++ 2M Trailing", "Euler A Trailing", "DPM++ SDE Trailing", "DDIM Trailing", "DPM++ 2M AYS", "Euler A AYS", "DPM++ SDE AYS"], {"default": "Euler A Trailing"}),
                "steps": ("INT", {"default": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate_image"

    def generate_image(self, model, prompt, seed, width, height, guidance_scale, sampler, steps):
        # Call the Draw Things API
        api_url = "http://127.0.0.1:7860/sdapi/v1/img2img"

        payload = {
            "model": model,
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "sampler": sampler,
            "steps": steps,
        }

#        response = requests.post(api_url, json=payload)
#
#        # Raise an error if the request failed
#        response.raise_for_status()
#
#        # Parse the JSON response
#        data = response.json()
#        print("Dia duit!")
#        #print(data)
#        print(type(data))
#        print(type(payload))

        # Path to your PNG image file
        image_path = "/Users/jparker/data/sd_outputs/.people/marbro/inputs/1_512sq.JPG"

        # Read the image and encode it as base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        payload["init_images"] = [encoded_string]
        #print(payload)

        response = requests.post(api_url, json=payload)

        #data = response.json()
        #print(data)
        # Raise an error if the request failed
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()
        print(data)

        # Process the images (assuming they are base64 encoded or raw binary data)
        images = []
        for img_data in data["images"]:
            image_bytes = base64.b64decode(img_data)
            # Convert the image data to a Pillow Image object
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            # Convert to float32 tensor and normalize
            tensor_image = torch.from_numpy(image_np.astype(np.float32) / 255.0)
            images.append(tensor_image)
        return (torch.stack(images),)


NODE_CLASS_MAPPINGS = {
  "DrawThingsTxt2Img": DrawThingsTxt2Img,
  "DrawThingsImg2Img": DrawThingsImg2Img,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "DrawThingsTxt2Img": "Draw Things Txt2Img", 
  "DrawThingsImg2Img": "Draw Things Img2Img",
}
