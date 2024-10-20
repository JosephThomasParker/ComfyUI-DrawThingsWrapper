#!/usr/bin/env python3
"""
Wrapper nodes for calling Draw Things from ComfyUI
"""

import base64
import numpy as np
import requests
from PIL import Image
import io
from io import BytesIO
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


def image_to_base64_with_alpha(image_tensor):
    # Check if the image tensor has an alpha channel
    has_alpha = image_tensor.shape[-1] == 4
    
    # Convert the image tensor to a NumPy array and scale it to the range 0-255
    i = 255. * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    # Ensure the image is in RGBA format if it has an alpha channel
    if has_alpha:
        print("has_alpha")
        img = img.convert("RGBA")
    else:
        print("no_alpha")
        img = img.convert("RGB")

    # Save the image to a BytesIO object (in memory) rather than to a file
    buffered = BytesIO()
    img.save(buffered, format="PNG")

    # Encode the image as base64
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_string


def image_to_base64(image_tensor):
    # Convert the image tensor to a NumPy array and scale it to the range 0-255
    i = 255. * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    # Save the image to a BytesIO object (in memory) rather than to a file
    buffered = BytesIO()
    img.save(buffered, format="PNG")

    # Encode the image as base64
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_string

def mask_to_base64(mask_tensor):
    # Convert the image tensor to a NumPy array and scale it to the range 0-255
    i = 255. * mask_tensor.squeeze(0).cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8), mode='L')

    # Save the image to a BytesIO object (in memory) rather than to a file
    buffered = BytesIO()
    img.save(buffered, format="PNG")

    # Encode the image as base64
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_string

def resize_for_inpainting(pixels, mask=None):
    print(type(pixels))
    x = (pixels.shape[1] // 64) * 64
    y = (pixels.shape[2] // 64) * 64
    if mask != None:
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

    orig_pixels = pixels
    pixels = orig_pixels.clone()
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 64) // 2
        y_offset = (pixels.shape[2] % 64) // 2
        pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
        if mask != None:
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

    # Add an alpha channel if the image doesn't have one
    if pixels.shape[-1] == 3:  # If RGB, convert to RGBA
        alpha_channel = torch.ones((pixels.shape[0], pixels.shape[1], pixels.shape[2], 1), dtype=pixels.dtype)
        pixels = torch.cat([pixels, alpha_channel], dim=-1)

    # Apply the mask to create transparency in the alpha channel
    if mask is not None:
        m = (1.0 - mask.round()).squeeze(1)  # Binary mask
        pixels[:, :, :, 3] *= m  # Modify alpha channel based on mask

#    if mask != None:
#        m = (1.0 - mask.round()).squeeze(1)
#        for i in range(3):
#            pixels[:,:,:,i] -= 0.5
#            pixels[:,:,:,i] *= m
#            pixels[:,:,:,i] += 0.5
    return pixels, mask

def get_image_size(pixels):
    """
       Get image size from a size image, i.e. assumed input size is [H, W, C]
    """
    print(type(pixels))
    print(np.shape(pixels))
    x = (pixels.shape[0] // 64) * 64
    y = (pixels.shape[1] // 64) * 64
    return x, y

class DrawThingsImg2Img:
    def __init__(self):
        pass

    CATEGORY = "DrawThingsWrapper"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "input image"}),
                "model": ("STRING", {"default": "flux_1_dev_q8p.ckpt"}),
                "prompt": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 42}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0, "max": 25, "step": 0.1}),
                "sampler": (["UniPC","DPM++ 2M Karras","Euler Ancestral", "DPM++ SDE Karras", "PLMS", "DDIM", "LCM", "Euler A Substep", "DPM++ SDE Substep", "TCD", "DPM++ 2M Trailing", "Euler A Trailing", "DPM++ SDE Trailing", "DDIM Trailing", "DPM++ 2M AYS", "Euler A AYS", "DPM++ SDE AYS"], {"default": "Euler A Trailing"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150, "step": 1}),
            },
            "optional": {
                "optional_mask": ("MASK", {"tooltip": "inpainting mask"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate_image"

    def generate_image(self, images, model, prompt, seed, width, height, guidance_scale, sampler, steps, optional_mask=None):
        # Call the Draw Things API
        api_url = "http://127.0.0.1:7860/sdapi/v1/img2img"

        encoded_images = []
        images_resized, mask_resized  = resize_for_inpainting(images, optional_mask)
        for image_tensor in images_resized:
            #encoded_images.append(image_to_base64_2(image_tensor, True))
            encoded_images.append(image_to_base64_with_alpha(image_tensor))

        height, width = get_image_size(images_resized[0])


        payload = {
            "model": model,
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "sampler": sampler,
            "steps": steps,
            "init_images": encoded_images,
        }

        #if mask_resized != None:
        #    #payload["mask"] = mask_to_base64(mask_resized[0])
        #    #payload["masks"] = mask_to_base64(mask_resized[0])
        #    payload["init_masks"] = mask_to_base64(mask_resized[0])

        response = requests.post(api_url, json=payload)

        data = response.json()
        print(data)
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
