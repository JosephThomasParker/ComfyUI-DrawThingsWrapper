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
                "sampler": (
                    [
                        "UniPC",
                        "DPM++ 2M Karras",
                        "Euler Ancestral",
                        "DPM++ SDE Karras",
                        "PLMS",
                        "DDIM",
                        "LCM",
                        "Euler A Substep",
                        "DPM++ SDE Substep",
                        "TCD",
                        "DPM++ 2M Trailing",
                        "Euler A Trailing",
                        "DPM++ SDE Trailing",
                        "DDIM Trailing",
                        "DPM++ 2M AYS",
                        "Euler A AYS",
                        "DPM++ SDE AYS",
                    ],
                    {"default": "Euler A Trailing"},
                ),
                "steps": ("INT", {"default": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate_image"

    def generate_image(
        self, model, prompt, seed, width, height, guidance_scale, sampler, steps
    ):
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


def image_to_base64(image_tensor):
    # Convert the image tensor to a NumPy array and scale it to the range 0-255
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    # Save the image to a BytesIO object (in memory) rather than to a file
    buffered = BytesIO()
    img.save(buffered, format="PNG")

    # Encode the image as base64
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_string


def resize_for_inpainting(pixels, mask=None):
    x = (pixels.shape[1] // 64) * 64
    y = (pixels.shape[2] // 64) * 64
    # mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

    orig_pixels = pixels
    pixels = orig_pixels.clone()
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 64) // 2
        y_offset = (pixels.shape[2] % 64) // 2
        pixels = pixels[:, x_offset : x + x_offset, y_offset : y + y_offset, :]
        # pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset]
        # mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

    # m = (1.0 - mask.round()).squeeze(1)
    # for i in range(3):
    #    pixels[:,:,:,i] -= 0.5
    #    pixels[:,:,:,i] *= m
    #    pixels[:,:,:,i] += 0.5
    return pixels


def get_image_size(pixels):
    """
    Get image size from a size image, i.e. assumed input size is [H, W, C]
    """
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
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.5, "min": 0, "max": 25, "step": 0.1},
                ),
                "sampler": (
                    [
                        "UniPC",
                        "DPM++ 2M Karras",
                        "Euler Ancestral",
                        "DPM++ SDE Karras",
                        "PLMS",
                        "DDIM",
                        "LCM",
                        "Euler A Substep",
                        "DPM++ SDE Substep",
                        "TCD",
                        "DPM++ 2M Trailing",
                        "Euler A Trailing",
                        "DPM++ SDE Trailing",
                        "DDIM Trailing",
                        "DPM++ 2M AYS",
                        "Euler A AYS",
                        "DPM++ SDE AYS",
                    ],
                    {"default": "Euler A Trailing"},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150, "step": 1}),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate_image"

    def generate_image(
        self, images, model, prompt, seed, guidance_scale, sampler, steps, denoise
    ):
        # Call the Draw Things API
        api_url = "http://127.0.0.1:7860/sdapi/v1/img2img"

        encoded_images = []
        images_resized = resize_for_inpainting(images)
        for image_tensor in images_resized:
            encoded_images.append(image_to_base64(image_tensor))

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
            "strength": denoise,
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


class DrawThingsTxt2ImgPipeline:
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
                "sampler": (
                    [
                        "UniPC",
                        "DPM++ 2M Karras",
                        "Euler Ancestral",
                        "DPM++ SDE Karras",
                        "PLMS",
                        "DDIM",
                        "LCM",
                        "Euler A Substep",
                        "DPM++ SDE Substep",
                        "TCD",
                        "DPM++ 2M Trailing",
                        "Euler A Trailing",
                        "DPM++ SDE Trailing",
                        "DDIM Trailing",
                        "DPM++ 2M AYS",
                        "Euler A AYS",
                        "DPM++ SDE AYS",
                    ],
                    {"default": "Euler A Trailing"},
                ),
                "steps": ("INT", {"default": 20}),
            }
        }

    RETURN_TYPES = ("dict",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "generate_pipeline"

    def generate_pipeline(
        self, model, prompt, seed, width, height, guidance_scale, sampler, steps
    ):

        payload = {
            "generation_mode": "txt2img",
            "model": model,
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "sampler": sampler,
            "steps": steps,
        }

        return (payload,)


class DrawThingsPipelineAddCustom:
    def __init__(self):
        pass

    CATEGORY = "DrawThingsWrapper"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("dict", {"tooltip": "Draw Things pipeline"}),
                "field": ("STRING",),
                "value_type": (
                    ["STRING", "INT", "FLOAT"],
                    {"tooltip": "Choose the type of the value"},
                ),
                "value": ("STRING",),
            }
        }

    RETURN_TYPES = ("dict",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "add_to_pipeline"

    def add_to_pipeline(self, pipeline, field, value, value_type):

        if value_type == "INT":
            value = int(value)
        elif value_type == "FLOAT":
            value = float(value)
        elif value_type == "STRING":
            value = str(value)

        pipeline[field] = value

        return (pipeline,)


class DrawThingsGenerateFromPipeline:
    def __init__(self):
        pass

    CATEGORY = "DrawThingsWrapper"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("dict", {"tooltip": "Draw Things pipeline"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate_image"

    def generate_image(self, pipeline):

        # Cannot include generation_mode in payload, but need its value
        gen_mode = pipeline["generation_mode"]
        # Call the Draw Things API
        if gen_mode == "txt2img":
            api_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
        elif gen_mode == "img2img":
            api_url = "http://127.0.0.1:7860/sdapi/v1/img2img"

        payload = {
            key: value for key, value in pipeline.items() if key != "generation_mode"
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


NODE_CLASS_MAPPINGS = {
    "DrawThingsTxt2Img": DrawThingsTxt2Img,
    "DrawThingsImg2Img": DrawThingsImg2Img,
    "DrawThingsTxt2ImgPipeline": DrawThingsTxt2ImgPipeline,
    "DrawThingsPipelineAddCustom": DrawThingsPipelineAddCustom,
    "DrawThingsGenerateFromPipeline": DrawThingsGenerateFromPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DrawThingsTxt2Img": "Draw Things Txt2Img",
    "DrawThingsImg2Img": "Draw Things Img2Img",
    "DrawThingsTxt2ImgPipeline": "Draw Things Txt2Img Pipeline",
    "DrawThingsPipelineAddCustom": "Draw Things Pipeline Add Custom Field",
    "DrawThingsGenerateFromPipeline": "Draw Things Generate from Pipeline",
}
