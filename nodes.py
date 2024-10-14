import math
import requests
from PIL import Image
import io

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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"

    def generate_image(self, prompt, seed, width, height):
        # Call the Draw Things API
        api_url = "http://127.0.0.1:7860//sdapi/v1/txt2img"

        payload = {
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height
        }

        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            image_data = response.content  # Assuming the API returns raw image data
            image = Image.open(io.BytesIO(image_data))  # Convert raw data to an image object
            return (image,)  # Return as a tuple (ComfyUI expects output in tuple form)
        else:
            raise Exception(f"Failed to generate image: {response.text}")

# Register the node (depending on ComfyUI's mechanism, adjust accordingly)

    
NODE_CLASS_MAPPINGS = {
    "DrawThingsWrapper": DrawThingsWrapper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DrawThingsWrapper": "Draw Things Wrapper"
}
