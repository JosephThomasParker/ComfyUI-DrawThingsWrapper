import math

class DrawThingsWrapper:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["SD","SDXL"],),
                "aspect_ratio_width": ("INT",{
                    "default": 1,
                    "step":1,
                    "display": "number"
                }),
                "aspect_ratio_height": ("INT",{
                    "default": 1,
                    "step":1,
                    "display": "number"
                })
            }
        }

    RETURN_TYPES = ("INT","INT")
    RETURN_NAMES = ("Width", "Height")

    FUNCTION = "run"

    CATEGORY = "DrawThingsWrapper"

    def run(self, model_type, aspect_ratio_width, aspect_ratio_height):
        # Define the total pixel counts for SD and SDXL
        total_pixels = {
            'SD': 512 * 512,
            'SDXL': 1024 * 1024
        }
        
        # Calculate the number of total pixels based on model type
        pixels = total_pixels.get(model_type, 0)
        
        # Calculate the aspect ratio decimal
        aspect_ratio_decimal = aspect_ratio_width / aspect_ratio_height
        
        # Calculate width and height
        width = math.sqrt(pixels * aspect_ratio_decimal)
        height = pixels / width
        
        # Return the width and height as a tuple of integers
        return (int(round(width)), int(round(height)))
    
NODE_CLASS_MAPPINGS = {
    "DrawThingsWrapper": DrawThingsWrapper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DrawThingsWrapper": "Draw Things Wrapper"
}
