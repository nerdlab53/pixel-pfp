import io
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Initialize the model (will download on first run)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

def generate_image(prompt, height=512, width=512):
    """Generate an image using Stable Diffusion and return it as bytes"""
    # Enhance prompt slightly for better results
    enhanced_prompt = f"{prompt}, detailed, high quality"
    
    # Generate the image
    image = pipe(enhanced_prompt, height=height, width=width, num_inference_steps=30).images[0]
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr.getvalue()