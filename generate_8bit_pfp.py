#!/usr/bin/env python3
import io
import torch
import argparse
import time
import os
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm
from colorama import init, Fore, Style
import subprocess
import tempfile
import numpy as np
from sklearn.cluster import KMeans
from pyxelate import Pyx, Pal
from skimage import io as skio
import random

# Initialize colorama for cross-platform colored terminal output
init()

# ASCII art logo
LOGO = f"""
{Fore.CYAN}╔═════════════════════════════════════════════╗
║ {Fore.GREEN}░█▀▀█░█▀▄░▀█▀░▀█▀  {Fore.MAGENTA}█▀█░█▀▀░█▀█  {Fore.YELLOW}█▀▀░█▀▀░█▀█ ║
║ {Fore.GREEN}░█▀▀█░█▀▄░░█░░░█░  {Fore.MAGENTA}█▀▀░█▀▀░█▀▀  {Fore.YELLOW}█░█░█▀▀░█░█ ║
║ {Fore.GREEN}░▀▀▀▀░▀▀▀░▀▀▀░░▀░  {Fore.MAGENTA}▀░░░▀░░░▀░░  {Fore.YELLOW}▀▀▀░▀▀▀░▀░▀ ║
{Fore.CYAN}╚═════════════════════════════════════════════╝
{Style.RESET_ALL}"""

# Import the Rust module if available, otherwise use a Python fallback
try:
    import rust_8bit
    use_rust = True
    print(f"{Fore.GREEN}✓ Using Rust implementation for 8-bit conversion{Style.RESET_ALL}")
except ImportError:
    use_rust = False
    print(f"{Fore.YELLOW}⚠ Rust module not found, using Python fallback for 8-bit conversion{Style.RESET_ALL}")

# Add this function to track elapsed time
_start_time = time.time()
def elapsed_time():
    """Return the elapsed time since the module was imported"""
    return time.time() - _start_time

def spinner_context(desc="Processing"):
    """Create a spinner for operations without progress reporting"""
    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spinner_idx = 0
    start_time = time.time()
    
    print(f"{desc}... ", end="", flush=True)
    
    try:
        while True:
            print(f"\r{desc}... {Fore.CYAN}{spinner_chars[spinner_idx]}{Style.RESET_ALL}", end="", flush=True)
            spinner_idx = (spinner_idx + 1) % len(spinner_chars)
            yield
            time.sleep(0.1)
    finally:
        elapsed = time.time() - start_time
        print(f"\r{desc}... {Fore.GREEN}Done!{Style.RESET_ALL} ({elapsed:.2f}s)")

def generate_with_stable_diffusion(prompt, height=512, width=512, model_name="stablediffusionapi/bluepencil-xl-v5"):
    """Generate an image using HuggingFace's diffusers"""
    print(f"{Fore.BLUE}🖌️ Generating image for prompt: {prompt}{Style.RESET_ALL}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{Fore.BLUE}💻 Using device: {device}{Style.RESET_ALL}")
    
    # Show a spinner during model loading
    spinner = spinner_context("Loading pipeline components...")
    next(spinner)
    
    # Record start time for model loading
    load_start_time = time.time()
    
    # Model loading parameters
    kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "cache_dir": os.environ.get("HF_HOME"),
        "safety_checker": None
    }
    
    try:
        # Use the simpler DiffusionPipeline approach for all models
        from diffusers import DiffusionPipeline
        
        # First try loading with quantization
        try:
            print(f"{Fore.BLUE}Attempting to load quantized model...{Style.RESET_ALL}")
            # Try loading with 8-bit quantization first
            from bitsandbytes.nn import Linear8bitLt
            pipe = DiffusionPipeline.from_pretrained(
                model_name, 
                device_map="auto",
                load_in_8bit=True,
                **kwargs
            )
            print(f"{Fore.GREEN}✓ Loaded 8-bit quantized model{Style.RESET_ALL}")
        except (ImportError, ValueError, Exception) as e:
            print(f"{Fore.YELLOW}⚠ 8-bit quantization not available: {str(e)}{Style.RESET_ALL}")
            
            # Try 4-bit quantization if 8-bit failed
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                pipe = DiffusionPipeline.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    **kwargs
                )
                print(f"{Fore.GREEN}✓ Loaded 4-bit quantized model{Style.RESET_ALL}")
            except (ImportError, ValueError, Exception) as e:
                print(f"{Fore.YELLOW}⚠ 4-bit quantization not available: {str(e)}{Style.RESET_ALL}")
                
                # Fall back to standard loading without quantization
                print(f"{Fore.BLUE}Falling back to standard model loading...{Style.RESET_ALL}")
                pipe = DiffusionPipeline.from_pretrained(model_name, **kwargs)
                pipe = pipe.to(device)
        
        # Enable attention slicing for memory efficiency
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        
        # Enable xformers if available and using CUDA
        if device == "cuda":
            try:
                import xformers
                if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                    pipe.enable_xformers_memory_efficient_attention()
                    print(f"{Fore.GREEN}✓ XFormers enabled{Style.RESET_ALL}")
            except ImportError:
                print(f"{Fore.YELLOW}⚠ XFormers not available{Style.RESET_ALL}")
        
        spinner.close()
        # Calculate the loading time
        load_time = time.time() - load_start_time
        print(f"{Fore.GREEN}Loading model... Done! ({load_time:.2f}s){Style.RESET_ALL}")
        
        # Generate the image
        generator = torch.Generator(device=device).manual_seed(random.randint(0, 2147483647))
        
        # Enhance the prompt for better results
        enhanced_prompt = f"{prompt}, detailed, high quality"
        print(f"{Fore.BLUE}🔮 Processing prompt: {enhanced_prompt}{Style.RESET_ALL}")
        
        # Show a spinner during image generation
        spinner = spinner_context("Generating")
        next(spinner)
        
        # Generate image using a consistent API
        image = pipe(
            prompt=enhanced_prompt,
            height=height,
            width=width,
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=generator,
        ).images[0]
        
        # Convert the PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        return img_bytes, image
        
    except Exception as e:
        spinner.close()
        print(f"{Fore.RED}Error generating image: {str(e)}{Style.RESET_ALL}")
        raise
    finally:
        spinner.close()

def convert_to_8bit(image, palette_size=64, dithering=True):
    """Convert an image to 8-bit style with limited color palette"""
    print(f"{Fore.BLUE}🎨 Converting to 8-bit style with {palette_size} colors{Style.RESET_ALL}")
    
    # Try to use Rust implementation first (faster)
    try:
        if hasattr(rust_8bit, 'convert_to_8bit'):
            print(f"{Fore.GREEN}Using Rust implementation for 8-bit conversion{Style.RESET_ALL}")
            return rust_8bit.convert_to_8bit(image, palette_size, dithering)
        else:
            # Check what functions are available in the module
            available_functions = [f for f in dir(rust_8bit) if not f.startswith('_')]
            print(f"{Fore.YELLOW}Note: 'convert_to_8bit' not found in rust_8bit module.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Available functions: {available_functions}{Style.RESET_ALL}")
            # Try alternative function name if it exists
            if 'pixelate' in available_functions:
                print(f"{Fore.GREEN}Using 'pixelate' function from Rust implementation{Style.RESET_ALL}")
                return rust_8bit.pixelate(image, palette_size, dithering)
    except ImportError:
        print(f"{Fore.YELLOW}Rust implementation not available. Using Python fallback.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}Error with Rust implementation: {str(e)}. Using Python fallback.{Style.RESET_ALL}")
    
    # Python implementation (slower but always available)
    print(f"{Fore.BLUE}Using Python implementation for 8-bit conversion{Style.RESET_ALL}")
    
    # Convert to numpy array if it's a PIL image
    if isinstance(image, Image.Image):
        np_image = np.array(image)
    else:
        np_image = image
        
    # Ensure RGB format
    if len(np_image.shape) == 2:  # Grayscale
        np_image = np.stack((np_image,) * 3, axis=-1)
    
    # Step 1: Resize the image to create a pixelated effect (optional)
    # We'll leave the original size for now

    # Step 2: Quantize the colors to create a limited palette
    h, w, c = np_image.shape
    pixels = np_image.reshape(-1, c)
    
    # Use K-means to find the most representative colors
    kmeans = KMeans(n_clusters=palette_size, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    palette = kmeans.cluster_centers_.astype(np.uint8)
    
    # Apply dithering if requested
    if dithering:
        # Create a quantized image without dithering first
        quantized = palette[labels].reshape(h, w, c)
        
        # Apply Floyd-Steinberg dithering
        dithered = np.copy(np_image).astype(np.float32)
        
        for y in range(h):
            for x in range(w):
                old_pixel = dithered[y, x].copy()
                # Find the closest color in the palette
                closest_idx = np.argmin(np.sum((palette - old_pixel) ** 2, axis=1))
                new_pixel = palette[closest_idx]
                dithered[y, x] = new_pixel
                
                # Calculate the error
                error = old_pixel - new_pixel
                
                # Distribute the error to neighboring pixels
                if x + 1 < w:
                    dithered[y, x + 1] += error * 7/16
                if x - 1 >= 0 and y + 1 < h:
                    dithered[y + 1, x - 1] += error * 3/16
                if y + 1 < h:
                    dithered[y + 1, x] += error * 5/16
                if x + 1 < w and y + 1 < h:
                    dithered[y + 1, x + 1] += error * 1/16
        
        # Clip values and convert back to uint8
        result = np.clip(dithered, 0, 255).astype(np.uint8)
    else:
        # Without dithering, just apply the palette
        result = palette[labels].reshape(h, w, c)
    
    # Convert back to PIL image
    return Image.fromarray(result)

def pixelate(image, pixel_size=8, palette_size=64, dithering=False):
    """Apply a pixelated effect to an image"""
    print(f"{Fore.BLUE}🎨 Pixelating image with pixel size {pixel_size} and {palette_size} colors{Style.RESET_ALL}")
    
    # Convert to PIL Image if it's not already
    if not isinstance(image, Image.Image):
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        else:
            image = Image.fromarray(image)
    
    # Get original size
    width, height = image.size
    
    # Calculate new dimensions that are divisible by pixel_size
    new_width = width - (width % pixel_size)
    new_height = height - (height % pixel_size)
    
    if new_width != width or new_height != height:
        # Crop to make dimensions divisible by pixel_size
        image = image.crop((0, 0, new_width, new_height))
        width, height = new_width, new_height
    
    # Calculate dimensions for smaller image
    small_width = width // pixel_size
    small_height = height // pixel_size
    
    # Resize down to create pixelation effect
    small_image = image.resize((small_width, small_height), Image.BILINEAR)
    
    # Resize back up with nearest neighbor to maintain pixelated look
    pixelated = small_image.resize((width, height), Image.NEAREST)
    
    # Apply color quantization if palette_size is specified
    if palette_size and palette_size < 256:
        # Convert to RGB mode if it's not already
        if pixelated.mode != "RGB":
            pixelated = pixelated.convert("RGB")
        
        # Use PIL's quantize for color reduction
        if dithering:
            # With dithering
            pixelated = pixelated.quantize(colors=palette_size, method=2).convert('RGB')
        else:
            # Without dithering
            pixelated = pixelated.quantize(colors=palette_size, method=0).convert('RGB')
    
    return pixelated

def convert_to_8bit_pyxelate(image, downsample_factor=8, palette_size=7, dithering=True):
    """Convert an image to 8-bit style using Pyxelate"""
    print(f"{Fore.BLUE}🎨 Converting to 8-bit style with Pyxelate (downsample: {downsample_factor}, colors: {palette_size}){Style.RESET_ALL}")
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        # Convert PIL image to numpy array
        np_image = np.array(image)
    elif isinstance(image, bytes):
        # Convert bytes to PIL image then to numpy array
        np_image = np.array(Image.open(io.BytesIO(image)))
    else:
        # Assume it's already a numpy array
        np_image = image
    
    try:
        # Create Pyxelate transformer with proper dithering parameter
        # For Pyxelate, dithering should be a string or None, not a boolean
        dither_algo = "floyd" if dithering else None
        
        pyx = Pyx(factor=downsample_factor, 
                palette=palette_size, 
                dither=dither_algo)
        
        # Fit to learn color palette
        print(f"{Fore.BLUE}Learning color palette...{Style.RESET_ALL}")
        pyx.fit(np_image)
        
        # Transform the image
        print(f"{Fore.BLUE}Transforming image...{Style.RESET_ALL}")
        pixelated_image = pyx.transform(np_image)
        
        # Convert back to PIL Image for further processing
        return Image.fromarray(pixelated_image)
    
    except Exception as e:
        print(f"{Fore.RED}Error during Pyxelate conversion: {str(e)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Falling back to simple pixelation...{Style.RESET_ALL}")
        # Fall back to the simpler pixelation method if Pyxelate fails
        return pixelate(image, pixel_size=downsample_factor, palette_size=palette_size, dithering=dithering)

def generate_with_candle(prompt, height=512, width=512, model_id="stablediffusionapi/bluepencil-xl-v5"):
    """Generate an image using Candle"""
    print(f"{Fore.BLUE}🦀 Generating image with Candle for prompt:{Style.RESET_ALL} '{prompt}'")
    
    # Create a temporary file to store the output image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    # Show a spinner during model generation
    spinner = spinner_context(f"Running Candle Stable Diffusion {model_id}")
    next(spinner)
    
    try:
        # Prepare the command to run Candle's stable-diffusion example
        cmd = [
            "cargo", "run", 
            "--manifest-path", os.path.join(os.environ.get("CANDLE_DIR", "./candle"), "Cargo.toml"),
            "--example", "stable-diffusion", 
            "--release", "--", 
            "--cpu",  # Explicitly use CPU mode
            "--prompt", prompt,
            "--height", str(height),
            "--width", str(width),
            "--final-image", output_path,
        ]
        
        # Add SD3 specific arguments
        if model_id == "stablediffusionapi/bluepencil-xl-v5":
            cmd.extend(["--sd-version", "3"])
            # SD3 can use SDXL weights path format
            cmd.extend(["--model-id", "stabilityai/stable-diffusion-3-medium-diffusers"])
            
        print(f"{Fore.YELLOW}Running command: {' '.join(cmd)}{Style.RESET_ALL}")
        
        # Run the command and capture all output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Collect stdout for debugging
        stdout_lines = []
        for line in process.stdout:
            stdout_lines.append(line)
            if "step" in line.lower() and "progress" in line.lower():
                print(f"\r{Fore.CYAN}{line.strip()}{Style.RESET_ALL}", end="", flush=True)
        
        # Wait for the process to complete
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            print(f"{Fore.RED}Command failed. Full output:{Style.RESET_ALL}")
            for line in stdout_lines:
                print(f"  {line.strip()}")
            print(f"{Fore.RED}Error output:{Style.RESET_ALL}")
            print(f"  {stderr}")
            print(f"{Fore.YELLOW}Falling back to PyTorch implementation...{Style.RESET_ALL}")
            return generate_with_stable_diffusion(prompt, height, width)
        
        # Check if the file was created
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"{Fore.RED}Candle did not create output image at {output_path}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Falling back to PyTorch implementation...{Style.RESET_ALL}")
            return generate_with_stable_diffusion(prompt, height, width)
            
    finally:
        spinner.close()
    
    try:
        # Load the generated image
        with open(output_path, 'rb') as f:
            img_bytes = f.read()
        
        # Load as PIL Image for further processing
        pil_image = Image.open(io.BytesIO(img_bytes))
        
        return img_bytes, pil_image
    except Exception as e:
        print(f"{Fore.RED}Failed to load generated image: {str(e)}{Style.RESET_ALL}")
        # Fall back to using stable diffusion
        print(f"{Fore.YELLOW}Falling back to PyTorch implementation...{Style.RESET_ALL}")
        return generate_with_stable_diffusion(prompt, height, width)

def generate_8bit_pfp(prompt, output_path="8bit_pfp.png", palette_size=7, dithering=True, model_id="stablediffusionapi/bluepencil-xl-v5", downsample_factor=8):
    """Generate an 8-bit style profile picture from a text prompt"""
    print(LOGO)
    
    print(f"{Fore.YELLOW}Current directory: {os.getcwd()}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Looking for candle in: {os.path.join(os.getcwd(), 'candle')}{Style.RESET_ALL}")
    
    # First check if environment variable is set
    candle_dir = os.environ.get("CANDLE_DIR")
    print(f"{Fore.YELLOW}CANDLE_DIR set to: {candle_dir}{Style.RESET_ALL}")
    
    # If not set, check if candle exists in the current directory
    if not candle_dir and os.path.exists(os.path.join(os.getcwd(), 'candle')):
        candle_dir = os.path.join(os.getcwd(), 'candle')
        print(f"{Fore.GREEN}✓ Found candle directory at: {candle_dir}{Style.RESET_ALL}")
    
    if not candle_dir:
        print(f"{Fore.YELLOW}⚠ CANDLE_DIR environment variable not set. Using default PyTorch implementation.{Style.RESET_ALL}")
        sd_image_bytes, pil_image = generate_with_stable_diffusion(prompt)
    else:
        if not os.path.exists(candle_dir):
            print(f"{Fore.YELLOW}⚠ Candle directory not found at {candle_dir}. Using default PyTorch implementation.{Style.RESET_ALL}")
            sd_image_bytes, pil_image = generate_with_stable_diffusion(prompt)
        else:
            # Use Candle for image generation
            sd_image_bytes, pil_image = generate_with_candle(prompt, model_id=model_id)
    
    # Save the original image for comparison if requested
    base_filename, ext = os.path.splitext(output_path)
    original_path = f"{base_filename}_original{ext}"
    print(f"{Fore.BLUE}💾 Saving original image to:{Style.RESET_ALL} {original_path}")
    pil_image.save(original_path)
    
    # Convert the image to 8-bit style using Pyxelate
    print(f"{Fore.BLUE}🎮 Converting to 8-bit style...{Style.RESET_ALL}")
    
    # Show a spinner during the conversion
    spinner = spinner_context("Converting")
    next(spinner)
    
    try:
        # Use Pyxelate for 8-bit conversion
        eight_bit_image = convert_to_8bit_pyxelate(
            pil_image, 
            downsample_factor=downsample_factor, 
            palette_size=palette_size, 
            dithering=dithering
        )
        
        # Save the output to the specified path
        print(f"{Fore.BLUE}💾 Saving 8-bit image to:{Style.RESET_ALL} {output_path}")
        eight_bit_image.save(output_path)
        
        return output_path
    finally:
        spinner.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an 8-bit style profile picture")
    parser.add_argument("prompt", help="Text description of the image you want to generate")
    parser.add_argument("--output", "-o", default="8bit_pfp.png", help="Output file path (default: 8bit_pfp.png)")
    parser.add_argument("--palette-size", "-p", type=int, default=7, help="Number of colors in the palette (2-256)")
    parser.add_argument("--dithering", "-d", action="store_true", help="Apply dithering for texture")
    parser.add_argument("--model", type=str, default="stablediffusionapi/bluepencil-xl-v5", 
                        help="Model to use for generation (default: stablediffusionapi/bluepencil-xl-v5)")
    parser.add_argument("--downsample-factor", "-f", type=int, default=8, 
                        help="Factor to downsample the image by (default: 8)")
    
    args = parser.parse_args()
    
    # Validate palette size
    if args.palette_size < 2 or args.palette_size > 256:
        parser.error("Palette size must be between 2 and 256")
    
    # Validate downsample factor
    if args.downsample_factor < 1:
        parser.error("Downsample factor must be at least 1")
    
    try:
        # Generate the image
        output_path = generate_8bit_pfp(
            args.prompt, 
            output_path=args.output,
            palette_size=args.palette_size,
            dithering=args.dithering,
            model_id=args.model,
            downsample_factor=args.downsample_factor
        )
        
        print(f"\n{Fore.GREEN}✨ Done! 8-bit image saved to:{Style.RESET_ALL} {output_path}")
        print(f"{Fore.YELLOW}👀 Original image saved to:{Style.RESET_ALL} {os.path.splitext(output_path)[0]}_original{os.path.splitext(output_path)[1]}")
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}⚠ Process interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {str(e)}{Style.RESET_ALL}")