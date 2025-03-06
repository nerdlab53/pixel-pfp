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

# Initialize colorama for cross-platform colored terminal output
init()

# ASCII art logo
LOGO = f"""
{Fore.CYAN}╔═══════════════════════════════════════╗
║ {Fore.GREEN}░█▀▀░█▀▄░▀█▀░▀█▀  {Fore.MAGENTA}█▀█░█▀▀░█▀█  {Fore.YELLOW}█▀▀░█▀▀░█▀█ ║
║ {Fore.GREEN}░█▀▀░█▀▄░░█░░░█░  {Fore.MAGENTA}█▀▀░█▀▀░█▀▀  {Fore.YELLOW}█░█░█▀▀░█░█ ║
║ {Fore.GREEN}░▀▀▀░▀░▀░░▀░░░▀░  {Fore.MAGENTA}▀░░░▀░░░▀░░  {Fore.YELLOW}▀▀▀░▀▀▀░▀░▀ ║
{Fore.CYAN}╚═══════════════════════════════════════╝
{Style.RESET_ALL}"""

# Import the Rust module if available, otherwise use a Python fallback
try:
    import rust_8bit
    use_rust = True
    print(f"{Fore.GREEN}✓ Using Rust implementation for 8-bit conversion{Style.RESET_ALL}")
except ImportError:
    use_rust = False
    print(f"{Fore.YELLOW}⚠ Rust module not found, using Python fallback for 8-bit conversion{Style.RESET_ALL}")

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

def generate_with_stable_diffusion(prompt, height=512, width=512):
    """Generate an image using Stable Diffusion"""
    print(f"{Fore.BLUE}🖌️ Generating image for prompt:{Style.RESET_ALL} '{prompt}'")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{Fore.BLUE}💻 Using device:{Style.RESET_ALL} {device}")
    
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Show a spinner during model loading
    spinner = spinner_context("Loading Stable Diffusion model")
    next(spinner)
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device)
    finally:
        spinner.close()
    
    # Generate the image with a progress bar
    enhanced_prompt = f"{prompt}, detailed, high quality"
    print(f"{Fore.BLUE}🔮 Processing prompt:{Style.RESET_ALL} {enhanced_prompt}")
    
    progress_bar = tqdm(total=30, desc="Generating", unit="steps", bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    def callback_fn(step, timestep, latents):
        progress_bar.update(1)
        return None
    
    image = pipe(enhanced_prompt, height=height, width=width, 
                 num_inference_steps=30, callback=callback_fn, 
                 callback_steps=1).images[0]
    
    progress_bar.close()
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr.getvalue(), image

def convert_to_8bit_python(image, palette_size, dithering):
    """Python fallback for 8-bit conversion if Rust module is not available"""
    print(f"{Fore.BLUE}🎮 Converting to 8-bit style using Python...{Style.RESET_ALL}")
    
    with tqdm(total=3, desc="Converting", unit="steps") as progress_bar:
        # Create a smaller version to pixelate
        small_size = 64
        pixelated = image.resize((small_size, small_size), Image.NEAREST)
        progress_bar.update(1)
        
        # Resize back to original size with nearest neighbor for pixelated look
        pixelated = pixelated.resize(image.size, Image.NEAREST)
        progress_bar.update(1)
        
        # Convert to a limited palette (simplified algorithm)
        pixelated = pixelated.convert('P', palette=Image.ADAPTIVE, colors=palette_size)
        
        # Convert back to RGB
        pixelated = pixelated.convert('RGB')
        progress_bar.update(1)
    
    # Return the pixelated image
    img_byte_arr = io.BytesIO()
    pixelated.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr.getvalue()

def generate_with_candle(prompt, height=512, width=512, sd_version=3):
    """Generate an image using Candle Stable Diffusion"""
    sd_version = int(sd_version)  # Ensure it's an integer
    if sd_version not in [1, 2, 3]:
        print(f"{Fore.YELLOW}⚠ Invalid SD version {sd_version}, defaulting to 3{Style.RESET_ALL}")
        sd_version = 3
        
    print(f"{Fore.BLUE}🦀 Generating image with Candle SD{sd_version} for prompt:{Style.RESET_ALL} '{prompt}'")
    
    # Create a temporary file to store the output image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    # Show a spinner during model generation
    spinner = spinner_context(f"Running Candle Stable Diffusion {sd_version}")
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
        if sd_version == 3:
            cmd.extend(["--sd-version", "3"])
            # SD3 can use SDXL weights path format
            cmd.extend(["--model-id", "stabilityai/stable-diffusion-3-medium-diffusers"])
        elif sd_version == 2:
            cmd.extend(["--sd-version", "2"])
            cmd.extend(["--model-id", "stabilityai/stable-diffusion-2-1-base"])
            
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

def generate_8bit_pfp(prompt, output_path="8bit_pfp.png", palette_size=64, dithering=True, sd_version=3):
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
            sd_image_bytes, pil_image = generate_with_candle(prompt, sd_version=sd_version)
    
    # Save the original image for comparison if requested
    base_filename, ext = os.path.splitext(output_path)
    original_path = f"{base_filename}_original{ext}"
    print(f"{Fore.BLUE}💾 Saving original image to:{Style.RESET_ALL} {original_path}")
    pil_image.save(original_path)
    
    # Step 2: Convert to 8-bit
    if use_rust:
        # Use the Rust implementation
        print(f"{Fore.BLUE}🦀 Converting to 8-bit style using Rust...{Style.RESET_ALL}")
        with tqdm(total=1, desc="Converting", unit="image") as progress_bar:
            eight_bit_bytes = rust_8bit.convert_to_8bit(sd_image_bytes, palette_size, dithering)
            progress_bar.update(1)
    else:
        # Use the Python fallback
        eight_bit_bytes = convert_to_8bit_python(pil_image, palette_size, dithering)
    
    # Step 3: Save the result
    print(f"{Fore.BLUE}💾 Saving 8-bit image to:{Style.RESET_ALL} {output_path}")
    with open(output_path, "wb") as f:
        f.write(eight_bit_bytes)
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an 8-bit style profile picture")
    parser.add_argument("prompt", help="Text description of the image you want to generate")
    parser.add_argument("--output", "-o", default="8bit_pfp.png", help="Output file path (default: 8bit_pfp.png)")
    parser.add_argument("--palette-size", "-p", type=int, default=64, help="Number of colors in the palette (8-256)")
    parser.add_argument("--dithering", "-d", action="store_true", help="Apply dithering for texture")
    parser.add_argument("--sd-version", type=int, default=3, choices=[1, 2, 3], 
                        help="Stable Diffusion version to use (1, 2, or 3, default: 3)")
    
    args = parser.parse_args()
    
    # Validate palette size
    if args.palette_size < 8 or args.palette_size > 256:
        parser.error("Palette size must be between 8 and 256")
    
    try:
        # Generate the image
        output_path = generate_8bit_pfp(
            args.prompt, 
            output_path=args.output,
            palette_size=args.palette_size,
            dithering=args.dithering,
            sd_version=args.sd_version
        )
        
        print(f"\n{Fore.GREEN}✨ Done! 8-bit PFP saved to:{Style.RESET_ALL} {output_path}")
        print(f"{Fore.YELLOW}👀 Original image saved to:{Style.RESET_ALL} {os.path.splitext(output_path)[0]}_original{os.path.splitext(output_path)[1]}")
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}⚠ Process interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {str(e)}{Style.RESET_ALL}")