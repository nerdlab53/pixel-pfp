import os
import io
import base64
from flask import Flask, request, render_template, send_file
from PIL import Image
from generate import generate_image
from rust_8bit import convert_to_8bit

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None
    
    if request.method == 'POST':
        prompt = request.form.get('prompt', 'pixel character')
        palette_size = int(request.form.get('palette_size', 64))
        dithering = request.form.get('dithering') == 'on'
        
        # Step 1: Generate image with Stable Diffusion
        sd_image_bytes = generate_image(prompt)
        
        # Step 2: Convert to 8-bit using our Rust algorithm
        eight_bit_bytes = convert_to_8bit(sd_image_bytes, palette_size, dithering)
        
        # Convert to base64 for display
        img_data = base64.b64encode(eight_bit_bytes).decode('utf-8')
    
    return render_template('index.html', img_data=img_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)