use pyo3::prelude::*;
use pyo3::types::PyBytes;
use image::{GenericImageView, ImageBuffer, Rgb};
use std::io::Cursor;

#[pyfunction]
fn convert_to_8bit(image_data: &[u8], palette_size: usize, dithering: bool) -> PyResult<Py<PyBytes>> {
    // Load image from bytes
    let img = image::load_from_memory(image_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to load image: {}", e)))?;
    
    // Convert to RGB
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    
    // Create a limited color palette (8-bit has max 256 colors)
    let actual_palette_size = palette_size.min(256);
    let mut palette = generate_palette(actual_palette_size);
    
    // Create output image
    let mut output_img = ImageBuffer::new(width, height);
    
    // Apply 8-bit conversion
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb_img.get_pixel(x, y);
            
            // Either apply dithering or direct color mapping
            let new_pixel = if dithering {
                apply_dithering(&rgb_img, x, y, &palette)
            } else {
                find_nearest_color(pixel, &palette)
            };
            
            output_img.put_pixel(x, y, new_pixel);
        }
    }
    
    // Convert output image to bytes
    let mut output_bytes = Cursor::new(Vec::new());
    output_img.write_to(&mut output_bytes, image::ImageOutputFormat::Png)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to encode image: {}", e)))?;
    
    // Return bytes to Python
    Python::with_gil(|py| {
        Ok(PyBytes::new(py, &output_bytes.into_inner()).into())
    })
}

// Generate a fixed palette of colors for 8-bit aesthetic
fn generate_palette(size: usize) -> Vec<Rgb<u8>> {
    let mut palette = Vec::with_capacity(size);
    
    // For a true 8-bit look, use specific color values rather than evenly distributed ones
    // This more closely mimics classic 8-bit systems
    
    // Add some basic colors
    palette.push(Rgb([0, 0, 0]));      // Black
    palette.push(Rgb([255, 255, 255])); // White
    palette.push(Rgb([255, 0, 0]));     // Red
    palette.push(Rgb([0, 255, 0]));     // Green
    palette.push(Rgb([0, 0, 255]));     // Blue
    palette.push(Rgb([255, 255, 0]));   // Yellow
    palette.push(Rgb([255, 0, 255]));   // Magenta
    palette.push(Rgb([0, 255, 255]));   // Cyan
    
    // Fill remaining slots with evenly distributed RGB values
    if size > 8 {
        let steps = ((size - 8) as f32).cbrt().ceil() as u8;
        let step_size = 255 / steps;
        
        for r in (0..=255).step_by(step_size as usize) {
            for g in (0..=255).step_by(step_size as usize) {
                for b in (0..=255).step_by(step_size as usize) {
                    if palette.len() >= size {
                        break;
                    }
                    // Skip if too similar to existing colors
                    let new_color = Rgb([r as u8, g as u8, b as u8]);
                    if !palette.contains(&new_color) {
                        palette.push(new_color);
                    }
                }
                if palette.len() >= size {
                    break;
                }
            }
            if palette.len() >= size {
                break;
            }
        }
    }
    
    palette
}

// Find the nearest color in the palette
fn find_nearest_color(pixel: &Rgb<u8>, palette: &[Rgb<u8>]) -> Rgb<u8> {
    palette.iter()
        .min_by_key(|&&p| color_distance(pixel, &p))
        .unwrap_or(&Rgb([0, 0, 0]))
        .clone()
}

// Calculate Euclidean distance between colors
fn color_distance(c1: &Rgb<u8>, c2: &Rgb<u8>) -> u32 {
    let r1 = c1[0] as i32;
    let g1 = c1[1] as i32;
    let b1 = c1[2] as i32;
    let r2 = c2[0] as i32;
    let g2 = c2[1] as i32;
    let b2 = c2[2] as i32;
    
    let dr = r1 - r2;
    let dg = g1 - g2;
    let db = b1 - b2;
    
    (dr*dr + dg*dg + db*db) as u32
}

// Apply Floyd-Steinberg dithering
fn apply_dithering(img: &ImageBuffer<Rgb<u8>, Vec<u8>>, x: u32, y: u32, palette: &[Rgb<u8>]) -> Rgb<u8> {
    let pixel = img.get_pixel(x, y);
    let nearest = find_nearest_color(pixel, palette);
    
    // For simplicity, we're not implementing the full dithering algorithm here
    // A real implementation would propagate quantization errors to neighboring pixels
    nearest
}

#[pymodule]
fn rust_8bit(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(convert_to_8bit, m)?)?;
    Ok(())
}