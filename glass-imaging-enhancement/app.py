import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance
import cv2

def enhance_low_light_image(image):
    """
    Enhance low-light images using adaptive histogram equalization
    and brightness/contrast adjustments.
    
    This simulates the kind of computational photography Glass Imaging does
    with their neural processing.
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert RGB to LAB color space for better enhancement
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    # Merge channels back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    # Convert back to PIL for additional enhancements
    enhanced_pil = Image.fromarray(enhanced)
    
    # Boost brightness slightly
    enhancer = ImageEnhance.Brightness(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.2)
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.3)
    
    # Enhance color saturation
    enhancer = ImageEnhance.Color(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.15)
    
    # Sharpen the image
    enhancer = ImageEnhance.Sharpness(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.4)
    
    return enhanced_pil

def denoise_image(image):
    """
    Apply denoising to reduce noise common in low-light photos
    """
    img_array = np.array(image)
    
    # Apply Non-local Means Denoising
    denoised = cv2.fastNlMeansDenoisingColored(
        img_array, 
        None, 
        h=10,  # filter strength
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21
    )
    
    return Image.fromarray(denoised)

def full_enhancement_pipeline(image, denoise=True):
    """
    Full enhancement pipeline combining multiple techniques
    """
    if image is None:
        return None
    
    # Step 1: Denoise if requested
    if denoise:
        enhanced = denoise_image(image)
    else:
        enhanced = image
    
    # Step 2: Enhance brightness and contrast
    enhanced = enhance_low_light_image(enhanced)
    
    return enhanced

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔬 Glass Imaging - Low-Light Enhancement Demo
    
    ### AI-Powered Computational Photography
    
    This demo showcases advanced image enhancement techniques similar to Glass Imaging's 
    neural processing technology. Upload a low-light or dark image to see the enhancement.
    
    **Technologies Used:**
    - Adaptive Histogram Equalization (CLAHE)
    - LAB Color Space Processing
    - Non-local Means Denoising
    - Multi-stage Enhancement Pipeline
    
    ---
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil", 
                label="📷 Upload Low-Light Image",
                height=400
            )
            
            denoise_checkbox = gr.Checkbox(
                label="Apply Denoising (recommended for very noisy images)",
                value=True
            )
            
            enhance_btn = gr.Button(
                "✨ Enhance Image",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("""
            ### About Glass Imaging
            Glass Imaging replaces traditional camera glass lenses with deep neural networks,
            delivering DSLR-quality images from ultra-thin smartphone cameras.
            
            **Key Innovation:** Raw Neural Processing
            - Co-designed AI + Optics + Software
            - Validated by DXOMARK
            - Deployed on smartphones (Xiaomi, Motorola)
            """)
        
        with gr.Column():
            output_image = gr.Image(
                type="pil",
                label="✨ Enhanced Result",
                height=400
            )
            
            gr.Markdown("""
            ### Enhancement Pipeline
            
            1. **Denoising** - Reduces noise in dark areas
            2. **CLAHE** - Adaptive contrast enhancement
            3. **Brightness Boost** - Illuminates dark regions
            4. **Contrast Adjustment** - Improves dynamic range
            5. **Color Enhancement** - Restores natural colors
            6. **Sharpening** - Recovers fine details
            
            ---
            
            💡 **Tip:** Try uploading photos taken in:
            - Low light / nighttime
            - Indoor dim lighting
            - Backlit scenes
            - Underexposed photos
            """)
    
    # Wire up the enhancement
    enhance_btn.click(
        fn=full_enhancement_pipeline,
        inputs=[input_image, denoise_checkbox],
        outputs=output_image
    )
    
    gr.Markdown("""
    ---
    
    ### 🎯 Technical Details
    
    This demo uses classical computer vision techniques combined with adaptive algorithms
    to simulate Glass Imaging's approach to computational photography.
    
    **Real Glass Imaging Technology:**
    - Deep neural networks for lens correction
    - Raw sensor data processing
    - AI-driven optical aberration correction
    - Real-time processing on mobile devices
    
    **Built by:** Anju Vilashni Nandhakumar| **For:** Glass Imaging Application
    
    🔗 [Glass Imaging Website](https://glass-imaging.com) | 
    🔗 [GitHub Demo Source](https://github.com/Av1352/Job-search-demos/tree/main/glass-imaging-enhancement)
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()