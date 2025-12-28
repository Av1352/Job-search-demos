import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance
import cv2

def enhance_low_light_image(image):
    """
    Enhance low-light images using adaptive histogram equalization
    and brightness/contrast adjustments.
    """
    img_array = np.array(image)
    
    # Convert RGB to LAB color space
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    # Merge channels back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    # Convert to PIL for additional enhancements
    enhanced_pil = Image.fromarray(enhanced)
    
    # Boost brightness
    enhancer = ImageEnhance.Brightness(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.2)
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.3)
    
    # Enhance color saturation
    enhancer = ImageEnhance.Color(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.15)
    
    # Sharpen
    enhancer = ImageEnhance.Sharpness(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.4)
    
    return enhanced_pil

def denoise_image(image):
    """Apply denoising to reduce noise"""
    img_array = np.array(image)
    
    # Non-local Means Denoising
    denoised = cv2.fastNlMeansDenoisingColored(
        img_array, 
        None, 
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21
    )
    
    return Image.fromarray(denoised)

def full_enhancement_pipeline(image, denoise=True):
    """Full enhancement pipeline"""
    if image is None:
        return None, "<p style='color: #ef4444; font-size: 16px;'>❌ Please upload an image first</p>"
    
    # Apply enhancements
    if denoise:
        enhanced = denoise_image(image)
    else:
        enhanced = image
    
    enhanced = enhance_low_light_image(enhanced)
    
    # Calculate improvement metrics
    original_array = np.array(image)
    enhanced_array = np.array(enhanced)
    
    original_brightness = np.mean(original_array)
    enhanced_brightness = np.mean(enhanced_array)
    brightness_improvement = ((enhanced_brightness - original_brightness) / original_brightness) * 100
    
    # Generate results report
    report = f"""
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 14px; padding: 24px; box-shadow: 0 6px 12px rgba(16, 185, 129, 0.2);">
        <h3 style="color: #065f46; font-size: 24px; font-weight: 800; margin: 0 0 20px 0;">✨ Enhancement Complete</h3>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px;">
            <div style="background: white; padding: 16px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
                <p style="color: #6b7280; font-size: 12px; margin: 0;">Original Brightness</p>
                <p style="color: #1f2937; font-size: 28px; font-weight: 800; margin: 5px 0;">{original_brightness:.1f}</p>
            </div>
            <div style="background: white; padding: 16px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
                <p style="color: #6b7280; font-size: 12px; margin: 0;">Enhanced Brightness</p>
                <p style="color: #10b981; font-size: 28px; font-weight: 800; margin: 5px 0;">{enhanced_brightness:.1f}</p>
            </div>
        </div>
        
        <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
            <p style="color: #6b7280; font-size: 14px; margin: 0 0 8px 0;">Brightness Improvement</p>
            <p style="color: #10b981; font-size: 42px; font-weight: 900; margin: 0;">+{brightness_improvement:.1f}%</p>
        </div>
        
        <div style="background: rgba(16, 185, 129, 0.15); padding: 15px; border-radius: 8px; margin-top: 20px;">
            <h4 style="color: #065f46; font-weight: 700; margin: 0 0 10px 0; font-size: 15px;">🔬 Enhancement Pipeline Applied:</h4>
            <ul style="margin: 0; padding-left: 24px; color: #047857; font-size: 13px; line-height: 2;">
                <li><strong>CLAHE</strong> - Adaptive contrast enhancement</li>
                <li><strong>LAB Color Space</strong> - Perceptual color processing</li>
                <li><strong>Brightness Boost</strong> - 20% increase</li>
                <li><strong>Contrast Enhancement</strong> - 30% increase</li>
                <li><strong>Color Saturation</strong> - 15% boost</li>
                <li><strong>Sharpening</strong> - 40% detail recovery</li>
                {('<li><strong>Denoising</strong> - Non-local means filtering</li>' if denoise else '')}
            </ul>
        </div>
    </div>
    """
    
    return enhanced, report

# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="emerald")) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 6px 16px rgba(16, 185, 129, 0.15);">
        <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #10b981 0%, #14b8a6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4); margin: 0 auto 20px auto;">
            <span style="font-size: 44px;">🔬</span>
        </div>
        
        <h1 style="font-size: 52px; font-weight: 900; background: linear-gradient(135deg, #10b981 0%, #14b8a6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0 0 15px 0;">
            Glass Imaging
        </h1>
        
        <p style="font-size: 26px; color: #1f2937; font-weight: 700; margin: 12px 0;">AI-Powered Low-Light Enhancement</p>
        <p style="font-size: 16px; color: #6b7280; font-weight: 500; margin-bottom: 24px;">Computational Photography for Medical & Mobile Imaging</p>
        
        <div style="display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; max-width: 700px; margin: 0 auto;">
            <span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);">CLAHE</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(139, 92, 246, 0.3);">LAB Color Space</span>
            <span style="background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(249, 115, 22, 0.3);">Denoising</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(16, 185, 129, 0.3);">Real-time</span>
        </div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            gr.HTML("<h3 style='color: #10b981; font-size: 22px; font-weight: 700; margin-bottom: 15px;'>📷 Input Image</h3>")
            
            input_image = gr.Image(
                type="pil", 
                label="Upload Low-Light Image",
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
            
            gr.HTML("""
            <hr style="margin: 25px 0; border: 1px solid #e5e7eb;">
            <div style="background: #f0fdf4; border: 2px solid #10b981; border-radius: 10px; padding: 20px;">
                <h4 style="color: #065f46; margin: 0 0 12px 0; font-size: 16px; font-weight: 700;">About Glass Imaging</h4>
                <p style="color: #047857; font-size: 14px; line-height: 1.8; margin: 0;">
                    Glass Imaging replaces traditional camera lenses with deep neural networks, 
                    delivering DSLR-quality images from ultra-thin smartphone cameras.
                </p>
                <hr style="margin: 15px 0; border: 1px solid #d1fae5;">
                <h4 style="color: #065f46; margin: 12px 0 8px 0; font-size: 14px; font-weight: 700;">Key Innovation:</h4>
                <ul style="margin: 0; padding-left: 20px; color: #047857; font-size: 13px; line-height: 2;">
                    <li><strong>Raw Neural Processing</strong></li>
                    <li>Co-designed AI + Optics + Software</li>
                    <li>Validated by DXOMARK</li>
                    <li>Deployed on Xiaomi & Motorola</li>
                </ul>
            </div>
            """)
        
        with gr.Column():
            gr.HTML("<h3 style='color: #14b8a6; font-size: 22px; font-weight: 700; margin-bottom: 15px;'>✨ Enhanced Output</h3>")
            
            output_image = gr.Image(
                type="pil",
                label="Enhanced Result",
                height=400
            )
            
            results_output = gr.HTML(label="Enhancement Report")
    
    gr.HTML("<hr style='border: 2px solid #e5e7eb; margin: 30px 0;'>")
    
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 2px solid #3b82f6; border-radius: 14px; padding: 24px; margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 700; margin: 0 0 18px 0;">🎨 Enhancement Pipeline</h3>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #10b981;">
                <p style="font-weight: 700; color: #065f46; margin: 0 0 5px 0;">1. Denoising</p>
                <p style="font-size: 12px; color: #6b7280; margin: 0;">Reduces noise in dark areas</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                <p style="font-weight: 700; color: #1e40af; margin: 0 0 5px 0;">2. CLAHE</p>
                <p style="font-size: 12px; color: #6b7280; margin: 0;">Adaptive contrast enhancement</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b;">
                <p style="font-weight: 700; color: #92400e; margin: 0 0 5px 0;">3. Brightness</p>
                <p style="font-size: 12px; color: #6b7280; margin: 0;">Illuminates dark regions</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #8b5cf6;">
                <p style="font-weight: 700; color: #6b21a8; margin: 0 0 5px 0;">4. Contrast</p>
                <p style="font-size: 12px; color: #6b7280; margin: 0;">Improves dynamic range</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #ec4899;">
                <p style="font-weight: 700; color: #9f1239; margin: 0 0 5px 0;">5. Color</p>
                <p style="font-size: 12px; color: #6b7280; margin: 0;">Restores natural colors</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #14b8a6;">
                <p style="font-weight: 700; color: #115e59; margin: 0 0 5px 0;">6. Sharpening</p>
                <p style="font-size: 12px; color: #6b7280; margin: 0;">Recovers fine details</p>
            </div>
        </div>
        
        <div style="background: rgba(59, 130, 246, 0.1); padding: 15px; border-radius: 8px; margin-top: 20px;">
            <p style="color: #1e40af; font-size: 13px; font-weight: 600; margin: 0;">
                💡 <strong>Tip:</strong> Best results with nighttime photos, indoor dim lighting, backlit scenes, or underexposed images
            </p>
        </div>
    </div>
    """)
    
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 2px solid #f59e0b; border-radius: 14px; padding: 24px;">
        <h3 style="color: #92400e; font-size: 22px; font-weight: 700; margin: 0 0 18px 0;">🎯 Technical Details</h3>
        
        <div style="background: white; padding: 18px; border-radius: 10px; margin-bottom: 15px;">
            <h4 style="color: #1f2937; font-weight: 700; margin: 0 0 10px 0; font-size: 15px;">This Demo Uses:</h4>
            <ul style="margin: 0; padding-left: 24px; color: #374151; font-size: 14px; line-height: 2;">
                <li><strong>CLAHE</strong> - Contrast Limited Adaptive Histogram Equalization</li>
                <li><strong>LAB Color Space</strong> - Perceptual color processing</li>
                <li><strong>Non-local Means</strong> - Advanced denoising algorithm</li>
                <li><strong>Multi-stage Pipeline</strong> - Brightness, contrast, color, sharpness</li>
            </ul>
        </div>
        
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <h4 style="color: #1f2937; font-weight: 700; margin: 0 0 10px 0; font-size: 15px;">Real Glass Imaging Technology:</h4>
            <ul style="margin: 0; padding-left: 24px; color: #374151; font-size: 14px; line-height: 2;">
                <li>Deep neural networks for lens correction</li>
                <li>Raw sensor data processing</li>
                <li>AI-driven optical aberration correction</li>
                <li>Real-time processing on mobile devices</li>
            </ul>
        </div>
    </div>
    """)
    
    gr.HTML("""
    <hr style="border: 2px solid #e5e7eb; margin: 40px 0;">
    
    <div style="text-align: center; padding: 28px; background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); border-radius: 16px; box-shadow: 0 4px 8px rgba(0,0,0,0.08);">
        <h3 style="color: #10b981; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">👨‍💻 About This Demo</h3>
        <p style="color: #1f2937; margin: 10px 0; font-size: 16px; line-height: 1.6;">
            Built for <strong style="color: #10b981;">Glass Imaging</strong> by 
            <strong style="color: #3b82f6;">Anju Vilashni Nandhakumar</strong>
        </p>
        <div style="margin: 20px 0; padding: 18px; background: white; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <p style="margin: 6px 0; font-size: 14px;">
                📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: #3b82f6; font-weight: 600;">nandhakumar.anju@gmail.com</a>
            </p>
            <p style="margin: 6px 0; font-size: 14px;">
                💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: #3b82f6; font-weight: 600;">LinkedIn</a> | 
                💻 <a href="https://github.com/Av1352" target="_blank" style="color: #3b82f6; font-weight: 600;">GitHub</a> | 
                🌐 <a href="https://vxanju.com" target="_blank" style="color: #3b82f6; font-weight: 600;">Portfolio</a>
            </p>
        </div>
        <p style="color: #6b7280; font-size: 14px; margin: 12px 0; font-weight: 600;">
            <strong style="color: #10b981;">Tech Stack:</strong> OpenCV, CLAHE, LAB Color Space, PIL Enhancement, Gradio
        </p>
        <hr style="border: 1px solid #e5e7eb; margin: 20px 0;">
        <p style="color: #9ca3af; font-size: 13px; font-style: italic; line-height: 1.6;">
            Demonstration of computational photography techniques for low-light enhancement.<br>
            Production Glass Imaging systems use deep neural networks for real-time processing.
        </p>
    </div>
    """)
    
    # Wire up the enhancement
    enhance_btn.click(
        fn=full_enhancement_pipeline,
        inputs=[input_image, denoise_checkbox],
        outputs=[output_image, results_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()