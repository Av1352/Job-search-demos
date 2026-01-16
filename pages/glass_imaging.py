"""
Glass Imaging - AI-Powered Low-Light Enhancement
Computational Photography for Medical & Mobile Imaging
Built for Glass Imaging by Anju Nandhakumar
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import cv2

# Page config
st.set_page_config(
    page_title="Glass Imaging Demo - Anju Vilashni",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main { background: white; }
.stButton button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px 32px;
    font-size: 16px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

def enhance_low_light_image(image):
    """Enhance low-light images using CLAHE and adjustments"""
    img_array = np.array(image)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    enhanced_pil = Image.fromarray(enhanced)
    enhancer = ImageEnhance.Brightness(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.3)
    enhancer = ImageEnhance.Color(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.15)
    enhancer = ImageEnhance.Sharpness(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.4)
    return enhanced_pil

def denoise_image(image):
    """Apply denoising"""
    img_array = np.array(image)
    denoised = cv2.fastNlMeansDenoisingColored(img_array, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    return Image.fromarray(denoised)

# Header - NO empty lines!
st.markdown("""
<div style="text-align: center; padding: 20px 30px 70px 20px; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 25px; box-shadow: 0 12px 28px rgba(16, 185, 129, 0.35);">
    <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #14b8a6 100%); border-radius: 50%; margin: 0 auto 25px auto; border: 5px solid white; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5);">
        <span style="font-size: 56px;">🔬</span>
    </div>
    <h1 style="font-size: 58px; font-weight: 900; color: #065f46; margin: 0 0 18px 0;">
        Glass Imaging
    </h1>
    <p style="font-size: 28px; color: #1f2937; font-weight: 700; margin: 15px 0;">AI-Powered Low-Light Enhancement</p>
    <p style="font-size: 18px; color: #6b7280; font-weight: 500; margin-bottom: 25px;">Computational Photography for Medical & Mobile Imaging</p>
    <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; max-width: 700px; margin: 28px auto 0 auto;">
        <span style="background:#3b82f6;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">CLAHE</span>
        <span style="background:#8b5cf6;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">LAB Color Space</span>
        <span style="background:#f97316;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Denoising</span>
        <span style="background:#10b981;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Real-time</span>
    </div>
    <p style="font-size: 16px; color: #374151; margin-top: 28px; font-weight: 600;">
        Built for <strong style="color:#065f46;">Glass Imaging</strong> by <strong style="color:#065f46;">Anju Nandhakumar</strong>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='color: #10b981; font-size: 22px; font-weight: 700; margin-bottom: 15px;'>📷 Input Image</h3>", unsafe_allow_html=True)
    
    input_image = st.file_uploader("Upload Low-Light Image", type=['png', 'jpg', 'jpeg'])
    
    denoise = st.checkbox("Apply Denoising (recommended for very noisy images)", value=True)
    
    enhance_btn = st.button("✨ Enhance Image", use_container_width=True, type="primary")
    
    st.markdown("""
<div style="background: #f0fdf4; border: 2px solid #10b981; border-radius: 10px; padding: 20px; margin-top: 25px;">
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
""", unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='color: #14b8a6; font-size: 22px; font-weight: 700; margin-bottom: 15px;'>✨ Enhanced Output</h3>", unsafe_allow_html=True)
    
    output_placeholder = st.empty()
    results_placeholder = st.empty()
    
    if enhance_btn and input_image is not None:
        image = Image.open(input_image)
        
        # Apply enhancements
        if denoise:
            enhanced = denoise_image(image)
        else:
            enhanced = image
        
        enhanced = enhance_low_light_image(enhanced)
        
        # Calculate metrics
        original_array = np.array(image)
        enhanced_array = np.array(enhanced)
        original_brightness = np.mean(original_array)
        enhanced_brightness = np.mean(enhanced_array)
        brightness_improvement = ((enhanced_brightness - original_brightness) / original_brightness) * 100
        
        # Display enhanced image
        output_placeholder.image(enhanced, caption="Enhanced Result", use_container_width=True)
        
        # Results report
        results_placeholder.markdown(f"""
<div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 14px; padding: 24px;">
    <h3 style="color: #065f46; font-size: 24px; font-weight: 800; margin: 0 0 20px 0;">✨ Enhancement Complete</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px;">
        <div style="background: white; padding: 16px; border-radius: 10px;">
            <p style="color: #6b7280; font-size: 12px; margin: 0;">Original Brightness</p>
            <p style="color: #1f2937; font-size: 28px; font-weight: 800; margin: 5px 0;">{original_brightness:.1f}</p>
        </div>
        <div style="background: white; padding: 16px; border-radius: 10px;">
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
""", unsafe_allow_html=True)
    elif enhance_btn:
        st.error("❌ Please upload an image first!")

# Pipeline explanation
st.markdown("""
<div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 2px solid #3b82f6; border-radius: 14px; padding: 24px; margin: 25px 0;">
    <h3 style="color: #1e40af; font-size: 22px; font-weight: 700; margin: 0 0 18px 0;">🎨 Enhancement Pipeline</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
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
""", unsafe_allow_html=True)

# Technical details
st.markdown("""
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
""", unsafe_allow_html=True)

# Footer
st.markdown("<hr style='border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 20px; color: #065f46;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: #10b981;">Glass Imaging</strong> by <strong style="color: #3b82f6;">Anju Vilashni Nandhakumar</strong>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: rgba(16, 185, 129, 0.1); border-radius: 16px; padding: 24px; margin-top: 20px; text-align: center;">
    <p style="margin: 8px 0; font-size: 16px;">
        📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: #10b981; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a>
    </p>
    <p style="margin: 8px 0; font-size: 16px;">
        💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: #10b981; font-weight: 700; text-decoration: none;">LinkedIn</a> | 
        💻 <a href="https://github.com/Av1352" target="_blank" style="color: #10b981; font-weight: 700; text-decoration: none;">GitHub</a> | 
        🌐 <a href="https://vxanju.com" target="_blank" style="color: #10b981; font-weight: 700; text-decoration: none;">Portfolio</a>
    </p>
    <p style="font-size: 15px; margin: 18px 0 0 0; font-weight: 700; color: #1f2937;">
        <strong>Tech Stack:</strong> OpenCV • CLAHE • LAB Color Space • PIL Enhancement • Streamlit
    </p>
</div>
""", unsafe_allow_html=True)