"""
Deep Lake Dataset Version Control Demo
Multi-modal AI dataset versioning and management
Built for Activeloop by Anju Nandhakumar
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont

# Page config
st.set_page_config(
    page_title="Activeloop Demo - Anju Vilashni",
    page_icon="🗂️",
    layout="wide"
)

# Custom CSS with your vibrant gradient
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px 24px;
}
.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px 32px;
    font-size: 16px;
    border: none;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}
</style>
""", unsafe_allow_html=True)

# Simulated dataset versions (same data as Gradio)
DATASET_VERSIONS = {
    "v1.0.0": {
        "created": "2024-12-15",
        "author": "data-team",
        "samples": 10000,
        "accuracy": 0.87,
        "description": "Initial release with base medical imaging dataset",
        "modalities": ["images", "labels"],
        "size_mb": 2400,
        "augmentations": ["rotation", "flip"],
        "splits": {"train": 7000, "val": 2000, "test": 1000}
    },
    "v1.1.0": {
        "created": "2024-12-20",
        "author": "ml-team",
        "samples": 15000,
        "accuracy": 0.91,
        "description": "Added 5K samples with enhanced augmentations",
        "modalities": ["images", "labels", "metadata"],
        "size_mb": 3600,
        "augmentations": ["rotation", "flip", "brightness", "contrast"],
        "splits": {"train": 10500, "val": 3000, "test": 1500}
    },
    "v2.0.0": {
        "created": "2024-12-25",
        "author": "anju-vilashni",
        "samples": 20000,
        "accuracy": 0.94,
        "description": "Major update: embeddings, multi-modal fusion, improved labels",
        "modalities": ["images", "labels", "embeddings", "metadata", "text"],
        "size_mb": 4800,
        "augmentations": ["rotation", "flip", "brightness", "contrast", "cutout", "mixup"],
        "splits": {"train": 14000, "val": 4000, "test": 2000}
    }
}

def generate_sample_image(version, index):
    """Generate a sample medical image for visualization"""
    img = Image.new('RGB', (200, 200), color=(240, 240, 250))
    draw = ImageDraw.Draw(img)
    
    colors = {
        "v1.0.0": [(100, 100, 200), (150, 150, 220)],
        "v1.1.0": [(120, 120, 220), (170, 170, 240)],
        "v2.0.0": [(140, 140, 240), (190, 190, 255)]
    }
    
    color1, color2 = colors.get(version, [(100, 100, 200), (150, 150, 220)])
    
    for i in range(5):
        radius = 80 - (i * 15)
        color = tuple(int(c1 + (c2 - c1) * i / 5) for c1, c2 in zip(color1, color2))
        draw.ellipse([100-radius, 100-radius, 100+radius, 100+radius], fill=color)
    
    draw.text((10, 10), f"{version} - Sample {index}", fill=(255, 255, 255))
    
    return img

# Header
st.markdown("""
<div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
    <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
        <span style="font-size: 56px;">🗂️</span>
    </div>
    
    <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        Deep Lake Dataset Versioning
    </h1>
    
    <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Multi-Modal AI Dataset Management</p>
    <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Version control for ML datasets • Track performance • Compare versions</p>
    
    <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 800px; margin: 28px auto 0 auto;">
        <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Multi-Modal</span>
        <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Version Control</span>
        <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Deep Lake</span>
        <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Activeloop</span>
    </div>
    
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
        Built for <strong style="color: white;">Activeloop</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Compare Versions", "📦 Version Details", "📅 Timeline"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Compare Dataset Versions</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Select two versions to compare performance, size, and modalities</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        version1 = st.selectbox("Version 1 (Older)", list(DATASET_VERSIONS.keys()), index=0, key="v1")
    with col2:
        version2 = st.selectbox("Version 2 (Newer)", list(DATASET_VERSIONS.keys()), index=2, key="v2")
    
    if st.button("🔄 Compare Versions", key="compare"):
        v1_data = DATASET_VERSIONS[version1]
        v2_data = DATASET_VERSIONS[version2]
        
        # Comparison HTML
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
            <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Version Comparison</h2>
            <div style="display: grid; grid-template-columns: 1fr auto 1fr; gap: 20px; align-items: center;">
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 24px; border: 2px solid rgba(255,255,255,0.3);">
                    <h3 style="color: white; font-size: 24px; font-weight: 800; margin: 0 0 15px 0;">{version1}</h3>
                    <p style="color: rgba(255,255,255,0.9); font-size: 14px; margin: 8px 0;"><strong>Date:</strong> {v1_data['created']}</p>
                    <p style="color: rgba(255,255,255,0.9); font-size: 14px; margin: 8px 0;"><strong>Samples:</strong> {v1_data['samples']:,}</p>
                    <p style="color: rgba(255,255,255,0.9); font-size: 14px; margin: 8px 0;"><strong>Accuracy:</strong> {v1_data['accuracy']:.1%}</p>
                    <p style="color: rgba(255,255,255,0.9); font-size: 14px; margin: 8px 0;"><strong>Size:</strong> {v1_data['size_mb']} MB</p>
                </div>
                
                <div style="color: white; font-size: 36px; font-weight: 900;">→</div>
                
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 24px; border: 2px solid rgba(255,255,255,0.3);">
                    <h3 style="color: white; font-size: 24px; font-weight: 800; margin: 0 0 15px 0;">{version2}</h3>
                    <p style="color: rgba(255,255,255,0.9); font-size: 14px; margin: 8px 0;"><strong>Date:</strong> {v2_data['created']}</p>
                    <p style="color: rgba(255,255,255,0.9); font-size: 14px; margin: 8px 0;"><strong>Samples:</strong> {v2_data['samples']:,}</p>
                    <p style="color: rgba(255,255,255,0.9); font-size: 14px; margin: 8px 0;"><strong>Accuracy:</strong> {v2_data['accuracy']:.1%}</p>
                    <p style="color: rgba(255,255,255,0.9); font-size: 14px; margin: 8px 0;"><strong>Size:</strong> {v2_data['size_mb']} MB</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Improvements section
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;">
            <h3 style="color: #065f46; font-size: 24px; font-weight: 800; margin: 0 0 20px 0;">📈 Improvements</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                    <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Sample Count</p>
                    <p style="font-size: 28px; color: #10b981; font-weight: 900; margin: 0;">+{v2_data['samples'] - v1_data['samples']:,}</p>
                    <p style="font-size: 12px; color: #9ca3af; margin: 6px 0 0 0;">{((v2_data['samples'] - v1_data['samples']) / v1_data['samples'] * 100):.1f}% increase</p>
                </div>
                
                <div style="background: white; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                    <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Accuracy</p>
                    <p style="font-size: 28px; color: #10b981; font-weight: 900; margin: 0;">+{(v2_data['accuracy'] - v1_data['accuracy']) * 100:.1f}%</p>
                    <p style="font-size: 12px; color: #9ca3af; margin: 6px 0 0 0;">{((v2_data['accuracy'] - v1_data['accuracy']) / v1_data['accuracy'] * 100):.1f}% improvement</p>
                </div>
                
                <div style="background: white; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                    <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">New Modalities</p>
                    <p style="font-size: 28px; color: #10b981; font-weight: 900; margin: 0;">+{len(v2_data['modalities']) - len(v1_data['modalities'])}</p>
                    <p style="font-size: 12px; color: #9ca3af; margin: 6px 0 0 0;">{', '.join(set(v2_data['modalities']) - set(v1_data['modalities']))}</p>
                </div>
                
                <div style="background: white; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                    <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">New Augmentations</p>
                    <p style="font-size: 28px; color: #10b981; font-weight: 900; margin: 0;">+{len(v2_data['augmentations']) - len(v1_data['augmentations'])}</p>
                    <p style="font-size: 12px; color: #9ca3af; margin: 6px 0 0 0;">{', '.join(set(v2_data['augmentations']) - set(v1_data['augmentations']))}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance chart
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Model Accuracy', 'Dataset Size'))
        
        fig.add_trace(
            go.Bar(x=[version1, version2], y=[v1_data['accuracy'], v2_data['accuracy']],
                   marker_color=['#3b82f6', '#10b981'],
                   text=[f'{v1_data["accuracy"]:.1%}', f'{v2_data["accuracy"]:.1%}'],
                   textposition='outside'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=[version1, version2], y=[v1_data['samples'], v2_data['samples']],
                   marker_color=['#8b5cf6', '#ec4899'],
                   text=[f'{v1_data["samples"]:,}', f'{v2_data["samples"]:,}'],
                   textposition='outside'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, title_text="Version Comparison Metrics")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample images
        col1, col2 = st.columns(2)
        with col1:
            st.image(generate_sample_image(version1, 1), caption=f"{version1} Sample", use_container_width=True)
        with col2:
            st.image(generate_sample_image(version2, 1), caption=f"{version2} Sample", use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">View Version Details</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Explore comprehensive information about a specific dataset version</p>
    </div>
    """, unsafe_allow_html=True)
    
    version = st.selectbox("Select Version", list(DATASET_VERSIONS.keys()), index=2, key="details_version")
    
    if st.button("👁️ View Details", key="view"):
        data = DATASET_VERSIONS[version]
        
        # Version details HTML
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
            <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 25px;">
                <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);">
                    <span style="font-size: 40px;">📦</span>
                </div>
                <div>
                    <h2 style="color: #1e40af; font-size: 32px; font-weight: 900; margin: 0;">Dataset Version {version}</h2>
                    <p style="color: #3b82f6; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">{data['description']}</p>
                </div>
            </div>
            
            <div style="background: white; border-radius: 16px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <h3 style="color: #1f2937; font-size: 20px; font-weight: 800; margin: 0 0 18px 0;">📋 Version Information</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                    <div>
                        <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Created</p>
                        <p style="font-size: 18px; color: #1f2937; font-weight: 700; margin: 0;">{data['created']}</p>
                    </div>
                    <div>
                        <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Author</p>
                        <p style="font-size: 18px; color: #1f2937; font-weight: 700; margin: 0;">{data['author']}</p>
                    </div>
                    <div>
                        <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Total Samples</p>
                        <p style="font-size: 18px; color: #1f2937; font-weight: 700; margin: 0;">{data['samples']:,}</p>
                    </div>
                </div>
            </div>
            
            <div style="background: white; border-radius: 16px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 15px 0;">🔬 Modalities</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                    {''.join([f'<span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 8px 18px; border-radius: 20px; font-size: 13px; font-weight: 700; box-shadow: 0 2px 6px rgba(16, 185, 129, 0.3);">{mod}</span>' for mod in data['modalities']])}
                </div>
            </div>
            
            <div style="background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 15px 0;">🎨 Augmentations</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                    {''.join([f'<span style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 8px 18px; border-radius: 20px; font-size: 13px; font-weight: 700; box-shadow: 0 2px 6px rgba(245, 158, 11, 0.3);">{aug}</span>' for aug in data['augmentations']])}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Training', 'Validation', 'Test'],
            values=[data['splits']['train'], data['splits']['val'], data['splits']['test']],
            marker=dict(colors=['#3b82f6', '#8b5cf6', '#ec4899']),
            hole=0.4,
            textinfo='label+percent'
        )])
        fig.update_layout(title=f"Dataset Split Distribution - {version}", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample images
        st.markdown("<h3 style='color: #667eea; font-size: 20px; font-weight: 700; margin: 20px 0 15px 0;'>📸 Sample Images from Dataset</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(generate_sample_image(version, 1), caption="Sample 1", use_container_width=True)
        with col2:
            st.image(generate_sample_image(version, 2), caption="Sample 2", use_container_width=True)
        with col3:
            st.image(generate_sample_image(version, 3), caption="Sample 3", use_container_width=True)

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #92400e; font-size: 22px; font-weight: 800; margin: 0;">Dataset Evolution Timeline</h3>
        <p style="color: #d97706; font-size: 14px; margin: 8px 0 0 0;">Track how your dataset has grown and improved over time</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Generate Timeline", key="timeline"):
        # Timeline HTML
        timeline_html = """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);">
            <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 25px 0;">📅 Version Timeline</h2>
        """
        
        for version, data in DATASET_VERSIONS.items():
            timeline_html += f"""
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; margin-bottom: 18px; border-left: 5px solid white;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="color: white; font-size: 22px; font-weight: 800; margin: 0 0 8px 0;">{version}</h3>
                        <p style="color: rgba(255,255,255,0.9); font-size: 14px; margin: 0;">{data['description']}</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="color: rgba(255,255,255,0.8); font-size: 13px; margin: 0 0 4px 0;">{data['created']}</p>
                        <p style="color: white; font-size: 16px; font-weight: 700; margin: 0;">{data['samples']:,} samples</p>
                    </div>
                </div>
            </div>
            """
        
        timeline_html += "</div>"
        st.markdown(timeline_html, unsafe_allow_html=True)
        
        # Evolution charts
        versions_list = list(DATASET_VERSIONS.keys())
        accuracies = [DATASET_VERSIONS[v]['accuracy'] for v in versions_list]
        samples_list = [DATASET_VERSIONS[v]['samples'] for v in versions_list]
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Model Accuracy Progression', 'Dataset Size Growth'), vertical_spacing=0.12)
        
        fig.add_trace(
            go.Scatter(x=versions_list, y=accuracies, mode='lines+markers',
                      line=dict(color='#10b981', width=4), marker=dict(size=12, color='#059669')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=versions_list, y=samples_list, marker_color=['#3b82f6', '#8b5cf6', '#ec4899']),
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Samples", row=2, col=1)
        fig.update_layout(height=600, showlegend=False, title_text="Dataset Evolution Over Time")
        
        st.plotly_chart(fig, use_container_width=True)

# Footer (same as Gradio)
st.markdown("""
<hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">

<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Activeloop</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
    </p>
    <div style="margin: 25px 0; padding: 22px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.2);">
        <p style="margin: 8px 0; font-size: 16px;">
            📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: white; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a>
        </p>
        <p style="margin: 8px 0; font-size: 16px;">
            💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">LinkedIn</a> | 
            💻 <a href="https://github.com/Av1352" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">GitHub</a> | 
            🌐 <a href="https://vxanju.com" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">Portfolio</a>
        </p>
    </div>
    <p style="font-size: 15px; margin: 18px 0; font-weight: 700;">
        <strong style="color: white;">Tech Stack:</strong> Python • Streamlit • Plotly • PIL • Deep Lake Concepts
    </p>
</div>
""", unsafe_allow_html=True)