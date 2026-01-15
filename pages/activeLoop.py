"""
Deep Lake Dataset Version Control Demo
Multi-modal AI dataset versioning and management
Built for Activeloop by Anju Nandhakumar
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw

# Page config
st.set_page_config(
    page_title="Activeloop Demo - Anju Vilashni",
    page_icon="🗂️",
    layout="wide"
)

# Simpler CSS
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
}
.gradient-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 50px 30px;
    border-radius: 25px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
}
.badge {
    background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
    color: white;
    padding: 8px 20px;
    border-radius: 20px;
    display: inline-block;
    margin: 5px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# Dataset versions
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
    """Generate a sample medical image"""
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
<div class="gradient-header">
    <h1 style="font-size: 48px; margin: 0;">🗂️ Deep Lake Dataset Versioning</h1>
    <p style="font-size: 22px; margin: 15px 0;">Multi-Modal AI Dataset Management</p>
    <p style="font-size: 16px; opacity: 0.9;">Version control for ML datasets • Track performance • Compare versions</p>
</div>
""", unsafe_allow_html=True)

# Badges
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <span class="badge">Multi-Modal</span>
    <span class="badge" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">Version Control</span>
    <span class="badge" style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%);">Deep Lake</span>
    <span class="badge" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">Activeloop</span>
</div>
""", unsafe_allow_html=True)

st.markdown("**Built for Activeloop by Anju Nandhakumar**")
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Compare Versions", "📦 Version Details", "📅 Timeline"])

with tab1:
    st.subheader("Compare Dataset Versions")
    st.write("Select two versions to compare performance, size, and modalities")
    
    col1, col2 = st.columns(2)
    with col1:
        version1 = st.selectbox("Version 1 (Older)", list(DATASET_VERSIONS.keys()), index=0, key="v1")
    with col2:
        version2 = st.selectbox("Version 2 (Newer)", list(DATASET_VERSIONS.keys()), index=2, key="v2")
    
    if st.button("🔄 Compare Versions", key="compare", type="primary"):
        v1_data = DATASET_VERSIONS[version1]
        v2_data = DATASET_VERSIONS[version2]
        
        # Use Streamlit columns instead of complex HTML
        st.markdown("### 📊 Version Comparison")
        
        col1, col2, col3 = st.columns([1, 0.2, 1])
        
        with col1:
            st.info(f"**{version1}**")
            st.write(f"**Date:** {v1_data['created']}")
            st.write(f"**Samples:** {v1_data['samples']:,}")
            st.write(f"**Accuracy:** {v1_data['accuracy']:.1%}")
            st.write(f"**Size:** {v1_data['size_mb']} MB")
        
        with col2:
            st.markdown("<h1 style='text-align: center;'>→</h1>", unsafe_allow_html=True)
        
        with col3:
            st.success(f"**{version2}**")
            st.write(f"**Date:** {v2_data['created']}")
            st.write(f"**Samples:** {v2_data['samples']:,}")
            st.write(f"**Accuracy:** {v2_data['accuracy']:.1%}")
            st.write(f"**Size:** {v2_data['size_mb']} MB")
        
        st.markdown("---")
        
        # Improvements using metrics
        st.subheader("📈 Improvements")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Sample Count", 
                f"+{v2_data['samples'] - v1_data['samples']:,}",
                f"{((v2_data['samples'] - v1_data['samples']) / v1_data['samples'] * 100):.1f}%"
            )
        
        with col2:
            st.metric(
                "Accuracy",
                f"+{(v2_data['accuracy'] - v1_data['accuracy']) * 100:.1f}%",
                f"{((v2_data['accuracy'] - v1_data['accuracy']) / v1_data['accuracy'] * 100):.1f}%"
            )
        
        with col3:
            st.metric(
                "New Modalities",
                f"+{len(v2_data['modalities']) - len(v1_data['modalities'])}",
                ', '.join(set(v2_data['modalities']) - set(v1_data['modalities']))
            )
        
        with col4:
            st.metric(
                "New Augmentations",
                f"+{len(v2_data['augmentations']) - len(v1_data['augmentations'])}",
                ', '.join(set(v2_data['augmentations']) - set(v1_data['augmentations']))
            )
        
        # Charts
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
    st.subheader("View Version Details")
    st.write("Explore comprehensive information about a specific dataset version")
    
    version = st.selectbox("Select Version", list(DATASET_VERSIONS.keys()), index=2, key="details_version")
    
    if st.button("👁️ View Details", key="view", type="primary"):
        data = DATASET_VERSIONS[version]
        
        st.markdown(f"## 📦 Dataset Version {version}")
        st.write(f"*{data['description']}*")
        
        # Version info using columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Created", data['created'])
        with col2:
            st.metric("Author", data['author'])
        with col3:
            st.metric("Total Samples", f"{data['samples']:,}")
        
        st.markdown("---")
        
        # Modalities
        st.subheader("🔬 Modalities")
        st.write(" • ".join(data['modalities']))
        
        # Augmentations
        st.subheader("🎨 Augmentations")
        st.write(" • ".join(data['augmentations']))
        
        st.markdown("---")
        
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
        st.subheader("📸 Sample Images from Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(generate_sample_image(version, 1), caption="Sample 1", use_container_width=True)
        with col2:
            st.image(generate_sample_image(version, 2), caption="Sample 2", use_container_width=True)
        with col3:
            st.image(generate_sample_image(version, 3), caption="Sample 3", use_container_width=True)

with tab3:
    st.subheader("Dataset Evolution Timeline")
    st.write("Track how your dataset has grown and improved over time")
    
    if st.button("📊 Generate Timeline", key="timeline", type="primary"):
        
        # Timeline using expanders
        st.markdown("### 📅 Version Timeline")
        
        for ver, data in DATASET_VERSIONS.items():
            with st.expander(f"**{ver}** - {data['created']} - {data['samples']:,} samples"):
                st.write(data['description'])
                st.write(f"**Author:** {data['author']}")
                st.write(f"**Accuracy:** {data['accuracy']:.1%}")
        
        st.markdown("---")
        
        # Evolution charts
        versions_list = list(DATASET_VERSIONS.keys())
        accuracies = [DATASET_VERSIONS[v]['accuracy'] for v in versions_list]
        samples_list = [DATASET_VERSIONS[v]['samples'] for v in versions_list]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Model Accuracy Progression', 'Dataset Size Growth'),
            vertical_spacing=0.12
        )
        
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

# Footer
st.markdown("---")
st.markdown("""
<div class="gradient-header">
    <h3>👨‍💻 About This Demo</h3>
    <p><strong>Built for Activeloop by Anju Vilashni Nandhakumar</strong></p>
    <p>📧 nandhakumar.anju@gmail.com</p>
    <p>
        💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white;">LinkedIn</a> | 
        💻 <a href="https://github.com/Av1352" target="_blank" style="color: white;">GitHub</a> | 
        🌐 <a href="https://vxanju.com" target="_blank" style="color: white;">Portfolio</a>
    </p>
    <p><strong>Tech Stack:</strong> Python • Streamlit • Plotly • PIL • Deep Lake Concepts</p>
</div>
""", unsafe_allow_html=True)