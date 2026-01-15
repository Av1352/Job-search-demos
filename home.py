import streamlit as st

st.set_page_config(
    page_title="ML Engineering Demos - Anju Vilashni",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size: 60px !important;
    font-weight: 900;
    color: white;
}
.gradient-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 60px 30px;
    border-radius: 25px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="gradient-box">
    <h1 class="big-font">ML Engineering Portfolio</h1>
    <p style="font-size: 24px; margin: 20px 0;">35 Production-Ready Custom Demos</p>
    <p style="font-size: 18px; opacity: 0.9;">Built for leading AI companies • Healthcare • MLOps • Developer Tools</p>
</div>
""", unsafe_allow_html=True)

# About section
st.markdown("""
<div style="background: white; padding: 30px; border-radius: 20px; margin-bottom: 30px;">
    <h2>Anju Vilashni Nandhakumar</h2>
    <p><strong>MS AI @ Northeastern University</strong> (May 2025)</p>
    <p>📧 nandhakumar.anju@gmail.com | 
    💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank">LinkedIn</a> | 
    💻 <a href="https://github.com/Av1352" target="_blank">GitHub</a> | 
    🌐 <a href="https://vxanju.com" target="_blank">Portfolio</a></p>
</div>
""", unsafe_allow_html=True)

# Stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Applications", "35")
with col2:
    st.metric("Days", "15")
with col3:
    st.metric("Domains", "5")
with col4:
    st.metric("Custom Built", "100%")

st.markdown("---")

# Demo categories
st.header("🏥 Healthcare AI (10 Demos)")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/activeLoop.py", label="**Activeloop** - Multi-Modal Dataset Versioning")
    # Add more as you convert them
with col2:
    st.write("More demos coming soon...")

st.markdown("---")

st.header("🤖 ML Infrastructure")
st.page_link("pages/activeLoop.py", label="**Activeloop** - Dataset Version Control")

st.markdown("---")

# Footer
st.markdown("""
<div class="gradient-box">
    <h3>Tech Stack</h3>
    <p><strong>Frameworks:</strong> Python, Streamlit, Gradio, FastAPI</p>
    <p><strong>ML/Data:</strong> PyTorch, TensorFlow, Plotly, Pandas</p>
    <p><strong>Deployment:</strong> Streamlit Cloud, Hugging Face Spaces, Docker</p>
</div>
""", unsafe_allow_html=True)