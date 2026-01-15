import streamlit as st

st.set_page_config(
    page_title="ML Engineering Demos - Anju Vilashni",
    page_icon="🚀",
    layout="wide"
)

# Test 1: Does basic markdown work?
st.title("🚀 ML Engineering Portfolio")
st.subheader("35 Production-Ready Custom Demos")

# Test 2: Does simple HTML work?
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 20px; color: white; text-align: center;">
    <h1>This should be purple with white text</h1>
</div>
""", unsafe_allow_html=True)

# Test 3: Basic info
st.write("**Anju Vilashni Nandhakumar**")
st.write("MS AI @ Northeastern University (May 2025)")
st.write("📧 nandhakumar.anju@gmail.com")

# Test navigation
st.markdown("---")
st.subheader("🏥 Healthcare AI Demos")
st.page_link("pages/activeLoop.py", label="Activeloop - Dataset Versioning")