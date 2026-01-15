import streamlit as st

# Page config
st.set_page_config(
    page_title="ML Engineering Demos - Anju Vilashni",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
}
.demo-card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    border-left: 5px solid;
    transition: transform 0.2s;
}
.demo-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.12);
}
.healthcare { border-left-color: #10b981; }
.infrastructure { border-left-color: #3b82f6; }
.devtools { border-left-color: #8b5cf6; }
.fintech { border-left-color: #f59e0b; }
.voice { border-left-color: #ec4899; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; padding: 60px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 40px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
    <h1 style="font-size: 56px; font-weight: 900; color: white; margin: 0 0 20px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        ML Engineering Portfolio
    </h1>
    <p style="font-size: 24px; color: rgba(255,255,255,0.95); font-weight: 600; margin: 15px 0;">
        35 Production-Ready Custom Demos
    </p>
    <p style="font-size: 18px; color: rgba(255,255,255,0.85); margin-bottom: 30px;">
        Built for leading AI companies • Healthcare • MLOps • Developer Tools
    </p>
    
    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 24px; max-width: 600px; margin: 0 auto; border: 1px solid rgba(255,255,255,0.2);">
        <p style="color: white; font-size: 18px; font-weight: 700; margin: 8px 0;">
            Anju Vilashni Nandhakumar
        </p>
        <p style="color: rgba(255,255,255,0.9); font-size: 16px; margin: 8px 0;">
            MS AI @ Northeastern University (May 2025)
        </p>
        <p style="color: rgba(255,255,255,0.9); font-size: 15px; margin: 15px 0 8px 0;">
            📧 nandhakumar.anju@gmail.com
        </p>
        <p style="color: rgba(255,255,255,0.9); font-size: 15px; margin: 8px 0;">
            💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; font-weight: 600; text-decoration: none;">LinkedIn</a> | 
            💻 <a href="https://github.com/Av1352" target="_blank" style="color: white; font-weight: 600; text-decoration: none;">GitHub</a> | 
            🌐 <a href="https://vxanju.com" target="_blank" style="color: white; font-weight: 600; text-decoration: none;">Portfolio</a>
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Intro section
st.markdown("""
<div style="background: white; border-radius: 20px; padding: 32px; margin-bottom: 40px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
    <h2 style="color: #1f2937; font-size: 32px; font-weight: 800; margin: 0 0 20px 0;">
        About This Portfolio
    </h2>
    <p style="color: #4b5563; font-size: 16px; line-height: 1.8; margin-bottom: 15px;">
        Over 15 days (Dec 19 - Jan 4, 2025), I built <strong>35 custom ML applications</strong> for specific companies, 
        demonstrating deep understanding of their technology stacks and business problems. Each demo showcases 
        production-ready code, thoughtful architecture, and domain expertise.
    </p>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-top: 25px;">
        <div style="text-align: center;">
            <p style="font-size: 36px; color: #667eea; font-weight: 900; margin: 0;">35</p>
            <p style="font-size: 14px; color: #6b7280; font-weight: 600; margin: 5px 0 0 0;">Applications</p>
        </div>
        <div style="text-align: center;">
            <p style="font-size: 36px; color: #10b981; font-weight: 900; margin: 0;">15</p>
            <p style="font-size: 14px; color: #6b7280; font-weight: 600; margin: 5px 0 0 0;">Days</p>
        </div>
        <div style="text-align: center;">
            <p style="font-size: 36px; color: #f59e0b; font-weight: 900; margin: 0;">5</p>
            <p style="font-size: 14px; color: #6b7280; font-weight: 600; margin: 5px 0 0 0;">Domains</p>
        </div>
        <div style="text-align: center;">
            <p style="font-size: 36px; color: #ec4899; font-weight: 900; margin: 0;">100%</p>
            <p style="font-size: 14px; color: #6b7280; font-weight: 600; margin: 5px 0 0 0;">Custom Built</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Demo categories
st.markdown("<h2 style='color: #1f2937; font-size: 36px; font-weight: 800; margin: 40px 0 30px 0;'>Select a Demo</h2>", unsafe_allow_html=True)

# Healthcare AI
with st.expander("🏥 Healthcare AI (10 Demos)", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.page_link("pages/16_Glass_Imaging.py", label="**Glass Imaging** - Medical Image Enhancement", icon="🏥")
        st.page_link("pages/26_PathAI.py", label="**PathAI** - Pathology Analysis", icon="🔬")
        st.page_link("pages/23_Novoflow.py", label="**Novoflow** - Medical Operations Automation", icon="⚕️")
        st.page_link("pages/25_Paratus_Health.py", label="**Paratus Health** - AI Intake Assistant", icon="📋")
        st.page_link("pages/05_Akute_Health.py", label="**Akute Health** - EMR for Digital Health", icon="💊")
    
    with col2:
        st.page_link("pages/03_Adentris.py", label="**Adentris** - AI Compliance for Hospitals", icon="✅")
        st.page_link("pages/29_Serif_Health.py", label="**Serif Health** - Healthcare Price Transparency", icon="💰")
        st.page_link("pages/28_Seal.py", label="**Seal** - GxP Platform", icon="🔒")

# ML Infrastructure
with st.expander("🤖 ML Infrastructure (8 Demos)", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.page_link("pages/01_Activeloop.py", label="**Activeloop** - Multi-Modal Dataset Versioning", icon="🗂️")
        st.page_link("pages/08_ClearML.py", label="**ClearML** - ML Experiment Tracking", icon="📊")
        st.page_link("pages/07_Centaur.py", label="**Centaur AI** - AI Quality Assurance", icon="✨")
        st.page_link("pages/02_Aden_Tech.py", label="**Aden Technologies** - AI Observability", icon="👁️")
    
    with col2:
        st.page_link("pages/22_Nous_Research.py", label="**Nous Research** - Distributed RL Training", icon="🧠")

# Developer Tools
with st.expander("🔧 Developer Tools & Enterprise AI (7 Demos)", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.page_link("pages/04_Adobe.py", label="**Adobe AEP** - Multi-Agent Marketing Assistant", icon="🎨")
        st.page_link("pages/27_Rebolt_AI.py", label="**Rebolt AI** - Build Apps with AI", icon="🚀")
        st.page_link("pages/24_Olive.py", label="**Olive** - Build Internal Tools with NLP", icon="🛠️")
        st.page_link("pages/17_HotGlue.py", label="**HotGlue** - SaaS Integrations", icon="🔌")
    
    with col2:
        st.page_link("pages/21_Noho_Labs.py", label="**Noho Labs** - Enterprise AI", icon="🏢")
        st.page_link("pages/30_Signal_Fire.py", label="**Signal Fire** - VC AI Engine", icon="💡")

# Voice & Conversational AI
with st.expander("🗣️ Voice & Conversational AI (3 Demos)", expanded=False):
    st.page_link("pages/31_Simple_AI.py", label="**Simple AI** - Enterprise Phone Agents", icon="📞")
    st.page_link("pages/35_Vapi.py", label="**Vapi AI** - Voice AI for Developers", icon="🎙️")

# Testing & QA
with st.expander("🧪 Testing & QA (3 Demos)", expanded=False):
    st.page_link("pages/12_Decipher_AI.py", label="**Decipher AI** - Automated Testing", icon="🔍")
    st.page_link("pages/34_Spur.py", label="**Spur** - AI Shopper Simulation", icon="🛒")

# Sales & Marketing
with st.expander("📊 Sales & Marketing AI (3 Demos)", expanded=False):
    st.page_link("pages/18_Hyperbound_AI.py", label="**Hyperbound AI** - Sales Call Analysis", icon="📈")
    st.page_link("pages/10_Conversion_AI.py", label="**Conversion AI** - Marketing Automation", icon="🎯")
    st.page_link("pages/19_Loop_AI.py", label="**Loop AI** - Food Delivery Intelligence", icon="🍕")

# Fintech
with st.expander("💰 Fintech (6 Demos)", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.page_link("pages/11_CTGT.py", label="**CTGT** - Fintech Solutions", icon="💳")
        st.page_link("pages/32_Slash.py", label="**Slash** - Payment Platform", icon="💸")
        st.page_link("pages/20_Method.py", label="**Method** - Financial Infrastructure", icon="🏦")
    
    with col2:
        st.page_link("pages/14_Dots.py", label="**Dots** - Payment APIs", icon="🔗")
        st.page_link("pages/15_Eddi.py", label="**Eddi** - Financial Tools", icon="📱")
        st.page_link("pages/06_Alinea.py", label="**Alinea Invest** - Investment Platform", icon="📈")

# Legal & Identity
with st.expander("🔐 Legal & Identity (2 Demos)", expanded=False):
    st.page_link("pages/13_Dioptra_AI.py", label="**Dioptra AI** - Contract Negotiation AI", icon="📄")
    st.page_link("pages/33_Spruce_ID.py", label="**Spruce ID** - Digital Identity", icon="🆔")

# Footer
st.markdown("<hr style='margin: 60px 0 40px 0; border: 2px solid #e5e7eb;'>", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 40px; text-align: center; color: white;">
    <h3 style="font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">Tech Stack</h3>
    <p style="font-size: 16px; margin: 10px 0; line-height: 2;">
        <strong>Frameworks:</strong> Python, Streamlit, Gradio, FastAPI<br>
        <strong>ML/Data:</strong> PyTorch, TensorFlow, Plotly, Pandas<br>
        <strong>Specialized:</strong> Computer Vision, NLP, MLOps, Multi-Agent Systems<br>
        <strong>Deployment:</strong> Streamlit Cloud, Hugging Face Spaces, Docker
    </p>
    <p style="font-size: 15px; margin-top: 25px; font-style: italic; opacity: 0.9;">
        Each demo demonstrates production-ready code, clean architecture, and deep understanding of the target company's domain.
    </p>
</div>
""", unsafe_allow_html=True)