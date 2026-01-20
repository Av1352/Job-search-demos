import streamlit as st

st.set_page_config(
    page_title="ML Engineering Demos - Anju Vilashni",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for both main content and sidebar
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}

[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}

/* Sidebar content */
.sidebar-content {
    padding: 20px 10px;
}

.sidebar-header {
    text-align: center;
    padding: 20px 10px;
    margin-bottom: 20px;
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    border: 2px solid rgba(255,255,255,0.2);
}

.sidebar-title {
    font-size: 24px;
    font-weight: 900;
    color: white;
    margin: 10px 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.sidebar-subtitle {
    font-size: 13px;
    color: rgba(255,255,255,0.9);
    font-weight: 600;
}

.nav-section {
    margin: 25px 0;
}

.nav-section-title {
    font-size: 12px;
    font-weight: 800;
    color: rgba(255,255,255,0.7);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    padding: 0 10px;
}

/* Override Streamlit's default link styling in sidebar */
[data-testid="stSidebar"] a {
    text-decoration: none !important;
}

[data-testid="stSidebar"] .stButton button {
    width: 100%;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    color: white;
    border: 2px solid rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 12px 16px;
    font-weight: 700;
    font-size: 14px;
    text-align: left;
    transition: all 0.3s ease;
    margin-bottom: 8px;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.25);
    border-color: rgba(255,255,255,0.4);
    transform: translateX(5px);
}

.sidebar-footer {
    margin-top: 30px;
    padding: 15px;
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    border: 2px solid rgba(255,255,255,0.2);
    text-align: center;
}

.sidebar-footer-text {
    font-size: 11px;
    color: rgba(255,255,255,0.8);
    line-height: 1.6;
}

.stats-badge {
    background: rgba(255,255,255,0.2);
    padding: 8px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    color: white;
    display: inline-block;
    margin: 5px 3px;
    border: 1px solid rgba(255,255,255,0.3);
}

/* Main content styles */
.hero-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 80px 40px;
    border-radius: 30px;
    text-align: center;
    color: white;
    margin-bottom: 40px;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

.hero-title {
    font-size: 72px;
    font-weight: 900;
    margin: 0;
    text-shadow: 0 4px 12px rgba(0,0,0,0.2);
    position: relative;
    z-index: 1;
}

.hero-subtitle {
    font-size: 28px;
    margin: 20px 0;
    font-weight: 700;
    opacity: 0.95;
    position: relative;
    z-index: 1;
}

.hero-description {
    font-size: 18px;
    opacity: 0.85;
    position: relative;
    z-index: 1;
}

.badge-container {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 30px;
    position: relative;
    z-index: 1;
}

.badge {
    background: rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
    padding: 12px 24px;
    border-radius: 25px;
    font-weight: 700;
    font-size: 14px;
    border: 2px solid rgba(255,255,255,0.3);
    transition: all 0.3s ease;
}

.badge:hover {
    background: rgba(255,255,255,0.3);
    transform: translateY(-2px);
}

.about-card {
    background: white;
    padding: 35px;
    border-radius: 20px;
    margin-bottom: 40px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.05);
}

.about-card h2 {
    color: #667eea;
    font-size: 32px;
    font-weight: 900;
    margin-bottom: 15px;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    margin: 40px 0;
}

.stat-card {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    border: 3px solid #3b82f6;
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 30px rgba(59, 130, 246, 0.3);
}

.stat-number {
    font-size: 48px;
    font-weight: 900;
    color: #3b82f6;
    margin: 10px 0;
}

.stat-label {
    font-size: 16px;
    color: #1e40af;
    font-weight: 700;
}

.category-section {
    background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
    padding: 40px;
    border-radius: 25px;
    margin-bottom: 30px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
}

.category-title {
    font-size: 32px;
    font-weight: 900;
    margin-bottom: 25px;
    color: #1f2937;
    display: flex;
    align-items: center;
    gap: 12px;
}

.demo-card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    margin-bottom: 15px;
    border-left: 5px solid #667eea;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
    cursor: pointer;
}

.demo-card:hover {
    transform: translateX(5px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    border-left-color: #764ba2;
}

.demo-title {
    font-size: 20px;
    font-weight: 800;
    color: #1f2937;
    margin-bottom: 8px;
}

.demo-description {
    font-size: 14px;
    color: #6b7280;
    line-height: 1.6;
}

.tech-stack-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 50px 40px;
    border-radius: 25px;
    color: white;
    text-align: center;
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
}

.tech-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-top: 30px;
}

.tech-card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 16px;
    border: 2px solid rgba(255,255,255,0.2);
}

.tech-card h4 {
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 12px;
}

.contact-section {
    background: white;
    padding: 30px;
    border-radius: 20px;
    margin: 30px 0;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    text-align: center;
}

.contact-links {
    display: flex;
    gap: 20px;
    justify-content: center;
    margin-top: 20px;
}

.contact-link {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 28px;
    border-radius: 25px;
    text-decoration: none;
    font-weight: 700;
    font-size: 15px;
    transition: all 0.3s ease;
    display: inline-block;
}

.contact-link:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
}

@media (max-width: 768px) {
    .hero-title { font-size: 42px; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
    .tech-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# Sidebar Content
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div style="font-size: 48px;">🚀</div>
        <div class="sidebar-title">ML Demos</div>
        <div class="sidebar-subtitle">35 Production-Ready Apps</div>
        <div style="margin-top: 15px;">
            <span class="stats-badge">15 Days</span>
            <span class="stats-badge">5 Domains</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation sections
    st.markdown('<div class="nav-section-title">💰 Fintech</div>', unsafe_allow_html=True)
    if st.button("💸 Slash - Payment Intelligence"):
        st.switch_page("pages/slash.py")
    
    st.markdown('<div class="nav-section-title">🔐 Identity & Security</div>', unsafe_allow_html=True)
    if st.button("🔐 Spruce ID - Identity Verification"):
        st.switch_page("pages/spruceID.py")
    
    st.markdown('<div class="nav-section-title">🛒 E-commerce</div>', unsafe_allow_html=True)
    if st.button("🛍️ Spur - AI Shopper Simulation"):
        st.switch_page("pages/spurAI.py")
    
    st.markdown('<div class="nav-section-title">🎙️ Voice AI</div>', unsafe_allow_html=True)
    if st.button("🎙️ Vapi AI - Voice Platform"):
        st.switch_page("pages/vapiAI.py")
    
    st.markdown('<div class="nav-section-title">🏥 Healthcare AI</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 10px; margin: 10px 0; border: 1px solid rgba(255,255,255,0.2);">
        <div style="color: white; font-size: 12px; font-weight: 600; text-align: center;">
            🚧 Coming Soon<br>
            <span style="font-size: 11px; opacity: 0.8;">10 Healthcare Demos</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="nav-section-title">🤖 ML Infrastructure</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 10px; margin: 10px 0; border: 1px solid rgba(255,255,255,0.2);">
        <div style="color: white; font-size: 12px; font-weight: 600; text-align: center;">
            🚧 Coming Soon<br>
            <span style="font-size: 11px; opacity: 0.8;">MLOps & Dev Tools</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer in sidebar
    st.markdown("""
    <div class="sidebar-footer">
        <div class="sidebar-footer-text">
            <strong style="font-size: 13px;">Anju Vilashni</strong><br>
            MS AI @ Northeastern<br>
            Graduating May 2025
        </div>
        <div style="margin-top: 12px;">
            <a href="mailto:nandhakumar.anju@gmail.com" style="color: white; font-size: 20px; margin: 0 8px;">📧</a>
            <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; font-size: 20px; margin: 0 8px;">💼</a>
            <a href="https://github.com/Av1352" target="_blank" style="color: white; font-size: 20px; margin: 0 8px;">💻</a>
            <a href="https://vxanju.com" target="_blank" style="color: white; font-size: 20px; margin: 0 8px;">🌐</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">🚀 ML Engineering Portfolio</h1>
    <p class="hero-subtitle">35 Production-Ready Custom Demos</p>
    <p class="hero-description">Built for leading AI companies • Healthcare AI • MLOps • Developer Tools • Voice AI</p>
    <div class="badge-container">
        <div class="badge">🏥 Healthcare AI</div>
        <div class="badge">🤖 ML Infrastructure</div>
        <div class="badge">💰 Fintech</div>
        <div class="badge">🎙️ Voice AI</div>
        <div class="badge">🛒 E-commerce</div>
    </div>
</div>
""", unsafe_allow_html=True)

# About Section
st.markdown("""
<div class="about-card">
    <h2>👋 Anju Vilashni Nandhakumar</h2>
    <p style="font-size: 18px; color: #1f2937; margin-bottom: 20px;"><strong>MS in Artificial Intelligence @ Northeastern University</strong> (Graduating May 2025)</p>
    <p style="font-size: 16px; color: #6b7280; line-height: 1.8; margin-bottom: 25px;">
        Specializing in Healthcare AI and Medical Imaging. Built 35 custom ML demos in 15 days to demonstrate technical capabilities 
        to target companies through working software instead of traditional resumes.
    </p>
    <div class="contact-links">
        <a href="mailto:nandhakumar.anju@gmail.com" class="contact-link">📧 Email</a>
        <a href="https://linkedin.com/in/anju-vilashni" target="_blank" class="contact-link">💼 LinkedIn</a>
        <a href="https://github.com/Av1352" target="_blank" class="contact-link">💻 GitHub</a>
        <a href="https://vxanju.com" target="_blank" class="contact-link">🌐 Portfolio</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Stats Section
st.markdown("""
<div class="stats-grid">
    <div class="stat-card">
        <div style="font-size: 40px;">📱</div>
        <div class="stat-number">35</div>
        <div class="stat-label">Custom Demos</div>
    </div>
    <div class="stat-card">
        <div style="font-size: 40px;">⚡</div>
        <div class="stat-number">15</div>
        <div class="stat-label">Days Built</div>
    </div>
    <div class="stat-card">
        <div style="font-size: 40px;">🎯</div>
        <div class="stat-number">5</div>
        <div class="stat-label">Domains</div>
    </div>
    <div class="stat-card">
        <div style="font-size: 40px;">✨</div>
        <div class="stat-number">100%</div>
        <div class="stat-label">Custom Built</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Demo Categories
st.markdown("""
<div class="category-section">
    <div class="category-title">
        <span>💰</span> Fintech & Payment Intelligence
    </div>
    <div style="display: grid; gap: 15px;">
        <div class="demo-card">
            <div class="demo-title">Slash - Payment Routing Intelligence</div>
            <div class="demo-description">AI-powered payment optimization and processor selection. Smart routing selects optimal processor per transaction with 15% lower fees.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 View Slash Demo", use_container_width=True):
        st.switch_page("pages/slash.py")

with col2:
    st.info("More fintech demos coming soon...")

st.markdown("""
<div class="category-section">
    <div class="category-title">
        <span>🔐</span> Identity & Security
    </div>
    <div style="display: grid; gap: 15px;">
        <div class="demo-card">
            <div class="demo-title">Spruce ID - Digital Identity Verification</div>
            <div class="demo-description">Decentralized identity verification with cryptographic proof. 2s verification vs days for manual checks, 99% faster than traditional methods.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("🔐 View Spruce ID Demo", use_container_width=True):
        st.switch_page("pages/spruce_id.py")

with col2:
    st.info("More identity demos in development...")

st.markdown("""
<div class="category-section">
    <div class="category-title">
        <span>🛒</span> E-commerce & Testing
    </div>
    <div style="display: grid; gap: 15px;">
        <div class="demo-card">
            <div class="demo-title">Spur - AI Shopper Simulation</div>
            <div class="demo-description">Automated A/B testing with AI shoppers. Test e-commerce changes 10x faster with zero risk. 5 distinct shopper personas, multivariate testing.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("🛍️ View Spur Demo", use_container_width=True):
        st.switch_page("pages/spur.py")

with col2:
    st.info("More e-commerce demos available...")

st.markdown("""
<div class="category-section">
    <div class="category-title">
        <span>🎙️</span> Voice AI & Developer Tools
    </div>
    <div style="display: grid; gap: 15px;">
        <div class="demo-card">
            <div class="demo-title">Vapi AI - Voice AI Platform</div>
            <div class="demo-description">API-first voice agents for developers. Build voice agents in 5 minutes with simple REST API. 650-920ms response time, $0.08 per call vs $2.50 human cost.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("🎙️ View Vapi AI Demo", use_container_width=True):
        st.switch_page("pages/vapi.py")

with col2:
    st.info("More voice AI demos in progress...")

# Coming Soon Section
st.markdown("""
<div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 30px; border-radius: 20px; margin: 30px 0; border: 3px solid #f59e0b;">
    <h3 style="color: #92400e; font-size: 24px; font-weight: 900; margin-bottom: 15px;">🚧 More Demos Coming Soon</h3>
    <p style="color: #78350f; font-size: 16px; line-height: 1.6;">
        <strong>Healthcare AI:</strong> Novoflow, Paratus Health, Akute Health, Adentris, Serif Health<br>
        <strong>ML Infrastructure:</strong> Active Loop, Centaur AI, Aden Technologies, Seal<br>
        <strong>Voice & Conversational AI:</strong> Simple AI, Additional Vapi integrations<br>
        <strong>Testing & QA:</strong> Decipher AI, Spur extensions<br>
        <strong>Developer Tools:</strong> Langbase, ClearML, HotGlue
    </p>
</div>
""", unsafe_allow_html=True)

# Tech Stack Section
st.markdown("""
<div class="tech-stack-section">
    <h2 style="font-size: 36px; font-weight: 900; margin-bottom: 10px;">⚡ Technical Stack</h2>
    <p style="font-size: 16px; opacity: 0.9;">Production-ready tools and frameworks used across all demos</p>
    <div class="tech-grid">
        <div class="tech-card">
            <h4>🎨 Frontend & UI</h4>
            <p style="font-size: 14px; opacity: 0.9;">Streamlit • Gradio • Plotly • Custom CSS</p>
        </div>
        <div class="tech-card">
            <h4>🤖 ML & AI</h4>
            <p style="font-size: 14px; opacity: 0.9;">PyTorch • TensorFlow • Scikit-learn • OpenCV</p>
        </div>
        <div class="tech-card">
            <h4>📊 Data & Analytics</h4>
            <p style="font-size: 14px; opacity: 0.9;">Pandas • NumPy • Matplotlib • Seaborn</p>
        </div>
        <div class="tech-card">
            <h4>🚀 Deployment</h4>
            <p style="font-size: 14px; opacity: 0.9;">Streamlit Cloud • Docker • Netlify • Vercel</p>
        </div>
        <div class="tech-card">
            <h4>🔧 MLOps</h4>
            <p style="font-size: 14px; opacity: 0.9;">MLflow • DVC • ClearML • Weights & Biases</p>
        </div>
        <div class="tech-card">
            <h4>☁️ Cloud & APIs</h4>
            <p style="font-size: 14px; opacity: 0.9;">FastAPI • REST APIs • AWS • Hugging Face</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="contact-section">
    <h3 style="color: #667eea; font-size: 28px; font-weight: 900; margin-bottom: 15px;">📬 Get in Touch</h3>
    <p style="font-size: 16px; color: #6b7280; margin-bottom: 20px;">
        Interested in discussing ML engineering opportunities or learning more about these demos?
    </p>
    <a href="mailto:nandhakumar.anju@gmail.com" class="contact-link" style="margin: 0 auto;">Send me an email →</a>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 20px; color: #6b7280; font-size: 14px; margin-top: 40px;">
    <p>Built with ❤️ by Anju Vilashni Nandhakumar • © 2025 • All demos are original work</p>
</div>
""", unsafe_allow_html=True)