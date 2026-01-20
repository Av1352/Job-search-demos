import streamlit as st

st.set_page_config(
    page_title="ML Engineering Portfolio - Anju Vilashni",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simpler, more human CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: #4f46e5;
}

[data-testid="stSidebar"] > div:first-child {
    background: #4f46e5;
}

.sidebar-content {
    padding: 20px;
    color: white;
}

.sidebar-header {
    text-align: center;
    padding: 15px;
    margin-bottom: 20px;
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
}

.hero {
    background: #4f46e5;
    padding: 60px 40px;
    border-radius: 15px;
    color: white;
    margin-bottom: 40px;
}

.hero h1 {
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 15px;
}

.hero p {
    font-size: 20px;
    opacity: 0.9;
}

.section {
    background: #f9fafb;
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 25px;
}

.section h2 {
    font-size: 24px;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 20px;
}

.demo-item {
    background: white;
    padding: 20px;
    margin-bottom: 12px;
    border-radius: 8px;
    border-left: 4px solid #4f46e5;
}

.demo-item h3 {
    font-size: 16px;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 5px;
}

.demo-item p {
    font-size: 14px;
    color: #6b7280;
    line-height: 1.5;
}

.stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    margin: 30px 0;
}

.stat {
    background: white;
    padding: 25px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #e5e7eb;
}

.stat-number {
    font-size: 36px;
    font-weight: 700;
    color: #4f46e5;
}

.stat-label {
    font-size: 14px;
    color: #6b7280;
    margin-top: 5px;
}

@media (max-width: 768px) {
    .hero h1 { font-size: 32px; }
    .stats { grid-template-columns: repeat(2, 1fr); }
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div style="font-size: 36px; margin-bottom: 10px;">👋</div>
        <div style="font-size: 18px; font-weight: 600;">Anju Vilashni</div>
        <div style="font-size: 13px; opacity: 0.8; margin-top: 5px;">MS AI @ Northeastern</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-top: 30px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px; text-align: center;">
        <div style="font-size: 12px; line-height: 1.6;">
            <strong>Contact</strong><br>
            <a href="mailto:nandhakumar.anju@gmail.com" style="color: white; text-decoration: none;">nandhakumar.anju@gmail.com</a>
        </div>
        <div style="margin-top: 12px; font-size: 20px;">
            <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; margin: 0 8px;">💼</a>
            <a href="https://github.com/Av1352" target="_blank" style="color: white; margin: 0 8px;">💻</a>
            <a href="https://vxanju.com" target="_blank" style="color: white; margin: 0 8px;">🌐</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <h1>ML Engineering Portfolio</h1>
    <p>36 custom demos built for AI companies across healthcare, computer vision, MLOps, and more</p>
</div>
""", unsafe_allow_html=True)

# About
st.markdown("""
<div style="background: white; padding: 30px; border-radius: 12px; margin-bottom: 30px;">
    <h2 style="font-size: 24px; font-weight: 600; margin-bottom: 15px;">About</h2>
    <p style="font-size: 16px; color: #374151; line-height: 1.7;">
        I'm Anju, an MS AI student at Northeastern University graduating in May 2025. I specialize in healthcare AI and medical imaging, 
        with experience building production ML systems. Instead of sending resumes, I built 36 custom demos in 15 days to show what I can actually build.
    </p>
</div>
""", unsafe_allow_html=True)

# Stats
st.markdown("""
<div class="stats">
    <div class="stat">
        <div class="stat-number">36</div>
        <div class="stat-label">Custom Demos</div>
    </div>
    <div class="stat">
        <div class="stat-number">15</div>
        <div class="stat-label">Days Built</div>
    </div>
    <div class="stat">
        <div class="stat-number">10</div>
        <div class="stat-label">Domains</div>
    </div>
    <div class="stat">
        <div class="stat-number">100%</div>
        <div class="stat-label">Custom Built</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Healthcare AI Section
st.markdown("""
<div class="section">
    <h2>🏥 Healthcare AI & Computer Vision</h2>
    <div class="demo-item">
        <h3>LabyrinthAI - Manufacturing QC Vision System</h3>
        <p>Real-time defect detection with YOLOv8. 0.94 mAP@0.5 accuracy, optimized for edge deployment.</p>
    </div>
    <div class="demo-item">
        <h3>PathAI - Tumor Detection & Classification</h3>
        <p>ResNet50-based histopathology analysis with 96.2% accuracy. Includes Grad-CAM explainability.</p>
    </div>
    <div class="demo-item">
        <h3>Glass Imaging - Medical Image Enhancement</h3>
        <p>Medical imaging enhancement and analysis for clinical workflows.</p>
    </div>
    <div class="demo-item">
        <h3>Novoflow - Medical Operations Automation</h3>
        <p>AI-powered medical triage and scheduling system.</p>
    </div>
    <div class="demo-item">
        <h3>Paratus Health - AI Intake Assistant</h3>
        <p>Symptom analysis chatbot for patient screening and triage.</p>
    </div>
    <div class="demo-item">
        <h3>Akute Health - Digital EMR Analytics</h3>
        <p>Patient analytics dashboard with EHR integration.</p>
    </div>
    <div class="demo-item">
        <h3>Adentris - Hospital Compliance AI</h3>
        <p>Automated compliance checking for healthcare regulations.</p>
    </div>
    <div class="demo-item">
        <h3>Serif Health - Healthcare Price Prediction</h3>
        <p>ML-based price transparency and cost optimization.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ML Infrastructure
st.markdown("""
<div class="section">
    <h2>🤖 ML Infrastructure & MLOps</h2>
    <div class="demo-item">
        <h3>ClearML - MLOps Platform</h3>
        <p>Experiment tracking and model pipeline automation.</p>
    </div>
    <div class="demo-item">
        <h3>Active Loop - Multi-modal Dataset Management</h3>
        <p>Dataset versioning for multi-modal AI applications.</p>
    </div>
    <div class="demo-item">
        <h3>Centaur AI - Model Quality Assurance</h3>
        <p>Model monitoring and performance drift detection.</p>
    </div>
    <div class="demo-item">
        <h3>Aden Technologies - Agent Observability</h3>
        <p>Real-time monitoring dashboard for AI agents.</p>
    </div>
    <div class="demo-item">
        <h3>Seal - GxP Compliance Platform</h3>
        <p>Data validation and compliance for biotech.</p>
    </div>
    <div class="demo-item">
        <h3>Langbase - AI Code Review Agent</h3>
        <p>Automated code review using AI models.</p>
    </div>
    <div class="demo-item">
        <h3>Nous Research - Distributed RL</h3>
        <p>Reinforcement learning training visualization.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Enterprise AI
st.markdown("""
<div class="section">
    <h2>🏢 Enterprise AI & Agentic Systems</h2>
    <div class="demo-item">
        <h3>Adobe AEP AI - Multi-Agent Marketing</h3>
        <p>Multi-agent system for enterprise marketing automation.</p>
    </div>
    <div class="demo-item">
        <h3>Signal Fire - VC Investment Analysis</h3>
        <p>AI engine for deal sourcing and investment analysis.</p>
    </div>
    <div class="demo-item">
        <h3>Noho Labs - Enterprise AI Solutions</h3>
        <p>Enterprise automation and AI integration.</p>
    </div>
    <div class="demo-item">
        <h3>Flowmentum/Cognara - Agentic Workflows</h3>
        <p>Automated workflow systems with AI agents.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Fintech
st.markdown("""
<div class="section">
    <h2>💰 Fintech & Payments</h2>
    <div class="demo-item">
        <h3>Slash - Smart Payment Routing</h3>
        <p>AI-optimized payment processor selection with 15% fee reduction.</p>
    </div>
    <div class="demo-item">
        <h3>CTGT, Method, Use Dots, Eddi, Alinea Invest</h3>
        <p>Payment infrastructure, financial analytics, and investment platforms.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Voice AI
st.markdown("""
<div class="section">
    <h2>🎙️ Voice & Conversational AI</h2>
    <div class="demo-item">
        <h3>Vapi AI - Voice Agent Platform</h3>
        <p>API-first voice agents with 650ms response time.</p>
    </div>
    <div class="demo-item">
        <h3>Simple AI - Enterprise Phone Agents</h3>
        <p>Automated phone systems for businesses.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sales & Marketing
st.markdown("""
<div class="section">
    <h2>📞 Sales & Marketing AI</h2>
    <div class="demo-item">
        <h3>Hyperbound AI - Sales Call Analysis</h3>
        <p>AI coaching for sales teams with 8-15% win rate improvement.</p>
    </div>
    <div class="demo-item">
        <h3>Conversion AI, Loop AI</h3>
        <p>Marketing automation and delivery optimization.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Other Categories
st.markdown("""
<div class="section">
    <h2>🔧 Testing, Developer Tools & More</h2>
    <div class="demo-item">
        <h3>Testing & QA</h3>
        <p>Decipher AI (automated testing), Spur (AI shopper simulation)</p>
    </div>
    <div class="demo-item">
        <h3>No-Code & Developer Tools</h3>
        <p>Rebolt AI (voice app building), Olive (internal tools), HotGlue (SaaS integrations)</p>
    </div>
    <div class="demo-item">
        <h3>Legal & Identity</h3>
        <p>Dioptra AI (contract negotiation), Spruce ID (decentralized identity)</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="background: white; padding: 30px; border-radius: 12px; text-align: center; margin-top: 40px;">
    <h3 style="font-size: 20px; font-weight: 600; margin-bottom: 15px;">Get in Touch</h3>
    <p style="font-size: 15px; color: #6b7280; margin-bottom: 20px;">
        Open to ML engineering opportunities in healthcare AI, computer vision, and MLOps.
    </p>
    <p style="font-size: 15px;">
        <a href="mailto:nandhakumar.anju@gmail.com" style="color: #4f46e5; text-decoration: none; font-weight: 500;">nandhakumar.anju@gmail.com</a>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 20px; color: #9ca3af; font-size: 13px; margin-top: 30px;">
    <p>Built by Anju Vilashni Nandhakumar • 2025</p>
</div>
""", unsafe_allow_html=True)