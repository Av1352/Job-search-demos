import streamlit as st
import base64
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Anju Vilashni - ML Engineer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebarNav"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

def get_logo_base64():
    """Load logo and convert to base64"""
    logo_path = Path(__file__).parent / "assets" / "logo.png"
    with open(logo_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# Load logo
logo_base64 = get_logo_base64()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a1a 0%, #2d7a5f 100%);
}

[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #1a1a1a 0%, #2d7a5f 100%);
}

.hero {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d7a5f 50%, #73BA9B 100%);
    padding: 70px 50px;
    border-radius: 20px;
    margin-bottom: 50px;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 60%;
    height: 200%;
    background: radial-gradient(circle, rgba(115, 186, 155, 0.15) 0%, transparent 70%);
}

.hero-content {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 30px;
}

.logo-container {
    flex-shrink: 0;
}

.logo-container img {
    width: 120px;
    height: 120px;
    filter: brightness(0) invert(1);
    opacity: 0.95;
}

.hero-text {
    flex: 1;
}

.hero h1 {
    color: white;
    font-size: 52px;
    font-weight: 800;
    margin: 0 0 20px 0;
    letter-spacing: -1px;
    line-height: 1.2;
}

.hero p {
    color: rgba(255,255,255,0.9);
    font-size: 20px;
    line-height: 1.6;
    max-width: 800px;
}

.content-section {
    background: white;
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 35px;
    border-left: 5px solid #73BA9B;
}

.content-section h2 {
    font-size: 28px;
    font-weight: 700;
    color: #1f2937;
    margin: 0 0 20px 0;
}

.content-section h3 {
    font-size: 20px;
    font-weight: 600;
    color: #374151;
    margin: 25px 0 15px 0;
}

.content-section p {
    font-size: 17px;
    line-height: 1.7;
    color: #4b5563;
    margin: 0 0 15px 0;
}

.content-section ul {
    font-size: 17px;
    line-height: 1.8;
    color: #4b5563;
}

.highlight-box {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border-left: 4px solid #73BA9B;
    padding: 25px;
    border-radius: 12px;
    margin: 25px 0;
}

.highlight-box h3 {
    color: #065f46;
    margin: 0 0 15px 0;
    font-size: 20px;
    font-weight: 700;
}

.highlight-box p {
    color: #047857;
    margin: 0;
    font-size: 16px;
    line-height: 1.6;
}

.demo-showcase {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
    margin: 30px 0;
}

.demo-card {
    background: #f8fafc;
    border-radius: 12px;
    padding: 25px;
    border-left: 4px solid #73BA9B;
}

.demo-card h4 {
    font-size: 18px;
    font-weight: 700;
    color: #1e293b;
    margin: 0 0 10px 0;
}

.demo-card p {
    font-size: 15px;
    color: #64748b;
    margin: 0;
    line-height: 1.6;
}

.tech-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin: 25px 0;
}

.tech-item {
    background: #f8fafc;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}

.tech-item h4 {
    font-size: 16px;
    font-weight: 600;
    color: #1e293b;
    margin: 0 0 10px 0;
}

.tech-item p {
    font-size: 14px;
    color: #64748b;
    margin: 0;
    line-height: 1.5;
}

.contact-box {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    margin: 50px 0 30px 0;
}

.contact-links {
    display: flex;
    gap: 15px;
    justify-content: center;
    margin-top: 25px;
    flex-wrap: wrap;
}

.contact-link {
    background: #73BA9B;
    color: white;
    padding: 12px 28px;
    border-radius: 10px;
    text-decoration: none;
    font-weight: 600;
    font-size: 15px;
    transition: all 0.2s ease;
    display: inline-block;
}

.contact-link:hover {
    background: #5fa385;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(115, 186, 155, 0.3);
}

@media (max-width: 768px) {
    .hero h1 { font-size: 36px; }
    .hero-content { flex-direction: column; text-align: center; }
    .logo-container img { width: 80px; height: 80px; }
    .demo-showcase { grid-template-columns: 1fr; }
    .tech-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 25px 15px; background: rgba(255,255,255,0.1); border-radius: 12px; margin-bottom: 25px;">
        <div style="margin-bottom: 15px;">
            <img src="data:image/png;base64,{logo_base64}" alt="AV Logo" style="width: 60px; height: 60px; filter: brightness(0) invert(1); opacity: 0.95;">
        </div>
        <div style="color: white; font-size: 20px; font-weight: 700; margin-bottom: 5px;">Anju Vilashni</div>
        <div style="color: rgba(255,255,255,0.85); font-size: 14px;">ML Engineer</div>
        <div style="color: rgba(255,255,255,0.75); font-size: 13px; margin-top: 8px;">56 Demos • 11 Domains</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search bar
    search_query = st.text_input("🔍 Search demos", placeholder="Type company name...", label_visibility="collapsed")
    
    # All demos organized by category
    categories = {
        "🏥 Healthcare AI & Biotech": [
            ("🏭 LabyrinthAI", "labyrinthAI"),
            ("🔬 PathAI", "pathAI"),
            ("🔍 Glass Imaging", "glass_imaging"),
            ("🏥 Novoflow", "novoflow"),
            ("🩺 Paratus Health", "paratus"),
            ("📋 Akute Health", "akuteHealth"),
            ("🏥 Adentris", "adentris"),
            ("💰 Serif Health", "serif_health"),
            ("🏥 Rovi Health", "roviHealth"),
            ("🧬 Blank Bio", "blankBio"),
            ("🦷 Toothy AI", "toothyAI")
        ],
        "🤖 ML Infrastructure & MLOps": [
            ("📊 ClearML", "clearML"),
            ("🗄️ Active Loop", "activeLoop"),
            ("🎯 Centaur AI", "centaur"),
            ("👁️ Aden Technologies", "adenTech"),
            ("🔒 Seal", "seal"),
            ("🚀 Langbase", "langbase"),
            ("🧠 Nous Research", "nous"),
            ("🤖 Everest", "everest"),
            ("📊 Confident AI", "confidentAI"),
            ("🔬 AfterQuery", "afterQuery"),
            ("🎮 hud", "hud")
        ],
        "🏢 Enterprise AI & Agentic Systems": [
            ("🎨 Adobe AEP AI", "adobe"),
            ("📈 Signal Fire", "signalFire"),
            ("🔬 Noho Labs", "nohoLabs"),
            ("🤖 Flowmentum/Cognara", "cognara")
        ],
        "💰 Fintech & Payments": [
            ("💸 Slash", "slash"),
            ("💳 CTGT", "ctgt"),
            ("🔗 Method", "method"),
            ("📊 Use Dots", "dots"),
            ("💼 Eddi", "eddi"),
            ("📈 Alinea Invest", "alinea"),
            ("💰 Autonomous Tech", "autonomousTech")
        ],
        "🎙️ Voice & Conversational AI": [
            ("🎙️ Vapi AI", "vapiAI"),
            ("📞 Simple AI", "simpleAI"),
            ("🎙️ careCycle", "careCycle")
        ],
        "📞 Sales & Marketing AI": [
            ("📞 Hyperbound AI", "hyperboundAI"),
            ("📈 Conversion AI", "conversionAI"),
            ("🍕 Loop AI", "loopAI")
        ],
        "📄 Document AI & Parsing": [
            ("📄 Unsiloed AI", "unsiloedAI")
        ],
        "🧪 Testing & E-commerce": [
            ("🧪 Decipher AI", "decipherAI"),
            ("🛍️ Spur", "spurAI")
        ],
        "🔧 Developer Tools & Operations": [
            ("🗣️ Rebolt AI", "reboltAI"),
            ("🌿 Olive", "olive"),
            ("🔗 HotGlue", "hotGlue"),
            ("💻 OpenBuilder", "openBuilder"),
            ("🏗️ Semble AI", "sembleAI")
        ],
        "🔐 Legal & Identity": [
            ("📄 Dioptra AI", "dioptraAI"),
            ("🔐 Spruce ID", "spruceID")
        ]
    }
    
    # Filter categories based on search
    for category, demos in categories.items():
        # Filter demos in this category
        if search_query:
            filtered_demos = [(name, page) for name, page in demos if search_query.lower() in name.lower()]
        else:
            filtered_demos = demos
        
        # Only show category if it has matching demos
        if filtered_demos:
            with st.expander(f"{category} ({len(filtered_demos)})", expanded=(search_query != "")):
                for demo_name, demo_page in filtered_demos:
                    if st.button(demo_name, key=demo_page):
                        st.switch_page(f"pages/{demo_page}.py")
    
    st.markdown("""
    <div style="padding: 20px 15px; background: rgba(255,255,255,0.08); border-radius: 10px; margin-top: 20px;">
        <div style="color: rgba(255,255,255,0.85); font-size: 13px; font-weight: 600; margin-bottom: 15px;">CONTACT</div>
        <div style="color: white; font-size: 13px; line-height: 1.8; word-break: break-all;">
            <a href="mailto:nandhakumar.anju@gmail.com" style="color: rgba(255,255,255,0.9); text-decoration: none;">nandhakumar.anju@gmail.com</a>
        </div>
        <div style="margin-top: 18px; font-size: 24px; display: flex; gap: 12px; justify-content: center;">
            <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; text-decoration: none;">💼</a>
            <a href="https://github.com/Av1352" target="_blank" style="color: white; text-decoration: none;">💻</a>
            <a href="https://vxanju.com" target="_blank" style="color: white; text-decoration: none;">🌐</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Hero
st.markdown(f"""
<div class="hero">
    <div class="hero-content">
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" alt="AV Logo">
        </div>
        <div class="hero-text">
            <h1>I'm Anju, an ML engineer specializing in healthcare AI and medical imaging</h1>
            <p>
                I build production ML systems that solve real clinical problems. My focus is on computer vision for medical imaging, 
                pathology analysis, and healthcare workflows. I believe in showing what I can build rather than just talking about it.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# About
st.markdown("""
<div class="content-section">
    <h2>About Me</h2>
    <p>
        I recently completed my Master's in Artificial Intelligence at Northeastern University (graduated May 2025). 
        Before grad school, I worked on machine learning projects in medical imaging, achieving 95%+ accuracy on tumor classification tasks.
    </p>
    <p>
        What excites me most is the intersection of ML and healthcare—building systems that can actually help clinicians make better decisions, 
        catch diseases earlier, and improve patient outcomes. I'm drawn to problems where getting it right really matters.
    </p>
    <h3>My Approach</h3>
    <p>
        Instead of sending resumes, I built 56 custom ML demos for companies I want to work with. 
        Each demo is tailored to a specific company's product and shows what I could contribute. 
        It's not about volume—it's about demonstrating that I understand the problem space and can build solutions.
    </p>
</div>
""", unsafe_allow_html=True)

# Expertise
st.markdown("""
<div class="content-section">
    <h2>What I'm Good At</h2>
    <div class="highlight-box">
        <h3>🏥 Healthcare AI & Medical Imaging</h3>
        <p>
            Computer vision for pathology slides, tumor detection with 96%+ accuracy, medical image enhancement, 
            clinical workflow automation. I understand both the ML and the clinical context—what makes a good prediction clinically useful.
        </p>
    </div>
    <h3>Core Technical Skills</h3>
    <div class="tech-grid">
        <div class="tech-item">
            <h4>Computer Vision</h4>
            <p>YOLOv8, ResNet, EfficientNet, object detection, image segmentation, transfer learning</p>
        </div>
        <div class="tech-item">
            <h4>ML Engineering</h4>
            <p>PyTorch, TensorFlow, model optimization, edge deployment, production pipelines</p>
        </div>
        <div class="tech-item">
            <h4>MLOps</h4>
            <p>Experiment tracking, model versioning, monitoring, CI/CD for ML, deployment automation</p>
        </div>
    </div>
    <h3>What I Care About</h3>
    <ul>
        <li><strong>Real-world impact:</strong> Building systems that actually get deployed and used, not just research projects</li>
        <li><strong>Production quality:</strong> Models that work reliably in the real world, with proper error handling and monitoring</li>
        <li><strong>Explainability:</strong> Especially in healthcare, understanding why a model makes a prediction is as important as the prediction itself</li>
        <li><strong>Fast execution:</strong> I built 56 demos in 20 days because I bias toward shipping and iterating quickly</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Example Work
st.markdown("""
<div class="content-section">
    <h2>Example Work</h2>
    <p>Here are a few demos that showcase different aspects of what I can build:</p>
    <div class="demo-showcase">
        <div class="demo-card">
            <h4>🔬 PathAI - Tumor Detection System</h4>
            <p>
                ResNet50 trained on histopathology images. 96.2% accuracy, Grad-CAM explainability to show which 
                regions the model focuses on, clinical metrics integration. Shows I can build medical imaging systems 
                that clinicians would actually trust.
            </p>
        </div>
        <div class="demo-card">
            <h4>🏭 LabyrinthAI - Manufacturing QC</h4>
            <p>
                Real-time defect detection with YOLOv8. 0.94 mAP@0.5 accuracy, optimized for edge deployment 
                (<500ms inference). Shows I can build computer vision systems for production environments beyond just healthcare.
            </p>
        </div>
        <div class="demo-card">
            <h4>📊 Confident AI - LLM Observability</h4>
            <p>
                Real-time monitoring for LLM applications. Tracks accuracy, latency, hallucination rate with 578 automated tests.
                Shows I understand MLOps, production monitoring, and quality assurance for AI systems.
            </p>
        </div>
        <div class="demo-card">
            <h4>🎮 hud - RL Environment Platform</h4>
            <p>
                Custom RL environment builder with 5 algorithms (DQN, PPO, SAC). Real-time training dashboards and 
                comprehensive evaluation suite. Shows I can work with reinforcement learning and agent systems.
            </p>
        </div>
    </div>
    <p style="font-size: 15px; color: #6b7280; margin-top: 20px;">
        All 56 demos are in the sidebar. Each one is custom-built for a specific company to show I understand their product and could contribute from day one.
    </p>
</div>
""", unsafe_allow_html=True)

# What I'm Looking For
st.markdown("""
<div class="content-section">
    <h2>What I'm Looking For</h2>
    <p>
        I'm looking for full-time ML engineering roles where I can:
    </p>
    <ul>
        <li>Work on healthcare AI or medical imaging systems (my main interest)</li>
        <li>Build production ML systems that get deployed and used</li>
        <li>Work with a team that ships fast and iterates based on real feedback</li>
        <li>Contribute to both the ML and the engineering side—training models and building the systems around them</li>
    </ul>
    <p>
        I'm on F-1 status and will need visa sponsorship. I'm specifically interested in companies that are building in healthcare, 
        or have strong computer vision/ML infrastructure challenges.
    </p>
</div>
""", unsafe_allow_html=True)

# Contact
st.markdown("""
<div class="contact-box">
    <h3 style="font-size: 26px; font-weight: 700; color: #065f46; margin: 0 0 15px 0;">Let's talk</h3>
    <p style="font-size: 16px; color: #047857; max-width: 600px; margin: 0 auto;">
        If you're working on healthcare AI, medical imaging, or interesting ML infrastructure problems, I'd love to hear from you.
    </p>
    <div class="contact-links">
        <a href="mailto:nandhakumar.anju@gmail.com" class="contact-link">📧 Email Me</a>
        <a href="https://linkedin.com/in/anju-vilashni" target="_blank" class="contact-link">💼 LinkedIn</a>
        <a href="https://github.com/Av1352" target="_blank" class="contact-link">💻 GitHub</a>
        <a href="https://vxanju.com" target="_blank" class="contact-link">🌐 Portfolio</a>
    </div>
</div>
""", unsafe_allow_html=True)