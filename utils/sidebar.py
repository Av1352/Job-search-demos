"""
Shared sidebar component for all demo pages
"""

import streamlit as st
import base64
from pathlib import Path

def get_logo_base64():
    try:
        logo_path = Path(__file__).parent.parent / "assets" / "logo.png"
        with open(logo_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

def render_sidebar():
    st.markdown("""
    <style>
    [data-testid="stSidebarNav"] { display: none; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a1a 0%, #2d7a5f 100%); }
    [data-testid="stSidebar"] > div:first-child { background: linear-gradient(180deg, #1a1a1a 0%, #2d7a5f 100%); }
    </style>
    """, unsafe_allow_html=True)

    logo_base64 = get_logo_base64()

    with st.sidebar:
        if st.button("← Back to Home", use_container_width=True, key="nav_home"):
            st.switch_page("home.py")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        if logo_base64:
            st.markdown(f"""
            <div style="text-align: center; padding: 25px 15px; background: rgba(255,255,255,0.1); border-radius: 12px; margin-bottom: 25px;">
                <div style="margin-bottom: 15px;">
                    <img src="data:image/png;base64,{logo_base64}" alt="AV Logo" style="width: 60px; height: 60px; filter: brightness(0) invert(1); opacity: 0.95;">
                </div>
                <div style="color: white; font-size: 20px; font-weight: 700; margin-bottom: 5px;">Anju Vilashni</div>
                <div style="color: rgba(255,255,255,0.85); font-size: 14px;">ML Engineer</div>
                <div style="color: rgba(255,255,255,0.75); font-size: 13px; margin-top: 8px;">110 Demos • 11 Domains</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 25px 15px; background: rgba(255,255,255,0.1); border-radius: 12px; margin-bottom: 25px;">
                <div style="font-size: 51px; margin-bottom: 12px;">👩‍💻</div>
                <div style="color: white; font-size: 20px; font-weight: 700; margin-bottom: 5px;">Anju Vilashni</div>
                <div style="color: rgba(255,255,255,0.85); font-size: 14px;">ML Engineer</div>
                <div style="color: rgba(255,255,255,0.75); font-size: 13px; margin-top: 8px;">110 Demos • 11 Domains</div>
            </div>
            """, unsafe_allow_html=True)

        search_query = st.text_input("🔍 Search demos", placeholder="Type company name...", label_visibility="collapsed")

        categories = {
            "🏥 Healthcare AI & Biotech": [
                ("🔍 Glass Imaging",          "glassImaging"),
                ("🏥 Novoflow",               "novoflow"),
                ("🩺 Paratus Health",          "paratus"),
                ("📋 Akute Health",            "akuteHealth"),
                ("🏥 Adentris",               "adentris"),
                ("💰 Serif Health",            "serifHealth"),
                ("🔬 PathAI",                 "pathAI"),
                ("🏥 Rovi Health",             "roviHealth"),
                ("🧬 Blank Bio",              "blankBio"),
                ("🚑 CareSwift",              "careSwift"),
                ("🧪 Novaflow",               "novaflow"),
                ("🦷 Toothy AI",              "toothyAI"),
                ("📄 Sample Healthcare",       "sampleHealthcare"),
                ("📋 Trellis AI",             "trellisAI"),
                ("🧬 Output Biosciences",      "outputBio"),
                ("🧠 Legion Health",           "legionHealth"),
                ("🩻 Mecha Health",            "mechaHealth"),
                ("🔬 Nucleo",                 "nucleo"),
                ("🧪 Double Blind Bio",        "doubleBlindBio"),
                ("⚗️ Phases",                 "phases"),
                ("🏥 Parachute",              "parachute"),
                ("🏥 Wedge",                  "wedge"),
                ("📋 Ruma Care",              "rumaCare"),
                ("🩺 Prana",                  "prana"),
                ("🔄 Locata",                 "locata"),
                ("🛡️ Aegis",                  "aegis"),
                ("💳 Avelis Health",           "avelisHealth"),
                ("🤖 Galen AI",               "galenAI"),
                ("💚 Nori",                   "nori"),
                ("🎀 Play Health",             "playHealth"),
                ("🟡 Saffron Health",          "saffronHealth"),
                ("🏥 Kaigo Health",            "kaigoHealth"),
                ("🤰 Delfina",                "delfina"),
                ("🏥 Tala Health",             "talaHealth"),
                ("💼 Avenir AI",              "avenirAI"),
                ("🥗 Revero",                 "revero"),
                ("🌻 Sunflower",              "sunflower"),
                ("❤️ Wearlinq",               "wearlinq"),
                ("💳 Logical Health",          "logicalHealth"),
                ("🌀 Swing Therapeutics",      "swingTherapeutics"),
                ("💪 Sweatpals",              "sweatpals"),
                ("🧬 The Generation Lab",      "generationLab"),
                ("⚡ Superpower",             "superpower"),
                ("📋 Avallon AI",             "avallonAI"),
            ],
            "👁️ Computer Vision & Robotics": [
                ("🏭 LabyrinthAI",            "labyrinthAI"),
                ("📦 dScribe AI",             "dScribeAI"),
                ("🎥 OnDeck AI",              "onDeckAI"),
                ("🤖 Verne Robotics",          "verneRobotics"),
                ("🏗️ Bild AI",                "bildAI"),
                ("✈️ Enhanced Radar",          "enhancedRadar"),
                ("🎥 Luma AI",                "lumaAI"),
                ("🤖 Revise Robotics",         "reviseRobotics"),
                ("💪 Tempo",                  "tempo"),
                ("🏗️ Bedrock Robotics",        "bedrockRobotics"),
                ("🤖 Omni Instrument",         "omniInstrument"),
            ],
            "🤖 ML Infrastructure & MLOps": [
                ("📊 ClearML",                "clearML"),
                ("🧠 Nous Research",           "nous"),
                ("🗄️ Active Loop",             "activeLoop"),
                ("🎯 Centaur AI",             "centaur"),
                ("👁️ Aden Technologies",       "adenTech"),
                ("🔒 Seal",                   "seal"),
                ("🤖 Everest",                "everest"),
                ("🎮 Halluminate",            "halluminate"),
                ("⚡ Wafer",                  "wafer"),
                ("🛡️ Metis",                  "metis"),
                ("📊 Confident AI",            "confidentAI"),
                ("🔬 AfterQuery",             "afterQuery"),
                ("🎮 hud",                    "hud"),
                ("🔧 Weave",                  "weave"),
                ("⚡ Modal",                  "modal"),
                ("🔥 Fireworks AI",            "fireworksAI"),
                ("🌐 Airweave",               "airweave"),
            ],
            "🏢 Enterprise AI & Agentic Systems": [
                ("🎨 Adobe AEP AI",            "adobe"),
                ("📈 Signal Fire",             "signalFire"),
                ("🔬 Noho Labs",              "nohoLabs"),
                ("🤖 Flowmentum/Cognara",      "cognara"),
                ("🏭 Candid Intelligence",     "candidIntelligence"),
                ("🏭 Dryft",                  "dryft"),
            ],
            "💰 Fintech & Payments": [
                ("💸 Slash",                  "slash"),
                ("💳 CTGT",                   "ctgt"),
                ("🔗 Method",                 "method"),
                ("📊 Use Dots",               "dots"),
                ("💼 Eddi",                   "eddi"),
                ("📈 Alinea Invest",           "alinea"),
                ("💰 Autonomous Tech",         "autonomousTech"),
                ("🛡️ Navasana",               "navasanaCyberRisk"),
            ],
            "🎙️ Voice & Conversational AI": [
                ("📞 Simple AI",              "simpleAI"),
                ("🎙️ Vapi AI",                "vapiAI"),
                ("🎙️ careCycle",              "careCycle"),
                ("🎙️ LunaBill",               "lunaBill"),
                ("🌐 Opalite Health",          "opaliteHealth"),
                ("🎤 VoiceCare AI",            "voiceCareAI"),
                ("🧠 Bland AI",                "blandHealthCallML"),
            ],
            "📞 Sales & Marketing AI": [
                ("📞 Hyperbound AI",           "hyperboundAI"),
                ("📈 Conversion AI",           "conversionAI"),
                ("🍕 Loop AI",                "loopAI"),
            ],
            "📄 Document AI & Parsing": [
                ("📄 Unsiloed AI",             "unsiloedAI"),
                ("🔬 LayerLens (Layer Health)", "layerLens"),
            ],
            "🧪 Testing & E-commerce": [
                ("🧪 Decipher AI",             "decipherAI"),
                ("🛍️ Spur",                   "spurAI"),
            ],
            "🔧 Developer Tools & Operations": [
                ("🗣️ Rebolt AI",              "reboltAI"),
                ("🌿 Olive",                  "olive"),
                ("🔗 HotGlue",               "hotGlue"),
                ("🏗️ Semble AI",              "sembleAI"),
                ("💻 OpenBuilder",            "openBuilder"),
                ("🌐 Browser Use",            "browserUse"),
                ("🌐 ThirdLayer",             "thirdLayer"),
                ("💜 Lovable",               "lovable"),
            ],
            "🔐 Legal & Identity": [
                ("📄 Dioptra AI",             "dioptraAI"),
                ("🔐 Spruce ID",              "spruceID"),
            ],
        }

        for category, demos in categories.items():
            if search_query:
                filtered_demos = [(n, p) for n, p in demos if search_query.lower() in n.lower()]
            else:
                filtered_demos = demos
            if filtered_demos:
                with st.expander(f"{category} ({len(filtered_demos)})", expanded=(search_query != "")):
                    for demo_name, demo_page in filtered_demos:
                        if st.button(demo_name, key=f"nav_{demo_page}"):
                            st.switch_page(f"pages/{demo_page}.py")

        st.markdown("""
        <div style="padding: 20px 15px; background: rgba(255,255,255,0.08); border-radius: 10px; margin-top: 20px;">
            <div style="color: rgba(255,255,255,0.85); font-size: 13px; font-weight: 600; margin-bottom: 15px;">CONTACT</div>
            <div style="color: white; font-size: 13px; line-height: 1.8; word-break: break-all;">
                <a href="mailto:nandhakumar.anju@gmail.com" style="color: rgba(255,255,255,0.9); text-decoration: none;">nandhakumar.anju@gmail.com</a>
            </div>
            <div style="margin-top: 18px; font-size: 24px; display: flex; gap: 12px; justify-content: center;">
                <a href="https://linkedin.com/in/anju-vilashini" target="_blank" style="color: white; text-decoration: none;">💼</a>
                <a href="https://github.com/Av1352" target="_blank" style="color: white; text-decoration: none;">💻</a>
                <a href="https://vxanju.com" target="_blank" style="color: white; text-decoration: none;">🌐</a>
            </div>
        </div>
        """, unsafe_allow_html=True)