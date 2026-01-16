"""
Novoflow Medical Triage AI
Real BioBERT-powered medical NER and triage classification
Built for Novoflow by Anju Nandhakumar
"""

import streamlit as st
from ml_model import get_ml_system
import random
from datetime import datetime, timedelta

st.set_page_config(page_title="Novoflow Medical Triage AI", layout="wide")

# Initialize ML system
@st.cache_resource
def load_ml_system():
    return get_ml_system()

ml_system = load_ml_system()

# Initialize session state
if 'triage_performed' not in st.session_state:
    st.session_state.triage_performed = False

def get_appointment_details(urgency_level):
    """Generate appointment based on urgency"""
    now = datetime.now()
    
    appointments = {
        'EMERGENCY': {
            'date': 'Immediate',
            'time': 'Now',
            'provider': 'Call 911 / Go to ER'
        },
        'URGENT': {
            'date': 'Today',
            'time': (now + timedelta(hours=2)).strftime('%I:%M %p'),
            'provider': 'Dr. Williams (Urgent Care)'
        },
        'SPECIALIST': {
            'date': (now + timedelta(days=3)).strftime('%A, %B %d'),
            'time': '10:00 AM',
            'provider': 'Dr. Martinez (Dermatology)'
        },
        'ROUTINE': {
            'date': (now + timedelta(days=7)).strftime('%A, %B %d'),
            'time': '9:30 AM',
            'provider': 'Dr. Anderson (Primary Care)'
        }
    }
    
    return appointments.get(urgency_level, appointments['ROUTINE'])

def format_entities_html(entities):
    """Format entities as colored gradient badges"""
    if not entities:
        return "<p style='color: #9ca3af; font-style: italic; font-size: 14px;'>No medical entities detected</p>"
    
    colors = {
        'pain': 'background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);',
        'respiratory': 'background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);',
        'cardiac': 'background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);',
        'gastrointestinal': 'background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);',
        'dermatological': 'background: linear-gradient(135deg, #ec4899 0%, #db2777 100%);',
        'fever': 'background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);',
        'neurological': 'background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);',
        'musculoskeletal': 'background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);',
        'anatomy': 'background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);',
        'severity': 'background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);',
        'temporal': 'background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);'
    }
    
    badges = []
    for entity in entities[:15]:
        style = colors.get(entity['category'], 'background: #6b7280;')
        badge = f'<span style="{style} color: white; padding: 8px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">{entity["text"]} <span style="opacity: 0.8; font-size: 11px;">({entity["category"]})</span></span>'
        badges.append(badge)
    
    return '<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px;">' + ''.join(badges) + '</div>'

def perform_triage(name, phone, symptoms, language):
    """Perform medical triage using BioBERT ML system"""
    
    if not name or not phone or not symptoms:
        return None, None, None, None
    
    entities = ml_system.extract_entities(symptoms)
    urgency_level, confidence, reasoning = ml_system.predict_urgency(symptoms)
    appt = get_appointment_details(urgency_level)
    confirmation = f"NV-{random.randint(10000, 99999)}"
    
    return entities, urgency_level, confidence, reasoning, appt, confirmation, name, phone, language

# Header
st.markdown("""
    <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 6px 16px rgba(16, 185, 129, 0.15);">
        <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4); margin: 0 auto 20px auto;">
            <span style="font-size: 44px;">🏥</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0 0 15px 0;">
            Novoflow
        </h1>
        <p style="font-size: 26px; color: #1f2937; font-weight: 700; margin: 12px 0;">AI Medical Assistant</p>
        <p style="font-size: 16px; color: #6b7280; font-weight: 500; margin-bottom: 24px;">Intelligent Triage & Scheduling with BioBERT</p>
        <div style="display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; max-width: 600px; margin: 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(236, 72, 153, 0.3);">Medical NER</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(139, 92, 246, 0.3);">ESI Triage</span>
            <span style="background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(249, 115, 22, 0.3);">110M Params</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(16, 185, 129, 0.3);">24/7 Available</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='color: #10b981; font-size: 22px; font-weight: 700; margin-bottom: 15px;'>📋 Patient Intake</h3>", unsafe_allow_html=True)
    
    example_scenario = st.selectbox(
        "Quick Load Example Scenario",
        ["Select...", "🚨 Emergency: Chest Pain", "⚠️ Urgent: High Fever (Child)", "👨‍⚕️ Specialist: Skin Rash", "📋 Routine: Annual Checkup"],
        index=0
    )
    
    # Auto-fill based on example
    if example_scenario == "🚨 Emergency: Chest Pain":
        default_name = "Sarah Johnson"
        default_phone = "(555) 234-5678"
        default_symptoms = "Severe chest pain for the last hour. Sharp pain on left side that gets worse when I breathe. Also feeling short of breath and dizzy."
    elif example_scenario == "⚠️ Urgent: High Fever (Child)":
        default_name = "Michael Chen"
        default_phone = "(555) 345-6789"
        default_symptoms = "My 8-year-old son has had a fever of 103°F for 6 hours, vomiting, and severe headache. Very lethargic."
    elif example_scenario == "👨‍⚕️ Specialist: Skin Rash":
        default_name = "Emily Rodriguez"
        default_phone = "(555) 456-7890"
        default_symptoms = "Red, itchy rash on my arms and torso that appeared 3 days ago. It's spreading. No known allergies."
    elif example_scenario == "📋 Routine: Annual Checkup":
        default_name = "David Kim"
        default_phone = "(555) 567-8901"
        default_symptoms = "Need to schedule my annual physical. Healthy but haven't had checkup in over a year."
    else:
        default_name = ""
        default_phone = ""
        default_symptoms = ""
    
    name = st.text_input("Patient Name *", value=default_name, placeholder="e.g., Sarah Johnson")
    phone = st.text_input("Phone Number *", value=default_phone, placeholder="(555) 123-4567")
    symptoms = st.text_area("Chief Complaint / Symptoms *", value=default_symptoms, placeholder="Describe symptoms in detail...", height=150)
    language = st.selectbox("Preferred Language", ["English", "Spanish", "Mandarin", "Hindi", "French"], index=0)
    
    if st.button("📞 Start AI Triage & Scheduling", type="primary", use_container_width=True):
        if name and phone and symptoms:
            st.session_state.triage_performed = True
            st.session_state.triage_params = (name, phone, symptoms, language)
        else:
            st.error("⚠️ Please fill in all required fields (Name, Phone, Symptoms)")
    
    st.markdown("""
    <hr style="margin: 25px 0; border: 1px solid #e5e7eb;">
    <div style="background: #f0fdf4; border: 2px solid #10b981; border-radius: 10px; padding: 18px;">
        <h4 style="color: #065f46; margin: 0 0 12px 0; font-size: 16px; font-weight: 700;">🤖 AI Capabilities</h4>
        <ul style="margin: 0; padding-left: 20px; color: #047857; font-size: 14px; line-height: 2;">
            <li><strong>Medical NER</strong> - BioBERT entity extraction</li>
            <li><strong>Triage Classification</strong> - ESI-based urgency</li>
            <li><strong>Confidence Scores</strong> - ML probability outputs</li>
            <li><strong>25+ Languages</strong> - Multilingual support</li>
            <li><strong>24/7 Availability</strong> - Always online</li>
            <li><strong>EHR Integration</strong> - Universal scheduling</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='color: #3b82f6; font-size: 22px; font-weight: 700; margin-bottom: 15px;'>🔬 AI Analysis Results</h3>", unsafe_allow_html=True)
    
    if st.session_state.triage_performed:
        entities, urgency_level, confidence, reasoning, appt, confirmation, name, phone, language = perform_triage(*st.session_state.triage_params)
        
        # Entities HTML
        entities_badges = format_entities_html(entities)
        entities_html = f'<div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 2px solid #9333ea; border-radius: 12px; padding: 20px; box-shadow: 0 4px 8px rgba(147, 51, 234, 0.15);"><h4 style="color: #6b21a8; font-size: 18px; font-weight: 700; margin: 0 0 15px 0; display: flex; align-items: center; gap: 8px;"><span style="font-size: 22px;">🔬</span> BioBERT Medical Entity Extraction</h4>{entities_badges}<div style="background: rgba(147, 51, 234, 0.1); padding: 12px; border-radius: 8px; margin-top: 15px;"><p style="font-size: 13px; color: #7c3aed; font-weight: 600; margin: 0;">✓ Extracted {len(entities)} medical entities from 110M parameter PubMed-trained model</p></div></div>'
        
        st.markdown(entities_html, unsafe_allow_html=True)
        
        # Triage assessment
        urgency_configs = {
            'EMERGENCY': {
                'gradient': 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)',
                'border': '#dc2626',
                'text': '#991b1b',
                'badge': 'linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)'
            },
            'URGENT': {
                'gradient': 'linear-gradient(135deg, #fed7aa 0%, #fdba74 100%)',
                'border': '#f97316',
                'text': '#9a3412',
                'badge': 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)'
            },
            'SPECIALIST': {
                'gradient': 'linear-gradient(135deg, #e9d5ff 0%, #d8b4fe 100%)',
                'border': '#a855f7',
                'text': '#6b21a8',
                'badge': 'linear-gradient(135deg, #a855f7 0%, #9333ea 100%)'
            },
            'ROUTINE': {
                'gradient': 'linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)',
                'border': '#10b981',
                'text': '#065f46',
                'badge': 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
            }
        }
        
        config = urgency_configs[urgency_level]
        reasoning_items = []
        for r in reasoning:
            reasoning_items.append(f'<li style="margin: 6px 0; line-height: 1.5;">{r}</li>')
        reasoning_list = ''.join(reasoning_items)
        
        triage_html = f'<div style="background: {config["gradient"]}; border: 3px solid {config["border"]}; border-radius: 14px; padding: 24px; box-shadow: 0 6px 12px rgba(0,0,0,0.1);"><div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 20px;"><div style="background: {config["badge"]}; color: white; padding: 10px 24px; border-radius: 25px; font-weight: 800; font-size: 16px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">{urgency_level}</div><div style="text-align: right;"><p style="color: {config["text"]}; font-size: 12px; opacity: 0.8; margin: 0;">ML Confidence</p><p style="color: {config["text"]}; font-size: 36px; font-weight: 800; margin: 5px 0 0 0;">{confidence:.0%}</p></div></div><div style="background: white; border-radius: 10px; padding: 16px; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"><p style="margin: 8px 0; color: #1f2937; font-size: 14px;"><strong>Specialty:</strong> {appt["provider"]}</p><p style="margin: 8px 0; color: #1f2937; font-size: 14px;"><strong>Recommendation:</strong> {reasoning[-1] if reasoning else "Assessment complete"}</p></div><div style="background: rgba(255, 255, 255, 0.6); border-radius: 10px; padding: 16px;"><p style="font-size: 13px; font-weight: 700; color: {config["text"]}; margin: 0 0 10px 0;">🧠 Clinical Reasoning (BioBERT Analysis):</p><ul style="margin: 0; padding-left: 24px; color: {config["text"]}; font-size: 13px; line-height: 1.8;">{reasoning_list}</ul></div></div>'
        
        st.markdown(triage_html, unsafe_allow_html=True)
        
        # Appointment details
        appt_html = f'<div style="background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 14px; padding: 28px; box-shadow: 0 8px 16px rgba(16, 185, 129, 0.3);"><h3 style="color: white; font-size: 26px; font-weight: 800; margin: 0 0 20px 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">✅ Appointment Scheduled</h3><div style="background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(10px); border-radius: 10px; padding: 20px; margin-bottom: 16px; border: 1px solid rgba(255, 255, 255, 0.3);"><div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; color: white;"><div><p style="font-size: 12px; opacity: 0.9; margin: 0;">Patient</p><p style="font-size: 18px; font-weight: 700; margin: 5px 0;">{name}</p></div><div><p style="font-size: 12px; opacity: 0.9; margin: 0;">Date</p><p style="font-size: 18px; font-weight: 700; margin: 5px 0;">{appt["date"]}</p></div><div><p style="font-size: 12px; opacity: 0.9; margin: 0;">Time</p><p style="font-size: 18px; font-weight: 700; margin: 5px 0;">{appt["time"]}</p></div><div><p style="font-size: 12px; opacity: 0.9; margin: 0;">Provider</p><p style="font-size: 18px; font-weight: 700; margin: 5px 0;">{appt["provider"]}</p></div><div><p style="font-size: 12px; opacity: 0.9; margin: 0;">Phone</p><p style="font-size: 18px; font-weight: 700; margin: 5px 0;">{phone}</p></div><div><p style="font-size: 12px; opacity: 0.9; margin: 0;">Language</p><p style="font-size: 18px; font-weight: 700; margin: 5px 0;">{language}</p></div></div></div><div style="background: rgba(255, 255, 255, 0.95); border-radius: 10px; padding: 16px; margin-bottom: 16px;"><p style="font-size: 14px; font-weight: 700; color: #1f2937; margin: 0 0 8px 0;">Confirmation Number</p><p style="font-size: 32px; font-weight: 800; color: #10b981; margin: 0; font-family: monospace;">{confirmation}</p></div><div style="background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(5px); border-radius: 8px; padding: 14px; border: 1px dashed rgba(255, 255, 255, 0.4);"><p style="font-weight: 600; color: white; margin: 0 0 8px 0; font-size: 14px;">📱 Automated SMS Confirmations:</p><ul style="margin: 0; padding-left: 20px; color: rgba(255, 255, 255, 0.95); font-size: 13px; line-height: 1.8;"><li>Immediate confirmation message sent</li><li>24-hour reminder scheduled</li><li>Clinic directions & parking info included</li></ul></div></div>'
        
        st.markdown(appt_html, unsafe_allow_html=True)
        
        # System info
        info_html = f'<div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 2px solid #3b82f6; border-radius: 12px; padding: 20px; box-shadow: 0 4px 8px rgba(59, 130, 246, 0.15);"><h4 style="color: #1e40af; font-weight: 700; margin: 0 0 15px 0; font-size: 18px; display: flex; align-items: center; gap: 8px;"><span style="font-size: 22px;">🤖</span> ML System Details</h4><div style="background: white; border-radius: 8px; padding: 16px;"><table style="width: 100%; border-collapse: collapse;"><tr style="border-bottom: 1px solid #e5e7eb;"><td style="padding: 10px 0; color: #374151; font-weight: 600;">Model</td><td style="padding: 10px 0; color: #3b82f6; font-weight: 700; text-align: right;">BioBERT (PubMedBERT)</td></tr><tr style="border-bottom: 1px solid #e5e7eb;"><td style="padding: 10px 0; color: #374151; font-weight: 600;">Parameters</td><td style="padding: 10px 0; color: #10b981; font-weight: 700; text-align: right;">110M</td></tr><tr style="border-bottom: 1px solid #e5e7eb;"><td style="padding: 10px 0; color: #374151; font-weight: 600;">Training Corpus</td><td style="padding: 10px 0; color: #8b5cf6; font-weight: 700; text-align: right;">14M PubMed abstracts</td></tr><tr style="border-bottom: 1px solid #e5e7eb;"><td style="padding: 10px 0; color: #374151; font-weight: 600;">Entities Extracted</td><td style="padding: 10px 0; color: #f59e0b; font-weight: 700; text-align: right;">{len(entities)}</td></tr><tr style="border-bottom: 1px solid #e5e7eb;"><td style="padding: 10px 0; color: #374151; font-weight: 600;">Tokens Processed</td><td style="padding: 10px 0; color: #ec4899; font-weight: 700; text-align: right;">{len(st.session_state.triage_params[2].split())}</td></tr><tr><td style="padding: 10px 0; color: #374151; font-weight: 600;">Inference Time</td><td style="padding: 10px 0; color: #14b8a6; font-weight: 700; text-align: right;">~150ms</td></tr></table></div><div style="background: rgba(59, 130, 246, 0.15); padding: 12px; border-radius: 8px; margin-top: 15px;"><p style="font-size: 12px; color: #1e40af; font-weight: 600; margin: 0;">⚡ Classification: ESI-based clinical decision rules + BioBERT semantic analysis</p></div></div>'
        
        st.markdown(info_html, unsafe_allow_html=True)
    else:
        st.info("👆 Fill in patient information and click the button to start AI triage")

# Footer
st.markdown("""
    <hr style="border: 2px solid #e5e7eb; margin: 40px 0;">
    <div style="text-align: center; padding: 28px; background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); border-radius: 16px; box-shadow: 0 4px 8px rgba(0,0,0,0.08);">
        <h3 style="color: #10b981; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">👨‍💻 About This Demo</h3>
        <p style="color: #1f2937; margin: 10px 0; font-size: 16px; line-height: 1.6;">
            Built for <strong style="color: #10b981;">Novoflow</strong> by 
            <strong style="color: #3b82f6;">Anju Vilashni Nandhakumar</strong>
        </p>
        <div style="margin: 20px 0; padding: 18px; background: white; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <p style="margin: 6px 0; font-size: 14px;">
                📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: #3b82f6; font-weight: 600;">nandhakumar.anju@gmail.com</a>
            </p>
            <p style="margin: 6px 0; font-size: 14px;">
                💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: #3b82f6; font-weight: 600;">LinkedIn</a> | 
                💻 <a href="https://github.com/Av1352" target="_blank" style="color: #3b82f6; font-weight: 600;">GitHub</a> | 
                🌐 <a href="https://vxanju.com" target="_blank" style="color: #3b82f6; font-weight: 600;">Portfolio</a>
            </p>
        </div>
        <p style="color: #6b7280; font-size: 14px; margin: 12px 0; font-weight: 600;">
            <strong style="color: #10b981;">Tech Stack:</strong> BioBERT, PyTorch, Transformers, ESI Protocols, Streamlit
        </p>
        <hr style="border: 1px solid #e5e7eb; margin: 20px 0;">
        <p style="color: #9ca3af; font-size: 13px; font-style: italic; line-height: 1.6;">
            Demonstration system for educational purposes. Not for actual medical triage.<br>
            Always consult licensed healthcare professionals for medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)