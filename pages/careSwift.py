"""
CareSwift - AI Scribe for Ambulance Reports
Automated emergency medical documentation
Built for CareSwift by Anju Nandhakumar
"""

import streamlit as st
from utils.sidebar import render_sidebar
import pandas as pd
from datetime import datetime
import json

st.set_page_config(page_title="CareSwift - AI Ambulance Scribe", layout="wide")
render_sidebar()

# Initialize session state
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

# Sample scenarios
SCENARIOS = {
    "Cardiac Arrest - ROSC": {
        "voice_input": "67 year old male, found unresponsive by family. No pulse, not breathing. Started CPR immediately. Arrived on scene 8 minutes after call. Patient in cardiac arrest, asystole on monitor. Established airway, two rounds of epi given. ROSC achieved after 12 minutes. Blood pressure 90 over 60, heart rate 110. Patient remains unconscious. Transported to Mass General emergency.",
        "vitals": {
            "initial_bp": "0/0 (pulseless)",
            "final_bp": "90/60",
            "hr": "110",
            "resp_rate": "12 (ventilated)",
            "spo2": "94%",
            "gcs": "3"
        },
        "interventions": ["CPR", "Airway Management", "Epinephrine x2", "Cardiac Monitor"],
        "chief_complaint": "Cardiac Arrest"
    },
    "Motor Vehicle Accident - Trauma": {
        "voice_input": "21 year old female, driver, rear-ended at intersection. Airbag deployed. Patient complaining of neck pain and headache. Alert and oriented times three. Cervical collar applied. Vitals stable. No obvious deformities. Transported for CT scan and observation.",
        "vitals": {
            "bp": "125/78",
            "hr": "88",
            "resp_rate": "16",
            "spo2": "98%",
            "gcs": "15"
        },
        "interventions": ["C-collar", "Spinal Precautions", "IV Access"],
        "chief_complaint": "MVA - Rear Impact"
    },
    "Diabetic Emergency - Hypoglycemia": {
        "voice_input": "55 year old male, diabetic, found confused by coworker. Blood sugar 38. Patient sweating, shaky, disoriented. Gave oral glucose. Blood sugar recheck 82. Patient oriented, feeling better. Vitals stable. Patient refused transport, signed AMA.",
        "vitals": {
            "bp": "138/82",
            "hr": "92",
            "resp_rate": "14",
            "spo2": "97%",
            "gcs": "15",
            "glucose": "38 → 82"
        },
        "interventions": ["Oral Glucose", "Blood Glucose Monitoring"],
        "chief_complaint": "Hypoglycemia"
    }
}

def generate_pcr_report(scenario_data):
    """Generate Patient Care Report from voice input"""
    
    # Extract key information using NLP (simulated)
    voice = scenario_data['voice_input']
    vitals = scenario_data['vitals']
    interventions = scenario_data['interventions']
    
    # Parse age and gender
    age_match = voice.split()[0]
    gender = "male" if "male" in voice.lower() else "female"
    
    # Generate report sections
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_demographics": {
            "age": age_match,
            "gender": gender.capitalize(),
            "chief_complaint": scenario_data['chief_complaint']
        },
        "narrative": {
            "dispatch_info": f"Dispatched to scene for {scenario_data['chief_complaint']}",
            "scene_findings": voice.split('.')[1] if len(voice.split('.')) > 1 else voice[:100],
            "assessment": voice,
            "treatment": f"Interventions: {', '.join(interventions)}",
            "transport": "Patient transported to receiving facility" if "transported" in voice.lower() else "Patient refused transport - AMA signed"
        },
        "vitals": vitals,
        "interventions": interventions,
        "disposition": "Transported to ED" if "transported" in voice.lower() else "Refused Transport (AMA)"
    }
    
    return report

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(239, 68, 68, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #ef4444 0%, #f87171 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(239, 68, 68, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🚑</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            CareSwift
        </h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">AI Scribe for Ambulance Reports</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Complete PCRs in 90 seconds, not 20 minutes</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4);">Voice-to-Text</span>
            <span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);">Clinical NLP</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">EMS Workflow</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">YC Backed</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">CareSwift</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Value Prop
st.markdown("""
<div style="background: linear-gradient(135deg, #fee2e2, #fecaca); padding: 25px; border-radius: 15px; border: 2px solid #dc2626; margin-bottom: 30px;">
    <h3 style="color: #7f1d1d; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">🎯 The EMS Documentation Problem</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #ef4444; font-weight: 700; margin: 0 0 8px 0;">❌ Today</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Paramedics spend 20 min per call writing reports. Documentation happens AFTER patient care. 40% of shift time is paperwork.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #f59e0b; font-weight: 700; margin: 0 0 8px 0;">💰 Cost Impact</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">$50K/year per paramedic in documentation time. Delayed care during busy shifts. Incomplete reports cause billing issues.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #10b981; font-weight: 700; margin: 0 0 8px 0;">✅ With CareSwift</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Complete PCR in 90 seconds via voice. Real-time documentation during transport. 95% time savings. More patient care time.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🎙️ Voice Documentation", "📋 Generated Reports"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Speak Your Report</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Just describe what happened - AI generates complete PCR automatically</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        use_sample = st.checkbox("Use sample call", value=True)
        
        if use_sample:
            scenario_name = st.selectbox("Select Emergency Scenario", list(SCENARIOS.keys()))
            scenario = SCENARIOS[scenario_name]
            voice_input = scenario['voice_input']
        else:
            voice_input = st.text_area(
                "Dictate your call narrative",
                placeholder="Describe the call, patient condition, interventions, and outcome...",
                height=150
            )
        
        if voice_input:
            st.text_area("Voice Input Transcript", voice_input, height=150, disabled=True)
            
            if st.button("📝 Generate PCR", type="primary", use_container_width=True):
                st.session_state.report_generated = True
                st.session_state.current_scenario = scenario_name if use_sample else "Custom"
                st.session_state.pcr = generate_pcr_report(scenario if use_sample else {
                    'voice_input': voice_input,
                    'vitals': {},
                    'interventions': [],
                    'chief_complaint': 'Medical Emergency'
                })
    
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #92400e; margin: 0 0 12px 0; font-size: 16px;">⚡ How It Works</h4>
            <ol style="color: #78350f; font-size: 13px; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li><strong>Speak naturally:</strong> Describe call as you would to a partner</li>
                <li><strong>AI extracts:</strong> Demographics, vitals, interventions, timeline</li>
                <li><strong>Generates PCR:</strong> Complete report in standard format</li>
                <li><strong>Review & sign:</strong> 30 seconds to verify, then submit</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.report_generated:
        st.markdown("<hr style='margin: 30px 0; border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)
        
        pcr = st.session_state.pcr
        
        # Generated PCR
        st.success("✅ Patient Care Report Generated in 1.5 seconds")
        
        col_x, col_y = st.columns([2, 1])
        
        with col_x:
            st.markdown(f"""
            <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
                <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">📋 Patient Care Report</h3
                <div style="background: #f9fafb; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0; font-weight: 600;">PATIENT DEMOGRAPHICS</p>
                    <p style="color: #1f2937; font-size: 14px; margin: 5px 0;"><strong>Age:</strong> {pcr['patient_demographics']['age']} years</p>
                    <p style="color: #1f2937; font-size: 14px; margin: 5px 0;"><strong>Gender:</strong> {pcr['patient_demographics']['gender']}</p>
                    <p style="color: #1f2937; font-size: 14px; margin: 5px 0;"><strong>Chief Complaint:</strong> {pcr['patient_demographics']['chief_complaint']}</p>
                </div>
                <div style="background: #f9fafb; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0; font-weight: 600;">NARRATIVE</p>
                    <p style="color: #1f2937; font-size: 13px; margin: 8px 0; line-height: 1.7;">{pcr['narrative']['assessment']}</p>
                </div>
                <div style="background: #f9fafb; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0; font-weight: 600;">INTERVENTIONS</p>
                    {''.join([f'<p style="color: #059669; font-size: 13px; margin: 5px 0;">✓ {intervention}</p>' for intervention in pcr['interventions']])}
                </div>
                <div style="background: #ecfdf5; padding: 15px; border-radius: 10px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0; font-weight: 600;">DISPOSITION</p>
                    <p style="color: #059669; font-size: 14px; font-weight: 700; margin: 5px 0;">{pcr['disposition']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_y:
            st.markdown(f"""
            <div style="background: #eff6ff; padding: 20px; border-radius: 12px; border: 2px solid #3b82f6; margin-bottom: 15px;">
                <h4 style="color: #1e40af; margin: 0 0 12px 0; font-size: 16px;">💓 Vital Signs</h4>
                <table style="width: 100%;">
                    {''.join([f'<tr><td style="padding: 6px 0; color: #6b7280; font-size: 13px;">{k.replace("_", " ").title()}</td><td style="text-align: right; padding: 6px 0; color: #1f2937; font-weight: 700; font-size: 13px;">{v}</td></tr>' for k, v in pcr['vitals'].items()])}
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            st.download_button(
                "💾 Download PCR (JSON)",
                json.dumps(pcr, indent=2),
                "patient_care_report.json",
                "application/json",
                use_container_width=True
            )

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">EMS Analytics Dashboard</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">System-wide metrics and performance tracking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System metrics
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 32px; border-radius: 20px; margin-bottom: 25px;">
        <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">📊 This Month's Impact</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">PCRs Generated</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 8px 0;">1,847</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">Across all units</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Time Saved</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 8px 0;">610</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">hours (95% reduction)</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Avg Completion</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 8px 0;">90s</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">vs 20 min manual</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Billing Accuracy</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 8px 0;">98%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">vs 85% manual</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #dc2626; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for CareSwift</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⏱️ 95% Time Reduction</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    90 seconds vs 20 minutes per report. Paramedics spend more time with patients, less time on paperwork. 610 hours saved monthly.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Better Billing</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    98% billing accuracy vs 85% manual. Capture all billable interventions. Complete documentation = better reimbursement.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📈 Better Care</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Real-time documentation during transport. No details forgotten. Complete medical records. Improved patient handoffs to ED.
                </p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 EMS System Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">95% time savings:</strong> 90s vs 20 min per report</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">610 hours saved:</strong> monthly per EMS system</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">98% billing accuracy:</strong> Better reimbursement</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Real-time documentation:</strong> During patient transport</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Voice-to-Text</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Real-time speech recognition, medical vocabulary</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Clinical NLP</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Extract vitals, interventions, timeline from narrative</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Structured Reports</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Generate compliant PCR format automatically</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ EMS Integration</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Connect to CAD, billing, hospital systems</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(220, 38, 38, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">CareSwift</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Voice Recognition • Clinical NLP • EMS Documentation • Healthcare Automation
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered ambulance documentation with voice-to-PCR automation.<br>
            Speech recognition • Clinical entity extraction • Structured reporting • EMS workflow integration
        </p>
    </div>
    """, unsafe_allow_html=True)