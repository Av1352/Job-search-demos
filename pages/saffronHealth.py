"""
Saffron Health - Referral Automation Platform
Automated specialist referral workflows and coordination
Built for Saffron Health by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Saffron Health", page_icon="🟡", layout="wide")

# Referral automation tasks
AUTOMATION_TASKS = {
    'Document Generation': {'automation': 98.7, 'time_saved': '12 min'},
    'Fax/Upload to Specialist': {'automation': 99.2, 'time_saved': '8 min'},
    'Insurance Pre-Auth': {'automation': 94.5, 'time_saved': '45 min'},
    'Patient Notification': {'automation': 99.8, 'time_saved': '5 min'},
    'Appointment Coordination': {'automation': 87.3, 'time_saved': '20 min'}
}

# Specialist network
SPECIALIST_NETWORK = {
    'Cardiology': {'providers': 156, 'avg_response': '2.3 hours', 'acceptance': 94.5},
    'Orthopedics': {'providers': 189, 'avg_response': '1.8 hours', 'acceptance': 96.2},
    'Neurology': {'providers': 134, 'avg_response': '3.5 hours', 'acceptance': 89.7},
    'Gastroenterology': {'providers': 123, 'avg_response': '2.1 hours', 'acceptance': 92.8},
    'Endocrinology': {'providers': 98, 'avg_response': '2.7 hours', 'acceptance': 91.3}
}

# Referral metrics
REFERRAL_METRICS = {
    'Processing Time': 8.5,  # minutes vs 90 min manual
    'Completion Rate': 92.8,
    'Specialist Acceptance': 93.1,
    'Patient Follow-Through': 88.7,
    'Insurance Approval': 89.5
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #eab308 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🟡</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Saffron Health</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Referral Automation Platform</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Automated workflows • 8.5 min processing • 92.8% completion</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🟡 Automated Referral", "📊 Workflow Dashboard", "📈 Performance", "💡 Automation Features"])

with tab1:
    st.markdown("### Automated Referral Processing")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Referral Request**")
        
        patient_name = st.text_input("Patient Name", "Robert Kim")
        dob = st.date_input("Date of Birth", datetime(1965, 8, 15))
        
        st.markdown("**Clinical Details**")
        
        specialty = st.selectbox("Specialty Needed", list(SPECIALIST_NETWORK.keys()))
        diagnosis = st.text_input("Diagnosis", "Chest pain, r/o CAD")
        urgency = st.selectbox("Urgency", ["Routine", "Urgent (within 1 week)", "STAT (24-48 hrs)"])
        
        clinical_summary = st.text_area(
            "Clinical Summary",
            "58yo M with exertional chest pain x3 weeks. Abnormal stress test showing ST depression. HTN, hyperlipidemia. On aspirin, statin, lisinopril.",
            height=80
        )
        
        st.markdown("**Insurance**")
        
        insurance = st.selectbox("Payer", ["Aetna PPO", "UnitedHealthcare", "BCBS", "Medicare"])
        prior_auth_needed = st.checkbox("Prior auth required", value=True)
        
        st.markdown("**AI Automation**")
        
        auto_generate_docs = st.checkbox("Auto-generate referral letter", value=True)
        auto_fax = st.checkbox("Auto-fax to specialist", value=True)
        auto_prior_auth = st.checkbox("Auto-submit prior auth", value=True)
        
        submit_btn = st.button("🟡 Process Referral", type="primary", use_container_width=True)
    
    with col2:
        if submit_btn:
            st.markdown("**Automated Processing**")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            stages = [
                ("Generating referral letter...", 0.2),
                ("Submitting prior authorization...", 0.4),
                ("Matching specialist network...", 0.6),
                ("Faxing documents...", 0.8),
                ("Notifying patient...", 1.0)
            ]
            
            for stage, progress in stages:
                status_text.text(stage)
                progress_bar.progress(progress)
                time.sleep(0.5)
            
            st.success("✅ Referral processed automatically in 8.5 minutes!")
            
            network_data = SPECIALIST_NETWORK[specialty]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #eab308 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Referral Status</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Specialty</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{specialty}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Processing Time</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">8.5 min</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Specialist Match</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Dr. Sarah Chen</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Expected Response</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{network_data['avg_response']}</p>
                    </div>
                </div>
                <div style="margin-top: 15px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Actions Completed</p>
                    <p style="font-size: 15px; color: white; margin: 0;">
                        ✅ Referral letter generated<br>
                        ✅ Prior auth submitted to {insurance}<br>
                        ✅ Clinical notes faxed to specialist<br>
                        ✅ Patient notified via SMS/email<br>
                        ✅ Appointment scheduled for Feb 12th @ 2:30 PM
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Time Saved", "81.5 min", "vs manual")
            col2.metric("Automation", "96.8%", "End-to-end")
            col3.metric("Prior Auth", "Submitted", "✓")
            col4.metric("Acceptance Prob", "94.5%", "High")

with tab2:
    st.markdown("### Referral Operations Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Referrals Today", "247", "+34")
    col2.metric("Automation Rate", "96.8%", "+8.3%")
    col3.metric("Completion", "92.8%", "+24.3%")
    col4.metric("Time Saved", "420 hrs", "Monthly")
    
    st.markdown("**Automation by Task**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        auto_data = []
        for task, data in AUTOMATION_TASKS.items():
            auto_data.append({
                'Task': task,
                'Automation': f"{data['automation']}%",
                'Time Saved': data['time_saved']
            })
        
        st.dataframe(pd.DataFrame(auto_data), hide_index=True, use_container_width=True)
    
    with col2:
        fig1 = go.Figure(data=[go.Bar(
            x=list(AUTOMATION_TASKS.keys()),
            y=[AUTOMATION_TASKS[t]['automation'] for t in AUTOMATION_TASKS.keys()],
            marker=dict(color='#eab308'),
            text=[f"{AUTOMATION_TASKS[t]['automation']}%" for t in AUTOMATION_TASKS.keys()],
            textposition='auto'
        )])
        fig1.update_layout(yaxis=dict(range=[85, 100]), height=300)
        st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("**Daily Referral Volume**")
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    referrals = [234, 256, 247, 251, 263, 89, 67]
    
    fig2 = go.Figure(data=[go.Scatter(
        x=days, y=referrals,
        mode='lines+markers',
        line=dict(color='#eab308', width=3),
        fill='tozeroy',
        fillcolor='rgba(234, 179, 8, 0.1)'
    )])
    fig2.update_layout(yaxis_title='Referrals', height=250)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Automation Performance")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**System Metrics**")
        
        metrics_data = []
        for metric, value in REFERRAL_METRICS.items():
            if 'Time' in metric:
                display = f"{value} min"
            else:
                display = f"{value}%"
            
            metrics_data.append({
                'Metric': metric,
                'Value': display,
                'Status': '✅ Excellent'
            })
        
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
        
        st.markdown("**vs Manual Process**")
        
        comparison = {
            'Step': ['Document prep', 'Prior auth', 'Faxing', 'Follow-up', 'Total'],
            'Manual': ['20 min', '45 min', '12 min', '13 min', '90 min'],
            'Saffron AI': ['1.2 min', '3.5 min', '0.8 min', '3.0 min', '8.5 min'],
            'Time Saved': ['18.8 min', '41.5 min', '11.2 min', '10 min', '81.5 min']
        }
        st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Completion Rate by Specialty**")
        
        specialties = list(SPECIALIST_NETWORK.keys())
        completion = [SPECIALIST_NETWORK[s]['acceptance'] for s in specialties]
        
        fig3 = go.Figure(data=[go.Bar(
            x=specialties,
            y=completion,
            marker=dict(
                color=completion,
                colorscale='YlOrBr',
                cmin=85,
                cmax=100
            ),
            text=[f"{c}%" for c in completion],
            textposition='auto'
        )])
        fig3.update_layout(yaxis=dict(range=[85, 100]), yaxis_title='Acceptance Rate (%)', height=250)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("**Monthly Impact**")
        
        impact = {
            'Metric': ['Referrals Processed', 'Hours Saved', 'Revenue Protected', 'Leakage Prevented'],
            'Value': ['5,847', '420 hrs', '$680K', '$125K']
        }
        st.dataframe(pd.DataFrame(impact), hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### Automation Technology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Document Automation**")
        st.markdown("""
        - ✅ Auto-generate referral letters
        - ✅ Clinical summary extraction (NLP)
        - ✅ ICD-10/CPT code population
        - ✅ Insurance info auto-fill
        - ✅ PDF generation & formatting
        - ✅ E-signature integration
        - ✅ HIPAA-compliant templates
        - ✅ Payer-specific requirements
        """)
        
        st.markdown("**Communication Automation**")
        st.markdown("""
        - ✅ Automated fax to specialists
        - ✅ Secure portal uploads
        - ✅ Direct EHR-to-EHR messaging
        - ✅ SMS/email to patients
        - ✅ Appointment confirmation
        - ✅ Follow-up reminders
        """)
    
    with col2:
        st.markdown("**Prior Auth Automation**")
        st.markdown("""
        - ✅ Auto-detect PA requirements
        - ✅ Extract clinical justification
        - ✅ Submit to payer portals (50+)
        - ✅ Status tracking
        - ✅ Appeal automation if denied
        - ✅ 94.5% automation rate
        """)
        
        st.markdown("**Integration**")
        st.markdown("""
        - ✅ EHR integration (Epic, Cerner, Athena)
        - ✅ Specialist scheduling systems
        - ✅ Insurance eligibility APIs
        - ✅ E-fax platforms
        - ✅ Practice management systems
        - ✅ Patient engagement tools
        """)
    
    st.markdown("**Specialist Network Performance**")
    
    network_data = []
    for specialty, data in SPECIALIST_NETWORK.items():
        network_data.append({
            'Specialty': specialty,
            'Providers': data['providers'],
            'Avg Response': data['avg_response'],
            'Acceptance': f"{data['acceptance']}%"
        })
    
    st.dataframe(pd.DataFrame(network_data), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #78350f; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #d97706; font-weight: 700; margin: 0 0 6px 0;">✓ 96.8% Automation</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">End-to-end workflow</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #d97706; font-weight: 700; margin: 0 0 6px 0;">✓ 8.5 Min Processing</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">vs 90 min manual</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #d97706; font-weight: 700; margin: 0 0 6px 0;">✓ 92.8% Completion</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">vs 68.5% manual</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #d97706; font-weight: 700; margin: 0 0 6px 0;">✓ 420 Hrs Saved</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Monthly per practice</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #eab308 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Saffron Health</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashvi Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)