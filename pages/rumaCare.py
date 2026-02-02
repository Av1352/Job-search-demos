"""
Ruma Care - Prior Authorization Automation
AI-powered prior auth processing for healthcare providers
Built for Ruma Care by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Ruma Care", page_icon="📋", layout="wide")

# Prior auth types
PRIOR_AUTH_TYPES = {
    'MRI Scan': {'approval_rate': 89.5, 'avg_time': '2.3 hrs', 'manual_time': '4 days'},
    'Specialist Referral': {'approval_rate': 92.3, 'avg_time': '1.8 hrs', 'manual_time': '3 days'},
    'Prescription Drug': {'approval_rate': 87.2, 'avg_time': '1.2 hrs', 'manual_time': '2 days'},
    'Surgery': {'approval_rate': 85.7, 'avg_time': '3.5 hrs', 'manual_time': '5 days'},
    'Physical Therapy': {'approval_rate': 94.8, 'avg_time': '0.9 hrs', 'manual_time': '2 days'}
}

# Insurance payers
PAYERS = ['UnitedHealthcare', 'Aetna', 'Blue Cross Blue Shield', 'Cigna', 'Humana']

# Automation stats
AUTOMATION_STATS = {
    'Auto-Submission Rate': 96.8,
    'Approval Rate': 89.9,
    'Processing Time': 2.1,  # hours
    'First-Pass Success': 92.5,
    'Appeal Success': 78.3
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #ec4899 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">📋</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Ruma Care</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Prior Authorization Automation</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">AI-powered prior auth • 96.8% automation • 2.1 hours vs 3.4 days</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Submit Prior Auth", "📊 Dashboard", "⚡ Performance", "💡 Features"])

with tab1:
    st.markdown("### AI-Powered Prior Authorization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Prior Auth Request**")
        
        auth_type = st.selectbox("Authorization Type", list(PRIOR_AUTH_TYPES.keys()))
        
        st.markdown("**Patient Information**")
        patient_name = st.text_input("Patient Name", "Sarah Martinez")
        member_id = st.text_input("Member ID", "UHC987654321")
        payer = st.selectbox("Insurance Payer", PAYERS)
        
        st.markdown("**Clinical Information**")
        diagnosis = st.text_input("Diagnosis Code", "M54.5 (Low back pain)")
        procedure = st.text_input("Procedure/Service", "Lumbar MRI")
        clinical_notes = st.text_area("Clinical Justification", "Patient has persistent low back pain for 8 weeks, unresponsive to conservative treatment. Neurological symptoms present.", height=80)
        
        st.markdown("**AI Processing**")
        auto_extract = st.checkbox("Auto-extract from EHR", value=True)
        auto_submit = st.checkbox("Auto-submit to payer", value=True)
        
        submit_btn = st.button("📋 Process Prior Auth", type="primary", use_container_width=True)
    
    with col2:
        if submit_btn:
            st.markdown("**Processing Status**")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            stages = [
                ("Extracting clinical data...", 0.25),
                ("Validating medical necessity...", 0.5),
                ("Generating justification letter...", 0.75),
                ("Submitting to payer portal...", 1.0)
            ]
            
            for stage, progress in stages:
                status_text.text(stage)
                progress_bar.progress(progress)
                time.sleep(0.6)
            
            st.success("✅ Prior authorization submitted successfully!")
            
            auth_data = PRIOR_AUTH_TYPES[auth_type]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ec4899 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Submission Details</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Authorization Type</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{auth_type}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Payer</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{payer}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Processing Time</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{auth_data['avg_time']}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Expected Approval</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{auth_data['approval_rate']}%</p>
                    </div>
                </div>
                <div style="margin-top: 15px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Status</p>
                    <p style="font-size: 16px; color: white; margin: 0;">🟢 Submitted to {payer} portal - Tracking #PA-2026-{np.random.randint(10000, 99999)}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Time Saved", "3.2 days", f"vs {auth_data['manual_time']}")
            col2.metric("Auto-Extracted", "18/18 fields", "100%")
            col3.metric("Confidence", "94.5%", "High")
            col4.metric("Next Action", "Monitor", "24-48 hrs")

with tab2:
    st.markdown("### Prior Auth Operations Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Submitted Today", "247", "+34")
    col2.metric("Approved", "222", "89.9%")
    col3.metric("Pending", "18", "7.3%")
    col4.metric("Denied", "7", "2.8%")
    
    st.markdown("**Approval Rates by Type**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        auth_data = []
        for auth_type, data in PRIOR_AUTH_TYPES.items():
            auth_data.append({
                'Type': auth_type,
                'Approval Rate': f"{data['approval_rate']}%",
                'Avg Time': data['avg_time'],
                'vs Manual': data['manual_time']
            })
        
        st.dataframe(pd.DataFrame(auth_data), hide_index=True, use_container_width=True)
    
    with col2:
        fig1 = go.Figure(data=[go.Bar(
            x=list(PRIOR_AUTH_TYPES.keys()),
            y=[PRIOR_AUTH_TYPES[t]['approval_rate'] for t in PRIOR_AUTH_TYPES.keys()],
            marker=dict(
                color=[PRIOR_AUTH_TYPES[t]['approval_rate'] for t in PRIOR_AUTH_TYPES.keys()],
                colorscale='RdYlGn',
                cmin=80,
                cmax=100
            ),
            text=[f"{PRIOR_AUTH_TYPES[t]['approval_rate']}%" for t in PRIOR_AUTH_TYPES.keys()],
            textposition='auto'
        )])
        fig1.update_layout(yaxis=dict(range=[80, 100]), height=250)
        st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("**Submission Volume Trends**")
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    submissions = [234, 256, 247, 251, 263, 89, 67]
    approvals = [210, 230, 222, 225, 236, 80, 60]
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name='Submitted', x=days, y=submissions, marker_color='#ec4899'))
    fig2.add_trace(go.Bar(name='Approved', x=days, y=approvals, marker_color='#10b981'))
    fig2.update_layout(barmode='group', yaxis_title='Count', height=300)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Automation Performance Metrics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Automation Statistics**")
        
        auto_data = []
        for metric, value in AUTOMATION_STATS.items():
            if 'Rate' in metric:
                display = f"{value}%"
            elif 'Time' in metric:
                display = f"{value} hrs"
            else:
                display = f"{value}%"
            
            auto_data.append({
                'Metric': metric,
                'Value': display,
                'Status': '✅ Excellent'
            })
        
        st.dataframe(pd.DataFrame(auto_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Time Comparison**")
        
        comparison = {
            'Method': ['Ruma AI', 'Manual Processing'],
            'Avg Time': ['2.1 hours', '3.4 days'],
            'Cost/Auth': ['$2.50', '$45.00'],
            'Approval Rate': ['89.9%', '82.3%']
        }
        st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Performance Scores**")
        
        fig3 = go.Figure(data=[go.Bar(
            x=list(AUTOMATION_STATS.keys()),
            y=list(AUTOMATION_STATS.values()),
            marker=dict(color='#ec4899'),
            text=[f"{v}%" if v > 10 else f"{v} hrs" for v in AUTOMATION_STATS.values()],
            textposition='auto'
        )])
        fig3.update_layout(yaxis_title='Score/Time', height=250)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("**Monthly Impact**")
        
        monthly_impact = {
            'Metric': ['Prior Auths', 'Hours Saved', 'Cost Savings', 'Patients Helped'],
            'Value': ['5,847', '1,980 hrs', '$248K', '4,920']
        }
        st.dataframe(pd.DataFrame(monthly_impact), hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### Platform Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**AI Automation**")
        st.markdown("""
        - ✅ Auto-extract from EHR (Epic, Cerner)
        - ✅ Clinical NLP for medical necessity
        - ✅ Payer portal auto-submission
        - ✅ Document generation (peer-to-peer letters)
        - ✅ Appeal letter creation
        - ✅ Status tracking & notifications
        """)
        
        st.markdown("**Supported Services**")
        st.markdown("""
        - ✅ Imaging (MRI, CT, PET scans)
        - ✅ Specialist referrals
        - ✅ Prescription drugs (specialty meds)
        - ✅ Surgeries & procedures
        - ✅ Physical therapy
        - ✅ Durable medical equipment
        - ✅ Home health services
        - ✅ Genetic testing
        """)
    
    with col2:
        st.markdown("**Payer Integration**")
        st.markdown("""
        - ✅ 50+ payer portals integrated
        - ✅ Real-time status tracking
        - ✅ Auto-appeals for denials
        - ✅ Fax/portal hybrid submission
        - ✅ Electronic prior auth (ePA)
        - ✅ CoverMyMeds integration
        """)
        
        st.markdown("**Clinical Intelligence**")
        st.markdown("""
        - ✅ Medical necessity assessment
        - ✅ Evidence-based guidelines
        - ✅ Payer-specific requirements
        - ✅ Denial prediction (78% accuracy)
        - ✅ Appeal strategy optimization
        - ✅ Historical approval patterns
        """)
    
    st.markdown("**Automation Performance by Payer**")
    
    payer_perf = {
        'Payer': PAYERS,
        'Submissions': [1234, 987, 1456, 823, 567],
        'Approval Rate': ['91.2%', '88.5%', '90.3%', '87.8%', '89.1%'],
        'Avg Time': ['1.9 hrs', '2.3 hrs', '2.0 hrs', '2.5 hrs', '2.1 hrs']
    }
    st.dataframe(pd.DataFrame(payer_perf), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #831843; font-size: 24px; font-weight: 800;">💡 System Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #be185d; font-weight: 700; margin: 0 0 6px 0;">✓ 96.8% Automation</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">End-to-end AI processing</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #be185d; font-weight: 700; margin: 0 0 6px 0;">✓ 2.1 Hour Processing</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">vs 3.4 days manual</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #be185d; font-weight: 700; margin: 0 0 6px 0;">✓ 89.9% Approval Rate</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">vs 82.3% manual</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #be185d; font-weight: 700; margin: 0 0 6px 0;">✓ $248K Saved</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Monthly cost reduction</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #ec4899 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Ruma Care</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)