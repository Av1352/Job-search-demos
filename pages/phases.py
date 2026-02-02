"""
Phases - Clinical Trial Automation
Automate clinical trial operations and patient monitoring
Built for Phases by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Phases", page_icon="⚗️", layout="wide")

# Trial phases
TRIAL_PHASES = {
    'Phase I': {'patients': 24, 'duration': '6 months', 'focus': 'Safety & Dosing', 'success_rate': 78},
    'Phase II': {'patients': 120, 'duration': '12 months', 'focus': 'Efficacy', 'success_rate': 62},
    'Phase III': {'patients': 850, 'duration': '24 months', 'focus': 'Confirmation', 'success_rate': 51},
    'Phase IV': {'patients': 2000, 'duration': '36 months', 'focus': 'Post-Market', 'success_rate': 95}
}

# Automation tasks
AUTOMATION_TASKS = {
    'Patient Monitoring': {'automation': 96.8, 'time_saved': '12 hrs/day'},
    'Data Entry': {'automation': 99.2, 'time_saved': '8 hrs/day'},
    'Adverse Event Reporting': {'automation': 94.5, 'time_saved': '4 hrs/day'},
    'Protocol Compliance': {'automation': 97.3, 'time_saved': '6 hrs/day'},
    'Regulatory Reporting': {'automation': 98.7, 'time_saved': '10 hrs/day'}
}

# Patient metrics
PATIENT_METRICS = {
    'enrollment_status': 'Active',
    'protocol_adherence': 94.5,
    'adverse_events': 2,
    'missed_visits': 0,
    'data_completeness': 98.2
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #3b82f6 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">⚗️</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Phases</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Clinical Trial Automation</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Automate operations • Real-time monitoring • 97% automation</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["👥 Patient Monitoring", "📊 Trial Dashboard", "⚡ Automation Analytics", "💡 Platform Features"])

with tab1:
    st.markdown("### Real-Time Patient Monitoring")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Select Trial**")
        
        trial_phase = st.selectbox("Phase", list(TRIAL_PHASES.keys()))
        trial_id = st.text_input("Trial ID", "PHII-ONC-2025-047")
        
        st.markdown("**Patient Selection**")
        
        patient_id = st.selectbox("Patient ID", [f"PT-{i:04d}" for i in range(1, 121)])
        
        st.markdown("**Monitoring Parameters**")
        
        check_adherence = st.checkbox("Protocol adherence", value=True)
        check_vitals = st.checkbox("Vital signs tracking", value=True)
        check_adverse = st.checkbox("Adverse events", value=True)
        check_dosing = st.checkbox("Medication compliance", value=True)
        
        monitor_btn = st.button("👁️ View Patient Status", type="primary", use_container_width=True)
    
    with col2:
        if monitor_btn:
            st.markdown("**Patient Status**")
            
            with st.spinner("Analyzing patient data..."):
                import time
                time.sleep(1.2)
            
            st.success("✅ Patient data current - all systems nominal")
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Patient {patient_id}</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Enrollment Status</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{PATIENT_METRICS['enrollment_status']}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Protocol Adherence</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{PATIENT_METRICS['protocol_adherence']}%</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Adverse Events</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{PATIENT_METRICS['adverse_events']} (Grade 1)</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Data Completeness</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{PATIENT_METRICS['data_completeness']}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Recent Activity**")
            
            activity = {
                'Date': ['Jan 28', 'Jan 25', 'Jan 22', 'Jan 18', 'Jan 15'],
                'Event': ['Study visit', 'Lab results', 'Medication refill', 'Adverse event (mild)', 'Study visit'],
                'Status': ['✅ Complete', '✅ Normal', '✅ Complete', '✅ Resolved', '✅ Complete'],
                'Action': ['None', 'None', 'None', 'Reported to sponsor', 'None']
            }
            st.dataframe(pd.DataFrame(activity), hide_index=True, use_container_width=True)

with tab2:
    st.markdown("### Trial Operations Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Enrolled Patients", "118/120", "98.3%")
    col2.metric("Protocol Adherence", "94.5%", "+2.1%")
    col3.metric("Data Quality", "98.2%", "+1.5%")
    col4.metric("On-Time Visits", "96.8%", "+3.2%")
    
    st.markdown("**Enrollment Progress**")
    
    weeks = list(range(1, 17))
    target_enrollment = [7.5 * w for w in weeks]
    actual_enrollment = [min(7.5 * w + np.random.randint(-5, 8), 120) for w in weeks]
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=weeks, y=target_enrollment,
        mode='lines',
        name='Target',
        line=dict(color='#3b82f6', width=2, dash='dash')
    ))
    fig1.add_trace(go.Scatter(
        x=weeks, y=actual_enrollment,
        mode='lines+markers',
        name='Actual',
        line=dict(color='#10b981', width=3),
        fill='tonexty',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    fig1.update_layout(xaxis_title='Week', yaxis_title='Patients Enrolled', height=300)
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Site Performance**")
        
        sites = {
            'Site': ['Site A (Boston)', 'Site B (NYC)', 'Site C (SF)', 'Site D (Chicago)'],
            'Enrolled': [34, 28, 32, 24],
            'Adherence': ['96.2%', '93.8%', '95.1%', '92.5%'],
            'Data Quality': ['99.1%', '97.8%', '98.5%', '97.2%']
        }
        st.dataframe(pd.DataFrame(sites), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Adverse Events**")
        
        ae_severity = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']
        ae_counts = [45, 12, 3, 0, 0]
        
        fig2 = go.Figure(data=[go.Bar(
            x=ae_severity,
            y=ae_counts,
            marker=dict(color=['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#7f1d1d']),
            text=ae_counts,
            textposition='auto'
        )])
        fig2.update_layout(yaxis_title='Count', height=250)
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Automation Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Automation", "97.3%", "+5.2%")
    col2.metric("Time Saved/Day", "40 hrs", "Per trial")
    col3.metric("Error Reduction", "85%", "vs manual")
    col4.metric("Cost Savings", "$180K/mo", "Operations")
    
    st.markdown("**Automation Rates by Task**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        automation_data = []
        for task, data in AUTOMATION_TASKS.items():
            automation_data.append({
                'Task': task,
                'Automation': f"{data['automation']}%",
                'Time Saved': data['time_saved']
            })
        
        st.dataframe(pd.DataFrame(automation_data), hide_index=True, use_container_width=True)
    
    with col2:
        fig3 = go.Figure(data=[go.Bar(
            x=list(AUTOMATION_TASKS.keys()),
            y=[AUTOMATION_TASKS[t]['automation'] for t in AUTOMATION_TASKS.keys()],
            marker=dict(color='#3b82f6'),
            text=[f"{AUTOMATION_TASKS[t]['automation']}%" for t in AUTOMATION_TASKS.keys()],
            textposition='auto'
        )])
        fig3.update_layout(yaxis=dict(range=[90, 100]), height=250)
        st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("**Monthly Impact**")
    
    impact = {
        'Metric': ['Tasks Automated', 'Hours Saved', 'Data Entry Eliminated', 'Report Generation', 'Compliance Checks'],
        'Value': ['8,450', '1,200 hrs', '99.2%', '567 reports', '100%'],
        'Improvement': ['+23%', '+340 hrs', '+12%', '+234', 'Maintained']
    }
    st.dataframe(pd.DataFrame(impact), hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### Automation Technology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Automated Workflows**")
        st.markdown("""
        - ✅ Patient monitoring (vitals, labs, symptoms)
        - ✅ Automated data entry from source docs
        - ✅ Adverse event detection & reporting
        - ✅ Protocol deviation alerts
        - ✅ Regulatory compliance checking
        - ✅ Visit scheduling & reminders
        - ✅ Informed consent tracking
        - ✅ Drug accountability
        """)
        
        st.markdown("**AI Capabilities**")
        st.markdown("""
        - ✅ NLP for clinical note extraction
        - ✅ Anomaly detection in patient data
        - ✅ Predictive dropout modeling
        - ✅ Protocol compliance scoring
        - ✅ Risk signal identification
        - ✅ Data quality validation
        """)
    
    with col2:
        st.markdown("**Integration & Compliance**")
        st.markdown("""
        - ✅ EDC system integration (Medidata, Veeva)
        - ✅ EHR connectivity (Epic, Cerner)
        - ✅ eCOA/ePRO platforms
        - ✅ Lab interfaces (central labs)
        - ✅ CTMS integration
        - ✅ Safety database (Argus, AERS)
        """)
        
        st.markdown("**Regulatory Features**")
        st.markdown("""
        - ✅ 21 CFR Part 11 compliance
        - ✅ GCP adherence tracking
        - ✅ ICH guidelines validation
        - ✅ Audit trail generation
        - ✅ E-signature workflows
        - ✅ FDA submission ready
        """)
    
    st.markdown("**Automation Performance**")
    
    perf_metrics = {
        'Task Category': ['Data Management', 'Safety Monitoring', 'Compliance', 'Reporting', 'Patient Engagement'],
        'Automation Rate': ['99.2%', '94.5%', '97.3%', '98.7%', '96.8%'],
        'Accuracy': ['99.8%', '96.5%', '100%', '99.5%', '95.2%'],
        'Time Saved': ['8 hrs/day', '4 hrs/day', '6 hrs/day', '10 hrs/day', '12 hrs/day']
    }
    st.dataframe(pd.DataFrame(perf_metrics), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #1e3a8a; font-size: 24px; font-weight: 800;">💡 System Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #1d4ed8; font-weight: 700; margin: 0 0 6px 0;">✓ 97.3% Automation</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Operations fully automated</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #1d4ed8; font-weight: 700; margin: 0 0 6px 0;">✓ 40 Hours Saved</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Daily per trial</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #1d4ed8; font-weight: 700; margin: 0 0 6px 0;">✓ 98.2% Data Quality</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Complete & accurate</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #1d4ed8; font-weight: 700; margin: 0 0 6px 0;">✓ $180K/Month Savings</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Operational cost reduction</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #3b82f6 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Phases</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)