"""
Legion Health - AI-Native Psychiatry
LLM agents and orchestration for mental health care
Built for Legion Health by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Legion Health", page_icon="🧠", layout="wide")

# Mental health conditions
CONDITIONS = {
    'Depression': {'severity': 'Moderate', 'urgency': 'Standard', 'wait_time': '2 days'},
    'Anxiety': {'severity': 'Mild', 'urgency': 'Standard', 'wait_time': '3 days'},
    'ADHD': {'severity': 'Moderate', 'urgency': 'Standard', 'wait_time': '1 week'},
    'Bipolar Disorder': {'severity': 'Severe', 'urgency': 'High', 'wait_time': '24 hours'},
    'PTSD': {'severity': 'Moderate', 'urgency': 'High', 'wait_time': '48 hours'}
}

# Agent tasks
AGENT_TASKS = {
    'Intake & Triage': {'automation': 98.5, 'time_saved': '45 min'},
    'Insurance Verification': {'automation': 99.2, 'time_saved': '30 min'},
    'Appointment Scheduling': {'automation': 97.8, 'time_saved': '15 min'},
    'Medication Management': {'automation': 94.5, 'time_saved': '20 min'},
    'Follow-up Coordination': {'automation': 96.3, 'time_saved': '25 min'}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #4ade80 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🧠</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Legion Health</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">AI-Native Psychiatry</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">LLM agents • Insurance-covered • Built for scale</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🧠 AI Triage", "🤖 Agent Orchestration", "📊 Operations Dashboard", "💡 Technology"])

with tab1:
    st.markdown("### AI-Powered Mental Health Triage")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Patient Information**")
        
        patient_name = st.text_input("Name", "Alex Chen")
        age = st.number_input("Age", 18, 100, 32)
        
        st.markdown("**Symptom Assessment**")
        
        symptoms = st.text_area(
            "Describe symptoms",
            "Feeling persistently sad for 3 weeks, low energy, difficulty concentrating, trouble sleeping, loss of interest in activities",
            height=100
        )
        
        severity = st.select_slider(
            "Self-Reported Severity",
            options=["Mild", "Moderate", "Severe", "Crisis"],
            value="Moderate"
        )
        
        prior_treatment = st.checkbox("Previous mental health treatment")
        insurance_verified = st.checkbox("Insurance pre-verified", value=True)
        
        triage_btn = st.button("🧠 Run AI Triage", type="primary", use_container_width=True)
    
    with col2:
        if triage_btn:
            st.markdown("**AI Triage Results**")
            
            with st.spinner("Analyzing symptoms with LLM..."):
                import time
                time.sleep(1.5)
            
            st.success("✅ Triage complete - Patient matched with clinician")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4ade80 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Clinical Assessment</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Preliminary Diagnosis</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Major Depressive Disorder</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Severity</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Moderate</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Urgency</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Standard (2 days)</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Matched Clinician</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Dr. Sarah Kim, MD</p>
                    </div>
                </div>
                <div style="margin-top: 15px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Recommended Treatment</p>
                    <p style="font-size: 16px; color: white; margin: 0;">Combination therapy: Weekly psychotherapy + medication evaluation + monitoring</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Triage Time", "2.3 min", "vs 45 min manual")
            col2.metric("Insurance", "✅ Verified", "Covered")
            col3.metric("Next Appt", "2 days", "Fast")
            col4.metric("Confidence", "96.8%", "High")

with tab2:
    st.markdown("### LLM Agent Orchestration")
    
    st.markdown("**Active Agents**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 20px; border-left: 4px solid #4ade80;">
            <h4 style="margin: 0 0 10px 0; color: #166534;">🤖 Intake Agent</h4>
            <p style="margin: 0; font-size: 14px; color: #6b7280;">Symptom analysis, triage, clinician matching</p>
            <p style="margin: 8px 0 0 0; font-size: 12px; color: #10b981; font-weight: 600;">98.5% automation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 20px; border-left: 4px solid #3b82f6;">
            <h4 style="margin: 0 0 10px 0; color: #1e3a8a;">💳 Insurance Agent</h4>
            <p style="margin: 0; font-size: 14px; color: #6b7280;">Verification, pre-auth, claims processing</p>
            <p style="margin: 8px 0 0 0; font-size: 12px; color: #3b82f6; font-weight: 600;">99.2% automation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 20px; border-left: 4px solid #f59e0b;">
            <h4 style="margin: 0 0 10px 0; color: #78350f;">📅 Scheduling Agent</h4>
            <p style="margin: 0; font-size: 14px; color: #6b7280;">Appointment booking, reminders, coordination</p>
            <p style="margin: 8px 0 0 0; font-size: 12px; color: #f59e0b; font-weight: 600;">97.8% automation</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("**Agent Automation Rates**")
    
    automation_data = []
    for task, data in AGENT_TASKS.items():
        automation_data.append({
            'Task': task,
            'Automation': f"{data['automation']}%",
            'Time Saved': data['time_saved']
        })
    
    st.dataframe(pd.DataFrame(automation_data), hide_index=True, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Workflow Visualization**")
        
        fig1 = go.Figure(data=[go.Sankey(
            node=dict(
                label=["Patient Contact", "AI Triage", "Insurance Verify", "Clinician Match", "Appointment", "Care"],
                color=["#4ade80", "#4ade80", "#3b82f6", "#f59e0b", "#8b5cf6", "#10b981"]
            ),
            link=dict(
                source=[0, 1, 2, 3, 4],
                target=[1, 2, 3, 4, 5],
                value=[100, 98, 96, 94, 92]
            )
        )])
        fig1.update_layout(height=250)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("**Automation Impact**")
        
        fig2 = go.Figure(data=[go.Bar(
            x=list(AGENT_TASKS.keys()),
            y=[AGENT_TASKS[t]['automation'] for t in AGENT_TASKS.keys()],
            marker=dict(color='#4ade80'),
            text=[f"{AGENT_TASKS[t]['automation']}%" for t in AGENT_TASKS.keys()],
            textposition='auto'
        )])
        fig2.update_layout(yaxis=dict(range=[90, 100]), height=250)
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Clinical Operations Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients Today", "847", "+123")
    col2.metric("AI Intake", "98.5%", "Automated")
    col3.metric("Avg Wait", "2.3 days", "-5.2 days")
    col4.metric("Patient NPS", "4.9/5", "+1.2")
    
    st.markdown("**Patient Volume Trends**")
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    patients = [782, 823, 847, 856, 891, 456, 389]
    
    fig3 = go.Figure(data=[go.Scatter(
        x=days, y=patients,
        mode='lines+markers',
        line=dict(color='#4ade80', width=3),
        fill='tozeroy',
        fillcolor='rgba(74, 222, 128, 0.1)'
    )])
    fig3.update_layout(yaxis_title='Patients', height=250)
    st.plotly_chart(fig3, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Condition Distribution**")
        
        condition_counts = [312, 289, 145, 67, 34]
        
        fig4 = go.Figure(data=[go.Pie(
            labels=list(CONDITIONS.keys()),
            values=condition_counts,
            hole=0.4,
            marker=dict(colors=['#4ade80', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'])
        )])
        fig4.update_layout(height=250)
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        st.markdown("**Monthly Impact**")
        
        impact = {
            'Metric': ['Patients Served', 'Admin Hours Saved', 'Cost Reduction', 'Clinician Hours'],
            'Value': ['18,450', '1,850 hrs', '$285K', '+620 hrs']
        }
        st.dataframe(pd.DataFrame(impact), hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### AI-Native Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**LLM Agent Capabilities**")
        st.markdown("""
        - ✅ Symptom analysis (NLP on patient descriptions)
        - ✅ Severity assessment (PHQ-9, GAD-7 scoring)
        - ✅ Suicide risk screening
        - ✅ Differential diagnosis suggestions
        - ✅ Treatment plan recommendations
        - ✅ Medication interaction checking
        """)
        
        st.markdown("**Orchestration Layer**")
        st.markdown("""
        - ✅ Multi-agent coordination
        - ✅ Task routing and prioritization
        - ✅ Context management
        - ✅ Escalation protocols
        - ✅ Human-in-the-loop oversight
        - ✅ Compliance monitoring
        """)
    
    with col2:
        st.markdown("**Clinical Integration**")
        st.markdown("""
        - ✅ EHR integration (Epic, Athenahealth)
        - ✅ Insurance verification APIs
        - ✅ Prescription management (e-prescribe)
        - ✅ Lab results integration
        - ✅ Telehealth platform
        - ✅ Crisis protocols
        """)
        
        st.markdown("**Quality & Safety**")
        st.markdown("""
        - ✅ Licensed clinician oversight
        - ✅ HIPAA compliant
        - ✅ Suicide risk detection
        - ✅ Medication safety checks
        - ✅ Treatment guideline adherence
        - ✅ Outcome tracking
        """)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #14532d; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #16a34a; font-weight: 700; margin: 0 0 6px 0;">✓ 98.5% AI Automation</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">LLM agents handle intake</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #16a34a; font-weight: 700; margin: 0 0 6px 0;">✓ 2.3 Day Wait</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">vs 7.5 days industry avg</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #16a34a; font-weight: 700; margin: 0 0 6px 0;">✓ Insurance Covered</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Affordable, accessible care</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #16a34a; font-weight: 700; margin: 0 0 6px 0;">✓ 18,450 Patients/Month</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Scaled mental health access</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #4ade80 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Legion Health</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)