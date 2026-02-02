"""
Wedge - Health AI Operating System
Unified platform for healthcare AI applications
Built for Wedge by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Wedge", page_icon="🏥", layout="wide")

# AI applications
AI_APPLICATIONS = {
    'Clinical Documentation': {'users': 847, 'time_saved': '45 min/day', 'satisfaction': 4.8},
    'Diagnostic Assistance': {'users': 623, 'time_saved': '30 min/day', 'satisfaction': 4.9},
    'Patient Triage': {'users': 1247, 'time_saved': '20 min/day', 'satisfaction': 4.7},
    'Medication Management': {'users': 534, 'time_saved': '25 min/day', 'satisfaction': 4.6},
    'Care Coordination': {'users': 789, 'time_saved': '35 min/day', 'satisfaction': 4.8}
}

# Platform metrics
PLATFORM_METRICS = {
    'Active Users': 4040,
    'AI Apps Deployed': 12,
    'Daily Interactions': 18450,
    'Avg Time Saved': 31,  # minutes per user
    'User Satisfaction': 4.8
}

# Healthcare workflows
WORKFLOWS = {
    'Patient Intake': ['Registration', 'Insurance verify', 'Triage', 'Scheduling'],
    'Clinical Care': ['Vitals', 'Assessment', 'Diagnosis', 'Treatment plan'],
    'Documentation': ['SOAP note', 'Orders', 'Prescriptions', 'Coding'],
    'Discharge': ['Instructions', 'Follow-up', 'Prescriptions', 'Billing']
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #0891b2 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🏥</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Wedge</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Health AI Operating System</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Unified AI platform • 12 apps • 4,040 users • 31 min saved daily</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏥 AI App Marketplace", "📊 Usage Dashboard", "⚡ Workflow Automation", "💡 Platform Features"])

with tab1:
    st.markdown("### Health AI Application Marketplace")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Browse AI Apps**")
        
        category = st.selectbox(
            "Category",
            ["All Apps", "Clinical Documentation", "Diagnostics", "Triage", "Medications", "Coordination"]
        )
        
        st.markdown("**Featured Apps**")
        
        for app_name in list(AI_APPLICATIONS.keys())[:3]:
            app_data = AI_APPLICATIONS[app_name]
            st.markdown(f"""
            <div style="background: white; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid #0891b2; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <h4 style="margin: 0 0 8px 0; color: #1f2937; font-size: 16px; font-weight: 700;">{app_name}</h4>
                <p style="margin: 4px 0; color: #6b7280; font-size: 13px;">👥 {app_data['users']} users</p>
                <p style="margin: 4px 0; color: #6b7280; font-size: 13px;">⏱️ Saves {app_data['time_saved']}</p>
                <p style="margin: 4px 0; color: #6b7280; font-size: 13px;">⭐ {app_data['satisfaction']}/5</p>
            </div>
            """, unsafe_allow_html=True)
        
        install_btn = st.button("📥 Install App", type="primary", use_container_width=True)
    
    with col2:
        if install_btn:
            st.markdown("**Installation Progress**")
            
            with st.spinner("Installing Clinical Documentation AI..."):
                import time
                time.sleep(1.5)
            
            st.success("✅ App installed successfully!")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #0891b2 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Clinical Documentation AI</h3>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                    <p style="color: white; margin: 0; line-height: 1.6;">
                        <strong>What it does:</strong> Automatically generates SOAP notes, ICD-10 codes, and CPT codes from patient encounters using ambient listening and clinical NLP.
                    </p>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Time Saved</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">45 min/day</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Accuracy</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">96.5%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Install Time", "90s", "Fast")
            col2.metric("EHR Integration", "✅ Epic", "Connected")
            col3.metric("Users", "847", "Active")

with tab2:
    st.markdown("### Platform Usage Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Users", "4,040", "+234")
    col2.metric("AI Apps", "12", "Deployed")
    col3.metric("Daily Interactions", "18,450", "+1,234")
    col4.metric("Time Saved/User", "31 min", "+4 min")
    
    st.markdown("**App Usage Distribution**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        app_names = list(AI_APPLICATIONS.keys())
        app_users = [AI_APPLICATIONS[app]['users'] for app in app_names]
        
        fig1 = go.Figure(data=[go.Pie(
            labels=app_names,
            values=app_users,
            hole=0.4,
            marker=dict(colors=['#0891b2', '#10b981', '#3b82f6', '#f59e0b', '#8b5cf6'])
        )])
        fig1.update_layout(height=300, title='User Distribution')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = go.Figure(data=[go.Bar(
            x=app_names,
            y=app_users,
            marker=dict(color='#0891b2'),
            text=app_users,
            textposition='auto'
        )])
        fig2.update_layout(yaxis_title='Active Users', height=300)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("**Weekly Activity Trends**")
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    interactions = [16234, 17856, 18450, 18123, 17890, 8234, 6789]
    time_saved = [28.5, 30.2, 31.0, 30.5, 29.8, 15.2, 12.8]
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=days, y=interactions,
        name='Interactions',
        marker=dict(color='#0891b2')
    ))
    fig3.add_trace(go.Scatter(
        x=days, y=time_saved,
        name='Time Saved/User (min)',
        yaxis='y2',
        line=dict(color='#10b981', width=3)
    ))
    fig3.update_layout(
        yaxis=dict(title='Interactions'),
        yaxis2=dict(title='Time Saved (min)', overlaying='y', side='right'),
        height=300
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.markdown("### Automated Clinical Workflows")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Workflow Selection**")
        
        workflow_type = st.selectbox("Select Workflow", list(WORKFLOWS.keys()))
        
        st.markdown(f"**Steps in {workflow_type}:**")
        
        for i, step in enumerate(WORKFLOWS[workflow_type], 1):
            automation_pct = np.random.uniform(85, 98)
            st.markdown(f"""
            <div style="background: white; border-radius: 8px; padding: 12px; margin: 8px 0; border-left: 3px solid #0891b2;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #1f2937; font-weight: 600;">{i}. {step}</span>
                    <span style="color: #059669; font-weight: 700;">{automation_pct:.1f}% automated</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        optimize_btn = st.button("⚡ Optimize Workflow", use_container_width=True)
    
    with col2:
        if optimize_btn:
            st.markdown("**Workflow Optimization Results**")
            
            st.success("✅ Workflow optimized - 15 minutes saved per patient")
            
            before_after = {
                'Step': WORKFLOWS[workflow_type],
                'Before (min)': [8, 12, 15, 10],
                'After (min)': [2, 3, 4, 2],
                'Time Saved': ['6 min', '9 min', '11 min', '8 min']
            }
            st.dataframe(pd.DataFrame(before_after), hide_index=True, use_container_width=True)
            
            total_before = sum([8, 12, 15, 10])
            total_after = sum([2, 3, 4, 2])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Before", f"{total_before} min", "Manual")
            col2.metric("After", f"{total_after} min", "AI-assisted")
            col3.metric("Saved", f"{total_before - total_after} min", f"{(total_before - total_after)/total_before*100:.0f}%")

with tab4:
    st.markdown("### Platform Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**AI Application Types**")
        st.markdown("""
        - ✅ Clinical documentation (ambient listening)
        - ✅ Diagnostic assistance (image analysis)
        - ✅ Patient triage (symptom analysis)
        - ✅ Medication management (interaction checking)
        - ✅ Care coordination (workflow automation)
        - ✅ Billing & coding (ICD-10/CPT)
        - ✅ Quality metrics (analytics)
        - ✅ Patient engagement (chatbots)
        """)
        
        st.markdown("**Platform Features**")
        st.markdown("""
        - ✅ Unified API for all health AI apps
        - ✅ Single sign-on (SSO)
        - ✅ Centralized data governance
        - ✅ Cross-app analytics
        - ✅ Marketplace for AI apps
        - ✅ Developer SDK
        """)
    
    with col2:
        st.markdown("**Integration & Security**")
        st.markdown("""
        - ✅ EHR integration (Epic, Cerner)
        - ✅ HL7/FHIR data pipelines
        - ✅ HIPAA compliant infrastructure
        - ✅ SOC 2 Type II certified
        - ✅ Encrypted data at rest/transit
        - ✅ Audit logging
        """)
        
        st.markdown("**Developer Tools**")
        st.markdown("""
        - ✅ Python/JavaScript SDKs
        - ✅ REST APIs
        - ✅ Webhook support
        - ✅ Testing sandbox
        - ✅ Documentation portal
        - ✅ App submission process
        """)
    
    st.markdown("**Deployed AI Applications**")
    
    apps_table = {
        'Application': ['Clinical Docs', 'Diagnostic AI', 'Patient Triage', 'Med Management', 'Care Coord', 'Billing AI'],
        'Category': ['Documentation', 'Diagnostics', 'Operations', 'Clinical', 'Operations', 'Revenue'],
        'Users': [847, 623, 1247, 534, 789, 456],
        'Satisfaction': ['4.8/5', '4.9/5', '4.7/5', '4.6/5', '4.8/5', '4.5/5']
    }
    st.dataframe(pd.DataFrame(apps_table), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #cffafe 0%, #a5f3fc 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #164e63; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0891b2; font-weight: 700; margin: 0 0 6px 0;">✓ 12 AI Apps</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Unified health AI platform</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0891b2; font-weight: 700; margin: 0 0 6px 0;">✓ 4,040 Users</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Active clinicians</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0891b2; font-weight: 700; margin: 0 0 6px 0;">✓ 31 Min Saved</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Per user daily</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0891b2; font-weight: 700; margin: 0 0 6px 0;">✓ 4.8/5 Satisfaction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">High user approval</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #0891b2 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Wedge</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)