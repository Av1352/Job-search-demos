"""
Parachute - Deploy AI in Hospitals
MLOps platform for healthcare AI deployment and monitoring
Built for Parachute by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Parachute", page_icon="🏥", layout="wide")

# AI models in hospitals
HOSPITAL_AI_MODELS = {
    'Sepsis Prediction': {'dept': 'ICU', 'accuracy': 94.5, 'alerts_day': 23, 'deployment': 'Active'},
    'Readmission Risk': {'dept': 'General Med', 'accuracy': 89.2, 'alerts_day': 67, 'deployment': 'Active'},
    'Fall Risk': {'dept': 'Geriatrics', 'accuracy': 91.8, 'alerts_day': 45, 'deployment': 'Active'},
    'Medication Error': {'dept': 'Pharmacy', 'accuracy': 97.3, 'alerts_day': 12, 'deployment': 'Active'},
    'ED Wait Time': {'dept': 'Emergency', 'accuracy': 88.7, 'alerts_day': 156, 'deployment': 'Active'}
}

# Deployment metrics
DEPLOYMENT_METRICS = {
    'Model Uptime': 99.97,
    'Inference Latency': 45,  # ms
    'Prediction Accuracy': 92.3,
    'Alert Response Time': 2.3,  # min
    'Integration Success': 98.5
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #16a34a 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🏥</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Parachute</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Deploy AI in Hospitals</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">MLOps for healthcare • Model monitoring • EHR integration</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Deploy Model", "📊 Model Dashboard", "⚡ Performance Monitoring", "💡 Platform Features"])

with tab1:
    st.markdown("### Deploy AI Model to Hospital")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Model Configuration**")
        
        model_type = st.selectbox(
            "Select AI Model",
            ["Sepsis Prediction", "Readmission Risk", "Fall Risk", "Medication Error Detection", "ED Wait Time Prediction"]
        )
        
        st.markdown("**Deployment Settings**")
        
        department = st.selectbox("Target Department", ["ICU", "Emergency", "General Med", "Geriatrics", "Pharmacy"])
        
        ehr_system = st.selectbox("EHR System", ["Epic", "Cerner", "Meditech", "Allscripts"])
        
        st.markdown("**Integration Options**")
        
        real_time = st.checkbox("Real-time predictions", value=True)
        ehr_integration = st.checkbox("Bi-directional EHR sync", value=True)
        alert_system = st.checkbox("Clinical decision support alerts", value=True)
        
        st.markdown("**Monitoring**")
        
        enable_drift = st.checkbox("Model drift detection", value=True)
        enable_explainability = st.checkbox("Prediction explainability", value=True)
        
        deploy_btn = st.button("🚀 Deploy to Production", type="primary", use_container_width=True)
    
    with col2:
        if deploy_btn:
            st.markdown("**Deployment Progress**")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            stages = [
                ("Validating model...", 0.2),
                ("Integrating with EHR...", 0.4),
                ("Setting up monitoring...", 0.6),
                ("Testing predictions...", 0.8),
                ("Activating alerts...", 1.0)
            ]
            
            for stage, progress in stages:
                status_text.text(stage)
                progress_bar.progress(progress)
                time.sleep(0.5)
            
            st.success("✅ Model deployed successfully!")
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #16a34a 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Deployment Status</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Model</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{model_type}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Department</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{department}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">EHR Integration</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{ehr_system} ✅</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Status</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">🟢 Live</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Deployment Time", "12 min", "Fast")
            col2.metric("Uptime", "99.97%", "✓")
            col3.metric("Latency", "45ms", "<100ms")
            col4.metric("Predictions", "847/day", "Active")

with tab2:
    st.markdown("### Hospital AI Model Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Models", "5", "Deployed")
    col2.metric("Predictions Today", "303", "Real-time")
    col3.metric("Avg Accuracy", "92.3%", "+1.2%")
    col4.metric("Alerts Generated", "303", "Actionable")
    
    st.markdown("**Deployed Models Overview**")
    
    model_data = []
    for model, data in HOSPITAL_AI_MODELS.items():
        model_data.append({
            'Model': model,
            'Department': data['dept'],
            'Accuracy': f"{data['accuracy']}%",
            'Alerts/Day': data['alerts_day'],
            'Status': f"🟢 {data['deployment']}"
        })
    
    st.dataframe(pd.DataFrame(model_data), hide_index=True, use_container_width=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Model Accuracy Distribution**")
        
        fig1 = go.Figure(data=[go.Bar(
            x=list(HOSPITAL_AI_MODELS.keys()),
            y=[HOSPITAL_AI_MODELS[m]['accuracy'] for m in HOSPITAL_AI_MODELS.keys()],
            marker=dict(
                color=[HOSPITAL_AI_MODELS[m]['accuracy'] for m in HOSPITAL_AI_MODELS.keys()],
                colorscale='RdYlGn',
                cmin=85,
                cmax=100
            ),
            text=[f"{HOSPITAL_AI_MODELS[m]['accuracy']}%" for m in HOSPITAL_AI_MODELS.keys()],
            textposition='auto'
        )])
        fig1.update_layout(yaxis=dict(range=[80, 100]), height=250)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("**Daily Alert Volume**")
        
        fig2 = go.Figure(data=[go.Bar(
            x=list(HOSPITAL_AI_MODELS.keys()),
            y=[HOSPITAL_AI_MODELS[m]['alerts_day'] for m in HOSPITAL_AI_MODELS.keys()],
            marker=dict(color='#16a34a'),
            text=[HOSPITAL_AI_MODELS[m]['alerts_day'] for m in HOSPITAL_AI_MODELS.keys()],
            textposition='auto'
        )])
        fig2.update_layout(yaxis_title='Alerts/Day', height=250)
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Real-Time Model Monitoring")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**System Metrics**")
        
        metrics_data = []
        for metric, value in DEPLOYMENT_METRICS.items():
            if 'Uptime' in metric or 'Accuracy' in metric or 'Success' in metric:
                display = f"{value}%"
            elif 'Latency' in metric:
                display = f"{value}ms"
            elif 'Time' in metric:
                display = f"{value} min"
            else:
                display = str(value)
            
            metrics_data.append({
                'Metric': metric,
                'Value': display,
                'Status': '✅ Optimal'
            })
        
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Model Drift Detection**")
        
        drift_status = {
            'Model': ['Sepsis Pred', 'Readmission', 'Fall Risk', 'Med Error', 'ED Wait'],
            'Data Drift': ['🟢 None', '🟢 None', '🟡 Minor', '🟢 None', '🟢 None'],
            'Performance': ['Stable', 'Stable', 'Stable', 'Stable', 'Stable']
        }
        st.dataframe(pd.DataFrame(drift_status), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Prediction Volume (Last 7 Days)**")
        
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        predictions = [289, 312, 303, 298, 315, 187, 156]
        
        fig3 = go.Figure(data=[go.Scatter(
            x=days, y=predictions,
            mode='lines+markers',
            line=dict(color='#16a34a', width=3),
            fill='tozeroy',
            fillcolor='rgba(22, 163, 74, 0.1)'
        )])
        fig3.update_layout(yaxis_title='Predictions', height=200)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("**Latency Distribution**")
        
        latencies = np.random.gamma(2, 20, 1000)
        
        fig4 = go.Figure(data=[go.Histogram(
            x=latencies,
            nbinsx=40,
            marker=dict(color='#16a34a')
        )])
        fig4.update_layout(xaxis_title='Latency (ms)', yaxis_title='Count', height=200)
        st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.markdown("### Healthcare MLOps Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Deployment Features**")
        st.markdown("""
        - ✅ One-click model deployment
        - ✅ EHR integration (Epic, Cerner, Meditech)
        - ✅ HL7/FHIR data pipelines
        - ✅ Real-time inference (<100ms)
        - ✅ Clinical decision support (CDS)
        - ✅ Alert workflow integration
        - ✅ A/B testing framework
        - ✅ Rollback capability
        """)
        
        st.markdown("**Monitoring & Observability**")
        st.markdown("""
        - ✅ Real-time performance dashboards
        - ✅ Model drift detection
        - ✅ Data quality monitoring
        - ✅ Prediction explainability (SHAP)
        - ✅ Alert analytics
        - ✅ Clinician feedback loops
        """)
    
    with col2:
        st.markdown("**Compliance & Security**")
        st.markdown("""
        - ✅ HIPAA compliant infrastructure
        - ✅ FDA 21 CFR Part 11
        - ✅ SOC 2 Type II certified
        - ✅ Audit trail logging
        - ✅ Role-based access control
        - ✅ De-identification pipelines
        """)
        
        st.markdown("**Supported Use Cases**")
        st.markdown("""
        - ✅ Sepsis early warning
        - ✅ Readmission risk scoring
        - ✅ Fall risk assessment
        - ✅ Medication error prevention
        - ✅ ED capacity forecasting
        - ✅ Bed allocation optimization
        - ✅ No-show prediction
        - ✅ Length of stay estimation
        """)
    
    st.markdown("**Integration Capabilities**")
    
    integrations = {
        'System': ['Epic EHR', 'Cerner EHR', 'Meditech', 'Philips Monitors', 'GE Healthcare', 'PACS Systems'],
        'Protocol': ['HL7 v2', 'FHIR', 'HL7 v2', 'HL7 v2', 'DICOM', 'DICOM'],
        'Status': ['✅ Supported', '✅ Supported', '✅ Supported', '✅ Supported', '✅ Supported', '✅ Supported']
    }
    st.dataframe(pd.DataFrame(integrations), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135d, #dcfce7 0%, #bbf7d0 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #14532d; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #15803d; font-weight: 700; margin: 0 0 6px 0;">✓ 99.97% Uptime</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Production-grade reliability</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #15803d; font-weight: 700; margin: 0 0 6px 0;">✓ 45ms Latency</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Real-time predictions</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #15803d; font-weight: 700; margin: 0 0 6px 0;">✓ 5 AI Models</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Multi-department deployment</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #15803d; font-weight: 700; margin: 0 0 6px 0;">✓ 98.5% Integration</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Seamless EHR connectivity</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #16a34a 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Parachute</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)