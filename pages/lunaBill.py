"""
LunaBill - Voice AI Medical Billing
Automated billing and coding through voice commands
Built for LunaBill by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="LunaBill", page_icon="🎙️", layout="wide")

# Billing codes
CPT_CODES = {
    '99213': {'description': 'Office visit, 15 min', 'RVU': 1.3, 'reimbursement': 95},
    '99214': {'description': 'Office visit, 25 min', 'RVU': 1.9, 'reimbursement': 145},
    '99215': {'description': 'Office visit, 40 min', 'RVU': 2.8, 'reimbursement': 215},
    '71045': {'description': 'Chest X-ray, single view', 'RVU': 0.7, 'reimbursement': 52},
    '80053': {'description': 'Comprehensive metabolic panel', 'RVU': 0.8, 'reimbursement': 28}
}

ICD10_CODES = {
    'J06.9': 'Upper respiratory infection',
    'M54.5': 'Low back pain',
    'E11.9': 'Type 2 diabetes',
    'I10': 'Essential hypertension',
    'F41.1': 'Generalized anxiety disorder'
}

# Voice billing metrics
VOICE_METRICS = {
    'Coding Accuracy': 96.8,
    'Voice Recognition': 98.5,
    'Processing Speed': 1.2,  # seconds
    'Auto-Submit Rate': 94.3,
    'Reimbursement Rate': 92.7
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #10b981 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🎙️</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">LunaBill</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Voice AI Medical Billing</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Voice-to-code • 96.8% accuracy • 1.2s processing • $420K monthly capture</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎙️ Voice Billing", "📊 Billing Dashboard", "💰 Revenue Capture", "💡 Technology"])

with tab1:
    st.markdown("### Voice-Activated Medical Billing")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Voice Input Simulation**")
        
        st.info("🎤 In production: Say commands to microphone")
        
        voice_command = st.text_area(
            "Voice Command (Simulated)",
            "Bill 99214 for Jennifer Martinez, diagnosis low back pain M54.5, completed 25-minute office visit with exam and treatment plan",
            height=100
        )
        
        st.markdown("**Encounter Details**")
        
        encounter_date = st.date_input("Visit Date", datetime.now())
        provider = st.selectbox("Provider", ["Dr. Sarah Kim", "Dr. Robert Chen", "Dr. Emily Park"])
        
        st.markdown("**Modifiers (Optional)**")
        
        modifier_25 = st.checkbox("Modifier 25 (Separate E/M)")
        modifier_59 = st.checkbox("Modifier 59 (Distinct service)")
        
        st.markdown("**Submission**")
        
        auto_submit = st.checkbox("Auto-submit to clearinghouse", value=True)
        verify_eligibility = st.checkbox("Verify insurance eligibility", value=True)
        
        process_btn = st.button("🎙️ Process Voice Billing", type="primary", use_container_width=True)
    
    with col2:
        if process_btn:
            st.markdown("**Voice Processing Results**")
            
            with st.spinner("Processing voice command..."):
                import time
                time.sleep(1.2)
            
            st.success("✅ Claim generated and submitted in 1.2 seconds!")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #10b981 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Generated Claim</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Patient</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Jennifer Martinez</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">CPT Code</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">99214</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">ICD-10 Code</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">M54.5</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Expected Reimbursement</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">$145</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Claim Details**")
            
            claim_details = {
                'Field': ['CPT Code', 'Description', 'ICD-10 Code', 'Diagnosis', 'RVU', 'Expected Payment'],
                'Value': ['99214', 'Office visit, 25 min', 'M54.5', 'Low back pain', '1.9', '$145'],
                'Extracted From': ['Voice', 'Auto-mapped', 'Voice', 'Voice', 'Code lookup', 'Fee schedule'],
                'Confidence': ['98.5%', '100%', '97.2%', '96.8%', '100%', '99.1%']
            }
            st.dataframe(pd.DataFrame(claim_details), hide_index=True, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Voice Accuracy", "98.5%", "✓")
            col2.metric("Coding Accuracy", "96.8%", "✓")
            col3.metric("Processing", "1.2s", "Fast")
            col4.metric("Submitted", "✅", "Auto")

with tab2:
    st.markdown("### Billing Operations Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Claims Today", "847", "+123")
    col2.metric("Voice Processed", "799", "94.3%")
    col3.metric("Avg Time/Claim", "1.2s", "vs 8 min")
    col4.metric("Capture Rate", "$420K", "Monthly")
    
    st.markdown("**Billing Volume by CPT Category**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        categories = ['E/M Visits', 'Procedures', 'Imaging', 'Lab Tests', 'Injections']
        volumes = [423, 189, 156, 234, 89]
        
        fig1 = go.Figure(data=[go.Pie(
            labels=categories,
            values=volumes,
            hole=0.4,
            marker=dict(colors=['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899'])
        )])
        fig1.update_layout(height=300, title='Service Mix')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = go.Figure(data=[go.Bar(
            x=categories,
            y=volumes,
            marker=dict(color='#10b981'),
            text=volumes,
            textposition='auto'
        )])
        fig2.update_layout(yaxis_title='Claims', height=300)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("**Weekly Revenue Capture**")
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    revenue = [98000, 112000, 105000, 108000, 115000, 45000, 32000]
    
    fig3 = go.Figure(data=[go.Scatter(
        x=days, y=revenue,
        mode='lines+markers',
        line=dict(color='#10b981', width=3),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    )])
    fig3.update_layout(yaxis_title='Revenue ($)', height=250)
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.markdown("### Revenue Capture & Optimization")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Revenue Impact**")
        
        revenue_metrics = {
            'Metric': ['Monthly Capture', 'Missed Charges Prevented', 'Coding Accuracy', 'Clean Claim Rate', 'Days in AR'],
            'Value': ['$420K', '$85K', '96.8%', '94.7%', '28 days'],
            'vs Manual': ['+$85K', '+$85K', '+4.2%', '+8.5%', '-12 days']
        }
        st.dataframe(pd.DataFrame(revenue_metrics), hide_index=True, use_container_width=True)
        
        st.markdown("**Capture Rate by Service**")
        
        services = ['Office Visits', 'Procedures', 'Imaging', 'Lab Tests']
        manual_capture = [87.5, 82.3, 79.8, 91.2]
        ai_capture = [98.2, 95.7, 94.3, 98.9]
        
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name='Manual', x=services, y=manual_capture, marker_color='#ef4444'))
        fig4.add_trace(go.Bar(name='LunaBill AI', x=services, y=ai_capture, marker_color='#10b981'))
        fig4.update_layout(barmode='group', yaxis=dict(range=[75, 100]), yaxis_title='Capture Rate (%)', height=250)
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        st.markdown("**Revenue Leakage Prevention**")
        
        leakage = {
            'Leak Type': ['Undercoding', 'Missed Services', 'Incorrect Modifiers', 'No Documentation Link'],
            'Monthly Loss (Manual)': ['$45K', '$28K', '$8K', '$4K'],
            'With LunaBill': ['$2K', '$1K', '$0', '$0'],
            'Prevented': ['$43K', '$27K', '$8K', '$4K']
        }
        st.dataframe(pd.DataFrame(leakage), hide_index=True, use_container_width=True)
        
        st.markdown("**Total Prevention: $82K/month**")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Revenue Capture", "$420K/mo", "+$85K")
        col2.metric("Leakage Prevented", "$82K/mo", "97% reduction")
        col3.metric("ROI", "680%", "High")

with tab4:
    st.markdown("### Voice AI Technology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Voice Recognition**")
        st.markdown("""
        - ✅ Medical terminology optimized
        - ✅ 98.5% speech recognition accuracy
        - ✅ Multi-accent support
        - ✅ Noise filtering (clinic environment)
        - ✅ Real-time transcription
        - ✅ Natural language understanding
        """)
        
        st.markdown("**Automated Coding**")
        st.markdown("""
        - ✅ CPT code extraction (E/M, procedures)
        - ✅ ICD-10 diagnosis coding
        - ✅ Modifier application
        - ✅ Documentation linking
        - ✅ Compliance checking
        - ✅ Upcoding prevention
        """)
    
    with col2:
        st.markdown("**Clinical Intelligence**")
        st.markdown("""
        - ✅ NLP for clinical notes
        - ✅ Service-diagnosis alignment
        - ✅ Medical necessity validation
        - ✅ Documentation completeness check
        - ✅ Billing guideline adherence
        - ✅ Audit risk detection
        """)
        
        st.markdown("**Workflow Integration**")
        st.markdown("""
        - ✅ EHR integration (Epic, Cerner)
        - ✅ Practice management systems
        - ✅ Clearinghouse submission
        - ✅ Real-time eligibility checks
        - ✅ Claim status tracking
        - ✅ Denial management
        """)
    
    st.markdown("**Voice Command Examples**")
    
    commands = {
        'Voice Command': [
            '"Bill 99214 for Sarah Johnson, diagnosis hypertension"',
            '"Add chest X-ray to claim for Michael Chen"',
            '"Office visit level 3, anxiety disorder, prescribed Lexapro"',
            '"Submit all pending claims for today"',
            '"Check reimbursement status for claim 12345"'
        ],
        'AI Action': [
            'Generates claim with CPT 99214 + ICD I10',
            'Adds CPT 71045 to existing claim',
            'Creates 99213 + F41.1 + prescription documentation',
            'Batch submits 47 claims to clearinghouse',
            'Queries payer portal, returns status'
        ],
        'Time': ['1.2s', '0.8s', '1.5s', '3.2s', '2.1s']
    }
    st.dataframe(pd.DataFrame(commands), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #065f46; font-size: 24px; font-weight: 800;">💡 System Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ 98.5% Voice Accuracy</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Medical terminology optimized</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ 96.8% Coding Accuracy</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">CPT + ICD-10 precision</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ 1.2s Processing</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">vs 8 min manual</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ $420K Capture</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Monthly revenue optimization</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #10b981 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for LunaBill</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)