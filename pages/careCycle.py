"""
careCycle - Voice AI for Medicare Agencies
AI-powered Medicare eligibility verification and patient engagement
Built for careCycle by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="careCycle", page_icon="🎙️", layout="wide")

# Medicare plans
MEDICARE_PLANS = {
    'Medicare Part A': {'monthly_premium': 0, 'deductible': 1600, 'coverage': 'Hospital'},
    'Medicare Part B': {'monthly_premium': 174.70, 'deductible': 240, 'coverage': 'Medical'},
    'Medicare Part D': {'monthly_premium': 55, 'deductible': 505, 'coverage': 'Prescription'},
    'Medicare Advantage': {'monthly_premium': 25, 'deductible': 0, 'coverage': 'All-in-One'},
    'Medigap Plan G': {'monthly_premium': 150, 'deductible': 240, 'coverage': 'Supplemental'}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #3b82f6 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🎙️</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">careCycle</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Voice AI for Medicare Agencies</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Automated eligibility checks • Patient engagement • 24/7 availability</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎙️ Voice Eligibility Check", "📊 Agency Dashboard", "💡 System Features"])

with tab1:
    st.markdown("### Medicare Eligibility Verification")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Patient Information**")
        patient_name = st.text_input("Patient Name", placeholder="Mary Johnson")
        dob = st.date_input("Date of Birth", value=datetime(1955, 3, 15))
        medicare_id = st.text_input("Medicare Number", placeholder="1AB2-CD3-EF45")
        
        st.markdown("**Voice Inquiry Simulation**")
        inquiry_type = st.selectbox(
            "Select Voice Query",
            [
                "Check my Medicare eligibility",
                "What does my plan cover?",
                "Find a doctor near me",
                "Schedule an appointment",
                "Refill my prescription"
            ]
        )
        
        check_btn = st.button("🎙️ Process Voice Request", type="primary", use_container_width=True)
    
    with col2:
        if check_btn and patient_name and medicare_id:
            age = (datetime.now() - datetime.combine(dob, datetime.min.time())).days // 365
            
            # Determine eligibility
            is_eligible = age >= 65
            has_part_a = np.random.random() > 0.1
            has_part_b = np.random.random() > 0.15
            
            status_color = "#10b981" if is_eligible else "#ef4444"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {status_color} 0%, #73BA9B 100%); padding: 30px; border-radius: 16px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 28px; font-weight: 900;">Voice Response</h3>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                    <p style="font-size: 16px; color: white; margin: 0; line-height: 1.6;">
                        "Hello {patient_name}! I've checked your Medicare eligibility. You are <strong>{'eligible' if is_eligible else 'not yet eligible'}</strong> for Medicare (age {age}). 
                        {'You have Medicare Part A and Part B active. Your coverage includes hospital care and doctor visits.' if is_eligible and has_part_a and has_part_b else 'Please enroll in Medicare Parts A and B for full coverage.'}
                        How can I help you today?"
                    </p>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Status</p>
                        <p style="font-size: 20px; color: white; font-weight: 700; margin: 0;">{'✅ Eligible' if is_eligible else '❌ Not Eligible'}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Age</p>
                        <p style="font-size: 20px; color: white; font-weight: 700; margin: 0;">{age} years</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Part A</p>
                        <p style="font-size: 20px; color: white; font-weight: 700; margin: 0;">{'✅ Active' if has_part_a else '❌ Inactive'}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Part B</p>
                        <p style="font-size: 20px; color: white; font-weight: 700; margin: 0;">{'✅ Active' if has_part_b else '❌ Inactive'}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Response time metrics
            st.markdown("**Performance Metrics**")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            metrics_col1.metric("Response Time", "1.2s", "-85%")
            metrics_col2.metric("Accuracy", "98.5%", "+12%")
            metrics_col3.metric("Patient Satisfaction", "4.8/5", "+0.9")

with tab2:
    st.markdown("### Agency Performance Dashboard")
    
    # Generate metrics
    calls_today = 847
    calls_handled = 812
    avg_call_time = 2.3
    eligibility_checks = 623
    appointments_scheduled = 189
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Calls Today", f"{calls_today}", "+23%")
    col2.metric("AI Handled", f"{calls_handled}", "96% auto")
    col3.metric("Avg Call Time", f"{avg_call_time} min", "-70%")
    col4.metric("Satisfaction", "4.8/5", "+0.9")
    
    # Call volume chart
    st.markdown("**Call Volume (Last 7 Days)**")
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    volumes = [780, 820, 790, 850, 870, 420, 380]
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=days,
        y=volumes,
        marker=dict(color='#3b82f6'),
        text=volumes,
        textposition='auto'
    ))
    fig1.update_layout(
        title='Daily Call Volume',
        xaxis_title='Day',
        yaxis_title='Calls',
        height=300
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Task breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Task Distribution**")
        tasks = ['Eligibility', 'Appointments', 'Prescriptions', 'Provider Search', 'Other']
        task_counts = [623, 189, 156, 98, 81]
        
        fig2 = go.Figure(data=[go.Pie(
            labels=tasks,
            values=task_counts,
            hole=0.4,
            marker=dict(colors=['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'])
        )])
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("**Monthly Impact**")
        
        impact_data = {
            'Metric': ['Calls Handled', 'Hours Saved', 'Cost Reduction', 'Patient Reach'],
            'Value': ['18,450', '1,240 hrs', '$186K', '12,300']
        }
        st.dataframe(pd.DataFrame(impact_data), hide_index=True, use_container_width=True)
        
        st.markdown("**ROI Analysis**")
        st.metric("Annual Savings", "$2.2M", "+340%")
        st.metric("Staff Efficiency", "+65%", "More patient time")

with tab3:
    st.markdown("### Voice AI Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Automated Tasks**")
        st.markdown("""
        - ✅ Medicare eligibility verification
        - ✅ Plan coverage explanation
        - ✅ Provider directory search
        - ✅ Appointment scheduling
        - ✅ Prescription refill requests
        - ✅ Claims status inquiry
        - ✅ Benefit utilization tracking
        - ✅ Prior authorization support
        """)
    
    with col2:
        st.markdown("**Medicare Plans Supported**")
        plans_df = pd.DataFrame([
            {'Plan': 'Part A', 'Premium': '$0', 'Coverage': 'Hospital'},
            {'Plan': 'Part B', 'Premium': '$174.70', 'Coverage': 'Medical'},
            {'Plan': 'Part D', 'Premium': '$55', 'Coverage': 'Rx'},
            {'Plan': 'Advantage', 'Premium': '$25', 'Coverage': 'All-in-One'},
            {'Plan': 'Medigap G', 'Premium': '$150', 'Coverage': 'Supplement'}
        ])
        st.dataframe(plans_df, hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #1e40af; font-size: 24px; font-weight: 800;">💡 System Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #2563eb; font-weight: 700; margin: 0 0 6px 0;">✓ 24/7 Availability</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Never miss a patient call</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #2563eb; font-weight: 700; margin: 0 0 6px 0;">✓ 96% Automation</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">AI handles most inquiries</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #2563eb; font-weight: 700; margin: 0 0 6px 0;">✓ 2.3 Min Avg Call</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">70% faster than manual</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #2563eb; font-weight: 700; margin: 0 0 6px 0;">✓ $2.2M Annual Savings</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">340% ROI per agency</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #3b82f6 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for careCycle</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)