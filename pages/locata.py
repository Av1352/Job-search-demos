"""
Locata - Referral Management Platform
AI-powered referral coordination and tracking
Built for Locata by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Locata", page_icon="🔄", layout="wide")

# Specialty types
SPECIALTIES = {
    'Cardiology': {'providers': 234, 'avg_wait': '12 days', 'acceptance': 89.5},
    'Orthopedics': {'providers': 189, 'avg_wait': '8 days', 'acceptance': 92.3},
    'Neurology': {'providers': 156, 'avg_wait': '18 days', 'acceptance': 85.7},
    'Gastroenterology': {'providers': 167, 'avg_wait': '14 days', 'acceptance': 87.2},
    'Dermatology': {'providers': 298, 'avg_wait': '6 days', 'acceptance': 94.8}
}

# Referral status
REFERRAL_STAGES = ['Submitted', 'Pending Review', 'Scheduled', 'Completed', 'Cancelled']

# Automation metrics
AUTOMATION_METRICS = {
    'Auto-Matching Rate': 96.8,
    'Referral Completion': 89.3,
    'Time to Appointment': 8.5,  # days
    'Provider Acceptance': 90.2,
    'Patient Show Rate': 87.5
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #f59e0b 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🔄</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Locata</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Referral Management Platform</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">AI-powered coordination • 96.8% auto-matching • 8.5-day appointments</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔄 Create Referral", "📊 Referral Dashboard", "📈 Performance", "💡 Features"])

with tab1:
    st.markdown("### AI-Powered Referral Coordination")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Referral Information**")
        
        patient_name = st.text_input("Patient Name", "Jennifer Lee")
        referring_md = st.text_input("Referring Physician", "Dr. Robert Smith")
        
        st.markdown("**Specialty & Urgency**")
        
        specialty = st.selectbox("Specialty Needed", list(SPECIALTIES.keys()))
        urgency = st.selectbox("Urgency", ["Routine", "Urgent", "STAT"])
        
        st.markdown("**Clinical Details**")
        
        diagnosis = st.text_input("Diagnosis", "Chest pain, r/o CAD")
        reason = st.text_area(
            "Reason for Referral",
            "52yo M with exertional chest pain x3 weeks. Abnormal stress test. Needs cardiology evaluation for possible CAD.",
            height=80
        )
        
        st.markdown("**Patient Preferences**")
        
        insurance = st.selectbox("Insurance", ["UnitedHealthcare", "Aetna", "BCBS", "Cigna"])
        max_distance = st.slider("Max Distance (miles)", 5, 50, 15)
        preferred_days = st.multiselect("Preferred Days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], default=["Tuesday", "Thursday"])
        
        match_btn = st.button("🔄 Find Specialist", type="primary", use_container_width=True)
    
    with col2:
        if match_btn:
            st.markdown("**AI Matching Results**")
            
            with st.spinner("Analyzing network and availability..."):
                import time
                time.sleep(1.5)
            
            st.success("✅ Found 3 optimal matches!")
            
            specialty_data = SPECIALTIES[specialty]
            
            # Top matches
            matches = [
                {'name': 'Dr. Emily Chen, MD', 'distance': '2.3 mi', 'wait': '5 days', 'rating': '4.9/5', 'score': 98},
                {'name': 'Dr. Michael Park, MD', 'distance': '8.7 mi', 'wait': '3 days', 'rating': '4.8/5', 'score': 96},
                {'name': 'Dr. Sarah Williams, MD', 'distance': '12.1 mi', 'wait': '7 days', 'rating': '4.7/5', 'score': 92}
            ]
            
            for i, match in enumerate(matches, 1):
                st.markdown(f"""
                <div style="background: white; border-left: 4px solid {'#10b981' if i == 1 else '#3b82f6' if i == 2 else '#f59e0b'}; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <h4 style="margin: 0; color: #1f2937; font-size: 18px; font-weight: 700;">#{i} {match['name']}</h4>
                        <span style="background: #f59e0b; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600;">Match: {match['score']}%</span>
                    </div>
                    <p style="margin: 5px 0; color: #6b7280; font-size: 14px;">📍 {match['distance']} away • 📅 Next available: {match['wait']} • ⭐ {match['rating']}</p>
                    <p style="margin: 8px 0 0 0; color: #6b7280; font-size: 13px;">✓ In-network • ✓ Accepts {insurance} • ✓ {specialty} specialist</p>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Match", "98%", "Dr. Chen")
            col2.metric("Soonest Appt", "3 days", "Dr. Park")
            col3.metric("Closest", "2.3 mi", "Dr. Chen")
            
            if st.button("📅 Schedule with Dr. Chen", use_container_width=True):
                st.success("✅ Appointment scheduled for Feb 7th @ 2:30 PM")

with tab2:
    st.markdown("### Referral Operations Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Referrals", "1,847", "+234")
    col2.metric("Completion Rate", "89.3%", "+3.2%")
    col3.metric("Avg Time to Appt", "8.5 days", "-6.5 days")
    col4.metric("Provider Acceptance", "90.2%", "+5.1%")
    
    st.markdown("**Referral Volume by Specialty**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        specialty_counts = [423, 389, 312, 356, 367]
        
        fig1 = go.Figure(data=[go.Pie(
            labels=list(SPECIALTIES.keys()),
            values=specialty_counts,
            hole=0.4,
            marker=dict(colors=['#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899'])
        )])
        fig1.update_layout(height=300, title='Specialty Distribution')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = go.Figure(data=[go.Bar(
            x=list(SPECIALTIES.keys()),
            y=specialty_counts,
            marker=dict(color='#f59e0b'),
            text=specialty_counts,
            textposition='auto'
        )])
        fig2.update_layout(yaxis_title='Referrals', height=300)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("**Referral Status Pipeline**")
    
    status_counts = [423, 312, 789, 234, 89]
    
    fig3 = go.Figure(data=[go.Funnel(
        y=REFERRAL_STAGES,
        x=status_counts,
        textinfo="value+percent initial",
        marker=dict(color=['#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ef4444'])
    )])
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.markdown("### System Performance Metrics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Automation Performance**")
        
        metrics_data = []
        for metric, value in AUTOMATION_METRICS.items():
            if 'Rate' in metric or 'Completion' in metric or 'Acceptance' in metric:
                display = f"{value}%"
            else:
                display = f"{value} days"
            
            metrics_data.append({
                'Metric': metric,
                'Value': display,
                'Status': '✅ Excellent'
            })
        
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
        
        st.markdown("**vs Manual Referrals**")
        
        comparison = {
            'Metric': ['Time to Appointment', 'Completion Rate', 'Provider Acceptance', 'Patient Show Rate'],
            'Locata AI': ['8.5 days', '89.3%', '90.2%', '87.5%'],
            'Manual': ['15 days', '68.5%', '75.3%', '72.8%'],
            'Improvement': ['-6.5 days', '+20.8%', '+14.9%', '+14.7%']
        }
        st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Completion Rate by Specialty**")
        
        specialties = list(SPECIALTIES.keys())
        completion = [92.3, 91.5, 85.7, 87.8, 94.2]
        
        fig4 = go.Figure(data=[go.Bar(
            x=specialties,
            y=completion,
            marker=dict(
                color=completion,
                colorscale='YlOrRd',
                cmin=80,
                cmax=100
            ),
            text=[f"{c}%" for c in completion],
            textposition='auto'
        )])
        fig4.update_layout(yaxis=dict(range=[80, 100]), height=250)
        st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("**Monthly Impact**")
        
        impact = {
            'Metric': ['Referrals Coordinated', 'Successful Appointments', 'Time Saved', 'Leakage Prevented'],
            'Value': ['5,847', '5,220', '1,456 hrs', '$890K']
        }
        st.dataframe(pd.DataFrame(impact), hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### AI Matching & Coordination")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Smart Matching Algorithm**")
        st.markdown("""
        - ✅ Insurance network validation
        - ✅ Distance/location optimization
        - ✅ Availability analysis (real-time)
        - ✅ Provider specialization matching
        - ✅ Patient preference alignment
        - ✅ Urgency-based prioritization
        - ✅ Quality metrics (ratings, outcomes)
        - ✅ Cultural/language matching
        """)
        
        st.markdown("**Workflow Automation**")
        st.markdown("""
        - ✅ Auto-schedule appointments
        - ✅ Send patient notifications (SMS/email)
        - ✅ Fax clinical notes to specialist
        - ✅ Track referral status
        - ✅ Follow-up reminders
        - ✅ No-show prevention
        """)
    
    with col2:
        st.markdown("**Integration**")
        st.markdown("""
        - ✅ EHR integration (Epic, Cerner)
        - ✅ Specialist scheduling systems
        - ✅ Insurance eligibility APIs
        - ✅ Patient engagement platforms
        - ✅ Analytics dashboards
        - ✅ Reporting tools
        """)
        
        st.markdown("**Analytics & Insights**")
        st.markdown("""
        - ✅ Referral leakage tracking
        - ✅ Network gap identification
        - ✅ Provider performance metrics
        - ✅ Patient journey mapping
        - ✅ Revenue optimization
        - ✅ Quality outcome tracking
        """)
    
    st.markdown("**Provider Network Coverage**")
    
    network_data = {
        'Specialty': list(SPECIALTIES.keys()),
        'Providers': [SPECIALTIES[s]['providers'] for s in SPECIALTIES.keys()],
        'Avg Wait Time': [SPECIALTIES[s]['avg_wait'] for s in SPECIALTIES.keys()],
        'Acceptance Rate': [f"{SPECIALTIES[s]['acceptance']}%" for s in SPECIALTIES.keys()]
    }
    st.dataframe(pd.DataFrame(network_data), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #78350f; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #d97706; font-weight: 700; margin: 0 0 6px 0;">✓ 96.8% Auto-Match</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Intelligent provider matching</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #d97706; font-weight: 700; margin: 0 0 6px 0;">✓ 8.5 Days to Appt</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">vs 15 days manual</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #d97706; font-weight: 700; margin: 0 0 6px 0;">✓ 89.3% Completion</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">vs 68.5% manual</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #d97706; font-weight: 700; margin: 0 0 6px 0;">✓ $890K Saved</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Monthly leakage prevention</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #f59e0b 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Locata</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)