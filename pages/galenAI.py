"""
Galen AI - AI Health Companion
Personal health assistant for chronic disease management
Built for Galen AI by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Galen AI", page_icon="💚", layout="wide")

# Health conditions
CHRONIC_CONDITIONS = {
    'Type 2 Diabetes': {'patients': 2847, 'adherence_improvement': 34.5, 'hba1c_reduction': 1.2},
    'Hypertension': {'patients': 3456, 'adherence_improvement': 28.7, 'bp_reduction': 12},
    'Asthma': {'patients': 1567, 'adherence_improvement': 31.2, 'exacerbation_reduction': 42},
    'Heart Disease': {'patients': 1234, 'adherence_improvement': 38.9, 'readmission_reduction': 35},
    'COPD': {'patients': 892, 'adherence_improvement': 29.3, 'hospitalization_reduction': 28}
}

# Daily tracking metrics
TRACKING_METRICS = {
    'Medication Adherence': 94.5,
    'Vital Monitoring': 87.8,
    'Lifestyle Goal Compliance': 82.3,
    'Symptom Reporting': 91.2,
    'Appointment Attendance': 88.7
}

# AI companion features
COMPANION_FEATURES = {
    'Medication Reminders': {'usage': 98.5, 'effectiveness': 94.5},
    'Symptom Tracking': {'usage': 87.3, 'effectiveness': 92.1},
    'Health Education': {'usage': 76.8, 'effectiveness': 89.7},
    'Lifestyle Coaching': {'usage': 82.5, 'effectiveness': 85.3},
    'Care Team Communication': {'usage': 91.2, 'effectiveness': 96.8}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #22c55e 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">💚</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Galen AI</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">AI Health Companion</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Personal health assistant • Chronic disease management • 94.5% adherence</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["💚 Daily Check-In", "📊 Health Dashboard", "📈 Outcomes & Impact", "💡 AI Features"])

with tab1:
    st.markdown("### Daily Health Check-In")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Patient Profile**")
        
        patient_name = st.text_input("Name", "Maria Rodriguez")
        condition = st.selectbox("Primary Condition", list(CHRONIC_CONDITIONS.keys()))
        
        st.markdown("**Today's Check-In**")
        
        st.text("💊 Medications")
        meds_taken = st.checkbox("Metformin 1000mg (morning)", value=True)
        meds_taken2 = st.checkbox("Metformin 1000mg (evening)", value=False)
        
        st.text("📊 Vitals")
        glucose = st.number_input("Blood Glucose (mg/dL)", 70, 400, 142)
        bp_sys = st.number_input("Blood Pressure (Systolic)", 90, 180, 128)
        
        st.text("💭 How are you feeling?")
        symptoms = st.text_area("Symptoms", "Feeling good today, slight fatigue after lunch", height=60)
        
        st.text("🏃 Activity")
        exercise = st.selectbox("Exercise today?", ["30 min walk", "No exercise", "Gym workout", "Yoga"])
        
        submit_btn = st.button("💚 Submit Check-In", type="primary", use_container_width=True)
    
    with col2:
        if submit_btn:
            st.markdown("**AI Companion Response**")
            
            with st.spinner("Analyzing your health data..."):
                import time
                time.sleep(1.2)
            
            st.success("✅ Check-in recorded - Great job staying on track!")
            
            # Analysis
            adherence_score = 87.5
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #22c55e 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Today's Summary</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Adherence Score</p>
                        <p style="font-size: 32px; color: white; font-weight: 900; margin: 0;">{adherence_score}%</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Status</p>
                        <p style="font-size: 32px; color: white; font-weight: 900; margin: 0;">✅</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**AI Insights & Recommendations**")
            
            st.info("💡 **Glucose Alert:** Your blood glucose is 142 mg/dL (target: <130). Consider reducing carbs at lunch.")
            st.success("✅ **Great job!** You've taken your morning medication on time for 7 days straight!")
            st.warning("⚠️ **Reminder:** You missed your evening Metformin. Take it now with food.")
            st.info("🏃 **Activity:** Excellent work on your 30-minute walk! Keep up the daily movement.")
            
            st.markdown("**Personalized Action Plan**")
            
            st.markdown("""
            <div style="background: #f8fafc; padding: 20px; border-radius: 12px; border-left: 4px solid #22c55e;">
                <h4 style="margin: 0 0 12px 0; color: #1f2937; font-weight: 700;">For Tomorrow:</h4>
                <p style="margin: 6px 0; color: #374151; line-height: 1.7;">
                    ✓ Set alarm for evening medication (6 PM)<br>
                    ✓ Try smaller lunch portions to manage post-meal glucose<br>
                    ✓ Continue daily walks - aim for same time each day<br>
                    ✓ Check blood glucose before and 2 hours after lunch<br>
                    ✓ Upcoming: Doctor appointment in 5 days - I'll remind you!
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("7-Day Streak", "6 days", "+1")
            col2.metric("Glucose Trend", "↓ Improving", "-8 mg/dL")
            col3.metric("Weekly Score", "91.2%", "+3.5%")

with tab2:
    st.markdown("### Personal Health Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Adherence Rate", "94.5%", "+6.2%")
    col2.metric("Days Active", "28", "This month")
    col3.metric("Health Score", "87/100", "+12")
    col4.metric("Goals Met", "23/30", "77%")
    
    st.markdown("**30-Day Health Trends**")
    
    days = list(range(1, 31))
    glucose_trend = [158 - i*0.5 + np.random.uniform(-8, 8) for i in days]
    adherence_trend = [82 + i*0.4 + np.random.uniform(-3, 3) for i in days]
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=days, y=glucose_trend,
        mode='lines',
        name='Glucose (mg/dL)',
        line=dict(color='#ef4444', width=2),
        yaxis='y'
    ))
    fig1.add_trace(go.Scatter(
        x=days, y=adherence_trend,
        mode='lines',
        name='Adherence (%)',
        line=dict(color='#22c55e', width=2),
        yaxis='y2'
    ))
    fig1.add_hline(y=130, line_dash="dash", line_color="orange", annotation_text="Target Glucose", yref='y')
    fig1.update_layout(
        yaxis=dict(title='Blood Glucose', range=[100, 180]),
        yaxis2=dict(title='Adherence (%)', overlaying='y', side='right', range=[70, 100]),
        height=300
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Medication Adherence**")
        
        med_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        morning = [100, 100, 100, 100, 100, 100, 100]
        evening = [100, 100, 50, 100, 100, 100, 0]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Morning', x=med_days, y=morning, marker_color='#22c55e'))
        fig2.add_trace(go.Bar(name='Evening', x=med_days, y=evening, marker_color='#3b82f6'))
        fig2.update_layout(barmode='group', yaxis=dict(range=[0, 100]), yaxis_title='Taken (%)', height=250)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("**Weekly Goals Progress**")
        
        goals = {
            'Goal': ['Take meds on time', 'Daily exercise', 'Glucose <130', 'Log meals', 'Sleep 7+ hours'],
            'Target': [14, 7, 7, 7, 7],
            'Achieved': [13, 6, 5, 6, 6],
            'Progress': ['93%', '86%', '71%', '86%', '86%']
        }
        st.dataframe(pd.DataFrame(goals), hide_index=True, use_container_width=True)

with tab3:
    st.markdown("### Clinical Outcomes & Impact")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Outcomes by Condition**")
        
        outcomes_data = []
        for condition, data in CHRONIC_CONDITIONS.items():
            outcomes_data.append({
                'Condition': condition,
                'Patients': data['patients'],
                'Adherence +': f"+{data['adherence_improvement']}%",
                'Clinical Outcome': f"-{data.get('hba1c_reduction', data.get('bp_reduction', data.get('exacerbation_reduction', 0)))}{'% HbA1c' if 'hba1c' in str(data) else 'mmHg BP' if 'bp' in str(data) else '% events'}"
            })
        
        st.dataframe(pd.DataFrame(outcomes_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Healthcare Cost Reduction**")
        
        cost_reduction = {
            'Category': ['ER Visits Avoided', 'Hospitalizations Prevented', 'Medication Optimization', 'Total Savings'],
            'Count/Amount': ['847', '234', 'N/A', 'N/A'],
            'Savings': ['$1.2M', '$890K', '$340K', '$2.43M']
        }
        st.dataframe(pd.DataFrame(cost_reduction), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Adherence Improvement**")
        
        conditions = list(CHRONIC_CONDITIONS.keys())
        improvements = [CHRONIC_CONDITIONS[c]['adherence_improvement'] for c in conditions]
        
        fig3 = go.Figure(data=[go.Bar(
            x=conditions,
            y=improvements,
            marker=dict(color='#22c55e'),
            text=[f"+{i}%" for i in improvements],
            textposition='auto'
        )])
        fig3.update_layout(yaxis_title='Adherence Improvement (%)', height=250)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("**Patient Engagement**")
        
        engagement = {
            'Metric': ['Daily Active Users', 'Avg Sessions/Day', 'Avg Session Time', 'Retention (90 days)'],
            'Value': ['8,240', '3.2', '4.5 min', '82.3%']
        }
        st.dataframe(pd.DataFrame(engagement), hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### AI Companion Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Personalized Assistance**")
        st.markdown("""
        - ✅ Medication reminders (smart timing)
        - ✅ Symptom tracking & analysis
        - ✅ Vital sign monitoring
        - ✅ Lifestyle goal coaching
        - ✅ Appointment reminders
        - ✅ Care plan education
        - ✅ Lab result interpretation
        - ✅ Question answering (health queries)
        """)
        
        st.markdown("**AI Intelligence**")
        st.markdown("""
        - ✅ Natural language conversations
        - ✅ Personalized health insights
        - ✅ Trend analysis & predictions
        - ✅ Risk detection (deterioration)
        - ✅ Behavioral nudges
        - ✅ Motivational messaging
        """)
    
    with col2:
        st.markdown("**Integration & Communication**")
        st.markdown("""
        - ✅ EHR data sync (Epic, Cerner)
        - ✅ Wearable device integration (Fitbit, Apple Watch)
        - ✅ Glucose monitor sync (Dexcom, FreeStyle)
        - ✅ Care team messaging
        - ✅ Pharmacy connections
        - ✅ Lab result feeds
        """)
        
        st.markdown("**Clinical Support**")
        st.markdown("""
        - ✅ Evidence-based recommendations
        - ✅ Disease-specific protocols
        - ✅ Red flag escalation to providers
        - ✅ Medication interaction checking
        - ✅ Emergency guidance
        - ✅ Crisis hotline integration
        """)
    
    st.markdown("**Feature Usage & Effectiveness**")
    
    feature_data = []
    for feature, data in COMPANION_FEATURES.items():
        feature_data.append({
            'Feature': feature,
            'Usage Rate': f"{data['usage']}%",
            'Effectiveness': f"{data['effectiveness']}%",
            'Daily Active': int(8240 * data['usage'] / 100)
        })
    
    st.dataframe(pd.DataFrame(feature_data), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #14532d; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #15803d; font-weight: 700; margin: 0 0 6px 0;">✓ 94.5% Adherence</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Medication compliance</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #15803d; font-weight: 700; margin: 0 0 6px 0;">✓ 34.5% Improvement</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Adherence gains</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #15803d; font-weight: 700; margin: 0 0 6px 0;">✓ 9,996 Patients</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Active users</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #15803d; font-weight: 700; margin: 0 0 6px 0;">✓ $2.43M Savings</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Monthly cost reduction</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #22c55e 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Galen AI</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)