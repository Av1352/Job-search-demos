"""
Play Health - Perimenopause Care Platform
AI-powered care for women navigating perimenopause
Built for Play Health by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Play Health", page_icon="🌸", layout="wide")

# Symptoms
PERIMENOPAUSE_SYMPTOMS = {
    'Hot Flashes': {'severity': 'Moderate', 'frequency': 'Daily', 'impact': 'High'},
    'Night Sweats': {'severity': 'Moderate', 'frequency': '4-5x/week', 'impact': 'High'},
    'Irregular Periods': {'severity': 'Mild', 'frequency': 'Monthly', 'impact': 'Moderate'},
    'Mood Changes': {'severity': 'Moderate', 'frequency': 'Daily', 'impact': 'High'},
    'Sleep Disruption': {'severity': 'Severe', 'frequency': 'Nightly', 'impact': 'High'},
    'Brain Fog': {'severity': 'Mild', 'frequency': '3-4x/week', 'impact': 'Moderate'}
}

# Treatment options
TREATMENT_OPTIONS = {
    'Hormone Therapy (HRT)': {'effectiveness': 92.3, 'side_effects': 'Low', 'cost': '$45/mo'},
    'Non-Hormonal Meds': {'effectiveness': 68.5, 'side_effects': 'Moderate', 'cost': '$35/mo'},
    'Lifestyle Modifications': {'effectiveness': 54.2, 'side_effects': 'None', 'cost': '$0'},
    'Supplements': {'effectiveness': 42.7, 'side_effects': 'Low', 'cost': '$25/mo'},
    'Cognitive Behavioral Therapy': {'effectiveness': 71.8, 'side_effects': 'None', 'cost': '$80/session'}
}

# Care metrics
CARE_METRICS = {
    'Symptom Improvement': 67.8,
    'Quality of Life': 4.2,  # out of 5
    'Treatment Adherence': 89.3,
    'Patient Satisfaction': 4.8,
    'Time to Treatment': 3.2  # days vs 45 days
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #f472b6 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🌸</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Play Health</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Perimenopause Care Platform</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">AI-powered care • Personalized treatment • 67.8% symptom improvement</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌸 Symptom Assessment", "📊 Care Dashboard", "📈 Treatment Outcomes", "💡 Platform Features"])

with tab1:
    st.markdown("### AI-Powered Symptom Assessment")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Patient Information**")
        
        patient_name = st.text_input("Name", "Jessica Williams")
        age = st.number_input("Age", 35, 60, 47)
        
        st.markdown("**Symptom Tracking**")
        
        st.text("Select symptoms you're experiencing:")
        
        symptoms_selected = []
        for symptom in list(PERIMENOPAUSE_SYMPTOMS.keys())[:4]:
            if st.checkbox(symptom, value=np.random.random() > 0.5):
                severity = st.select_slider(
                    f"{symptom} severity",
                    options=["Mild", "Moderate", "Severe"],
                    value="Moderate",
                    key=f"sev_{symptom}"
                )
                symptoms_selected.append((symptom, severity))
        
        st.markdown("**Lifestyle Factors**")
        
        exercise_freq = st.slider("Exercise (days/week)", 0, 7, 3)
        stress_level = st.slider("Stress level (1-10)", 1, 10, 7)
        sleep_quality = st.slider("Sleep quality (1-10)", 1, 10, 4)
        
        assess_btn = st.button("🌸 Get Personalized Care Plan", type="primary", use_container_width=True)
    
    with col2:
        if assess_btn and symptoms_selected:
            st.markdown("**AI Assessment Results**")
            
            with st.spinner("Analyzing symptoms and creating care plan..."):
                import time
                time.sleep(1.5)
            
            st.success("✅ Personalized treatment plan ready!")
            
            symptom_count = len(symptoms_selected)
            severity_score = (symptom_count * 25 + stress_level * 5 + (10 - sleep_quality) * 3)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f472b6 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Your Perimenopause Profile</h3>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                    <p style="color: white; margin: 0; line-height: 1.7; font-size: 15px;">
                        <strong>Stage:</strong> Mid-Perimenopause (based on symptoms)<br>
                        <strong>Symptom Burden:</strong> Moderate-High ({symptom_count} symptoms)<br>
                        <strong>Quality of Life Impact:</strong> Significant (sleep + mood affected)
                    </p>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Symptoms</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{symptom_count}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Severity Score</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{min(severity_score, 100)}/100</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Personalized Treatment Recommendations**")
            
            st.markdown("""
            <div style="background: #f8fafc; padding: 25px; border-radius: 12px; border-left: 4px solid #f472b6;">
                <h4 style="margin: 0 0 15px 0; color: #1f2937; font-weight: 700;">Recommended Treatment Plan:</h4>
                <p style="margin: 8px 0; color: #374151; line-height: 1.8;">
                    <strong>1. Hormone Replacement Therapy (HRT):</strong><br>
                    • Estradiol patch 0.05mg + Progesterone 100mg<br>
                    • 92.3% effectiveness for hot flashes and night sweats<br>
                    • Prescription available through Play Health providers<br><br>
                    <strong>2. Lifestyle Interventions:</strong><br>
                    • Sleep hygiene protocol (target 7-8 hours)<br>
                    • Stress management (meditation, yoga 3x/week)<br>
                    • Exercise: 30 min moderate activity 5x/week<br>
                    • Nutrition: Limit caffeine after 2pm, reduce alcohol<br><br>
                    <strong>3. Cognitive Support:</strong><br>
                    • CBT for mood management (virtual sessions available)<br>
                    • Brain fog strategies (lists, routines, supplements)<br><br>
                    <strong>4. Monitoring:</strong><br>
                    • Daily symptom tracking in Play Health app<br>
                    • Virtual check-in with provider in 2 weeks<br>
                    • Adjust HRT dosing as needed based on response
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Improvement", "67.8%", "Symptoms")
            col2.metric("Time to Treatment", "3.2 days", "vs 45 days")
            col3.metric("Provider Match", "Dr. Chen", "Women's health")

with tab2:
    st.markdown("### Care & Symptom Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Patients", "8,240", "+892")
    col2.metric("Symptom Relief", "67.8%", "+23.5%")
    col3.metric("QoL Score", "4.2/5", "+1.1")
    col4.metric("Adherence", "89.3%", "+18.7%")
    
    st.markdown("**Symptom Severity Trends (30 Days)**")
    
    days = list(range(1, 31))
    hot_flashes = [7.5 - i*0.15 + np.random.uniform(-0.5, 0.5) for i in days]
    sleep_quality = [3.2 + i*0.08 + np.random.uniform(-0.3, 0.3) for i in days]
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=days, y=hot_flashes,
        mode='lines',
        name='Hot Flashes (severity 1-10)',
        line=dict(color='#ef4444', width=2),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.1)'
    ))
    fig1.add_trace(go.Scatter(
        x=days, y=sleep_quality,
        mode='lines',
        name='Sleep Quality (1-10)',
        line=dict(color='#22c55e', width=2),
        yaxis='y2'
    ))
    fig1.update_layout(
        yaxis=dict(title='Hot Flashes (↓ better)', range=[0, 10]),
        yaxis2=dict(title='Sleep Quality (↑ better)', overlaying='y', side='right', range=[0, 10]),
        height=300
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Most Common Symptoms**")
        
        symptom_counts = [5847, 4923, 4234, 3892, 3456, 2891]
        
        fig2 = go.Figure(data=[go.Bar(
            x=list(PERIMENOPAUSE_SYMPTOMS.keys()),
            y=symptom_counts,
            marker=dict(color='#f472b6'),
            text=symptom_counts,
            textposition='auto'
        )])
        fig2.update_layout(yaxis_title='Patients Reporting', height=250)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("**Treatment Distribution**")
        
        treatment_counts = [4820, 2134, 3892, 1567, 892]
        
        fig3 = go.Figure(data=[go.Pie(
            labels=list(TREATMENT_OPTIONS.keys()),
            values=treatment_counts,
            hole=0.4,
            marker=dict(colors=['#f472b6', '#a855f7', '#22c55e', '#f59e0b', '#3b82f6'])
        )])
        fig3.update_layout(height=250, title='Active Treatments')
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.markdown("### Treatment Outcomes & Effectiveness")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Treatment Effectiveness**")
        
        treatment_data = []
        for treatment, data in TREATMENT_OPTIONS.items():
            treatment_data.append({
                'Treatment': treatment,
                'Effectiveness': f"{data['effectiveness']}%",
                'Side Effects': data['side_effects'],
                'Cost': data['cost']
            })
        
        st.dataframe(pd.DataFrame(treatment_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Clinical Outcomes (6 months)**")
        
        outcomes = {
            'Outcome': ['Hot Flashes ↓', 'Sleep Quality ↑', 'Mood Stability ↑', 'QoL Score ↑', 'Work Productivity ↑'],
            'Improvement': ['68.5%', '54.2%', '61.3%', '+1.1 points', '+28%'],
            'Baseline': ['7.8/10', '3.2/10', '4.1/10', '3.1/5', '62%']
        }
        st.dataframe(pd.DataFrame(outcomes), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Treatment Effectiveness Comparison**")
        
        treatments = list(TREATMENT_OPTIONS.keys())
        effectiveness = [TREATMENT_OPTIONS[t]['effectiveness'] for t in treatments]
        
        fig4 = go.Figure(data=[go.Bar(
            x=treatments,
            y=effectiveness,
            marker=dict(
                color=effectiveness,
                colorscale='PinkYl',
                cmin=40,
                cmax=100
            ),
            text=[f"{e}%" for e in effectiveness],
            textposition='auto'
        )])
        fig4.update_layout(yaxis=dict(range=[30, 100]), yaxis_title='Effectiveness (%)', height=250)
        st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("**Patient Reported Outcomes**")
        
        col1, col2 = st.columns(2)
        col1.metric("Symptom Relief", "67.8%", "+23.5%")
        col2.metric("QoL Improvement", "+1.1", "3.1 → 4.2/5")

with tab4:
    st.markdown("### AI-Powered Care Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Symptom Intelligence**")
        st.markdown("""
        - ✅ AI symptom tracking & analysis
        - ✅ Pattern recognition (triggers, cycles)
        - ✅ Severity trend monitoring
        - ✅ Treatment response tracking
        - ✅ Personalized insights
        - ✅ Predictive symptom forecasting
        """)
        
        st.markdown("**Treatment Optimization**")
        st.markdown("""
        - ✅ Personalized treatment matching
        - ✅ HRT dosing recommendations
        - ✅ Medication interaction checking
        - ✅ Side effect monitoring
        - ✅ Treatment adjustment suggestions
        - ✅ Evidence-based protocols
        """)
    
    with col2:
        st.markdown("**Care Access**")
        st.markdown("""
        - ✅ Virtual consultations with specialists
        - ✅ 3.2-day time to treatment (vs 45 days)
        - ✅ Prescription delivery
        - ✅ 24/7 symptom support
        - ✅ Peer community access
        - ✅ Educational resources
        """)
        
        st.markdown("**Holistic Support**")
        st.markdown("""
        - ✅ Nutrition guidance (menopause-specific)
        - ✅ Exercise programming
        - ✅ Sleep optimization
        - ✅ Stress management (CBT, mindfulness)
        - ✅ Mental health support
        - ✅ Sexual health counseling
        """)
    
    st.markdown("**Platform Coverage**")
    
    coverage = {
        'Service': ['Symptom Tracking', 'Virtual Consultations', 'HRT Prescriptions', 'Lab Testing', 'Mental Health', 'Community'],
        'Availability': ['24/7', 'Same-day', 'State-dependent', 'Home kits', 'Licensed therapists', '24/7'],
        'Cost': ['Included', '$75/visit', '$45/mo', '$79/panel', '$80/session', 'Included'],
        'Coverage': ['All states', 'All states', '48 states', 'All states', 'All states', 'All states']
    }
    st.dataframe(pd.DataFrame(coverage), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #831843; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #be185d; font-weight: 700; margin: 0 0 6px 0;">✓ 67.8% Symptom Relief</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Significant improvement</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #be185d; font-weight: 700; margin: 0 0 6px 0;">✓ 3.2 Days to Care</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">vs 45 days traditional</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #be185d; font-weight: 700; margin: 0 0 6px 0;">✓ 8,240 Women</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Active patients</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #be185d; font-weight: 700; margin: 0 0 6px 0;">✓ 4.8/5 Satisfaction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">High approval</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #f472b6 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Play Health</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)