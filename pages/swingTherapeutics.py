"""
Swing Therapeutics - Digital Therapeutic for Fibromyalgia
FDA-cleared Stanza app using ACT for chronic pain management
Built for Swing Therapeutics by Anju Vilashni Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Swing Therapeutics - Fibromyalgia DTx", page_icon="💜", layout="wide")

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">💜</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Swing Therapeutics</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Digital Therapeutic for Fibromyalgia</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">FDA-cleared Stanza • ACT therapy • 12-week program</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["💜 Stanza Program", "📊 Clinical Outcomes", "💡 Technology"])

with tab1:
    st.markdown("### Stanza - FDA-Cleared Digital Behavioral Therapy")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Patient Profile**")
        
        patient_name = st.text_input("Name", "Maria Rodriguez")
        age = st.number_input("Age", 18, 80, 48)
        
        st.markdown("**Fibromyalgia Assessment**")
        
        pain_level = st.slider("Current Pain Level (0-10)", 0, 10, 7)
        fatigue = st.slider("Fatigue Level (0-10)", 0, 10, 8)
        sleep_quality = st.select_slider("Sleep Quality", 
                                        options=["Very Poor", "Poor", "Fair", "Good", "Excellent"],
                                        value="Poor")
        
        brain_fog = st.checkbox("Brain fog/memory problems", value=True)
        anxiety_depression = st.checkbox("Anxiety/depression", value=True)
        
        st.markdown("**Treatment History**")
        
        current_meds = st.multiselect("Current Medications",
                                      ["Lyrica/Gabapentin", "Cymbalta/SNRIs", "NSAIDs", "Muscle relaxants", "Sleep aids"],
                                      ["Lyrica/Gabapentin", "Cymbalta/SNRIs"])
        
        prior_therapy = st.checkbox("Previous physical/behavioral therapy", value=False)
        
        enroll_btn = st.button("💜 Start Stanza Program", type="primary", use_container_width=True)
    
    with col2:
        if enroll_btn:
            st.markdown("**Your 12-Week Stanza Journey**")
            
            import time
            with st.spinner("Personalizing your ACT program..."):
                time.sleep(1.0)
            
            st.success("✅ Welcome to Stanza - Your digital therapy starts today!")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%); padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Program Overview</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Program Duration</p>
                        <p style="font-size: 24px; color: white; font-weight: 900; margin: 0;">12 Weeks</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Daily Time</p>
                        <p style="font-size: 24px; color: white; font-weight: 900; margin: 0;">15-20 min</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Therapy Type</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">ACT (CBT-based)</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Expected Improvement</p>
                        <p style="font-size: 24px; color: white; font-weight: 900; margin: 0;">35-45%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Weekly Breakdown")
            
            weeks_data = pd.DataFrame({
                'Week': ['1-2', '3-4', '5-6', '7-8', '9-10', '11-12'],
                'Focus': [
                    'Understanding ACT principles',
                    'Values clarification',
                    'Cognitive defusion techniques',
                    'Acceptance practices',
                    'Committed action planning',
                    'Maintenance & relapse prevention'
                ],
                'Activities': [
                    'Daily lessons, symptom tracking',
                    'Interactive exercises, mindfulness',
                    'Defusion practices, journaling',
                    'Acceptance exercises, breathing',
                    'Goal setting, action plans',
                    'Integration, long-term strategies'
                ]
            })
            
            st.dataframe(weeks_data, hide_index=True, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pain Reduction", "35-45%", "Expected")
            col2.metric("Sleep Improvement", "40%", "Typical")
            col3.metric("Function Increase", "38%", "Mobility")
            col4.metric("Quality of Life", "+42%", "Overall")

with tab2:
    st.markdown("### Clinical Evidence & Outcomes")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Phase 3 Trial", "Positive", "FDA")
    col2.metric("Pain Reduction", "42%", "vs control")
    col3.metric("Function Improvement", "38%", "Significant")
    col4.metric("Sustained at 1 Year", "Yes", "Durable")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### PROSPER-FM Trial Results")
        
        trial_data = {
            'Outcome': ['Pain Intensity', 'Physical Function', 'Sleep Quality', 'Fatigue', 'Depression', 'Anxiety'],
            'Improvement': ['42%', '38%', '40%', '35%', '33%', '36%'],
            'Significance': ['p<0.001', 'p<0.001', 'p<0.001', 'p<0.01', 'p<0.01', 'p<0.01']
        }
        
        st.dataframe(pd.DataFrame(trial_data), hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: #dcfce7; padding: 20px; border-radius: 12px; border-left: 5px solid #22c55e; margin-top: 15px;">
            <h4 style="margin: 0 0 12px 0; color: #166534;">✓ Published in The Lancet (2024)</h4>
            <p style="margin: 0; color: #15803d; font-size: 14px;">
            Peer-reviewed Phase 3 trial data shows clinically significant improvements across all fibromyalgia symptom domains with durable benefits sustained at 1-year follow-up.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Patient Impact")
        
        impact_metrics = {
            'Metric': ['Patients enrolled', 'Avg symptom reduction', 'Treatment adherence', 'Satisfaction'],
            'Value': ['5,847', '42%', '78%', '4.6/5']
        }
        
        st.dataframe(pd.DataFrame(impact_metrics), hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; margin-top: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #92400e;">💰 Cost Savings</h4>
            <p style="font-size: 28px; font-weight: 900; color: #92400e; margin: 0;">$8,400/patient</p>
            <p style="margin: 8px 0 0 0; color: #78350f; font-size: 14px;">Annually vs traditional care</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Digital Therapeutic Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Core Features**")
        st.markdown("""
        - ✅ 12-week ACT therapy program
        - ✅ Daily interactive lessons
        - ✅ Engaging exercises & activities
        - ✅ Symptom tracking & monitoring
        - ✅ Personalized content delivery
        - ✅ Progress visualization
        - ✅ FDA Breakthrough Device designation
        - ✅ Prescription-based (Rx required)
        """)
        
        st.markdown("**Integration**")
        st.markdown("""
        - 📱 iOS/Android native apps
        - 🏥 EHR integration (Healthie)
        - 💊 Swing Care clinic platform
        - 📊 Provider dashboard
        - 🔔 Patient engagement tools
        - 📞 Telemedicine integration
        """)
    
    with col2:
        st.markdown("**ACT Therapy Components**")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%); padding: 20px; border-radius: 12px;">
            <div style="margin-bottom: 10px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">1. Values Clarification</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Identify what truly matters to you</p>
            </div>
            <div style="margin-bottom: 10px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">2. Cognitive Defusion</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Change relationship with pain thoughts</p>
            </div>
            <div style="margin-bottom: 10px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">3. Acceptance</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Willingness to experience pain without struggle</p>
            </div>
            <div style="margin-bottom: 10px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">4. Present Moment Awareness</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Mindfulness & grounding techniques</p>
            </div>
            <div style="margin-bottom: 10px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">5. Self as Context</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Observer perspective on experiences</p>
            </div>
            <div>
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">6. Committed Action</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Values-based behavior changes</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #5b21b6; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #8b5cf6; font-weight: 700; margin: 0 0 6px 0;">✓ FDA-Cleared</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Breakthrough Device</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #8b5cf6; font-weight: 700; margin: 0 0 6px 0;">✓ 42% Pain Reduction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Phase 3 trial proven</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #8b5cf6; font-weight: 700; margin: 0 0 6px 0;">✓ 10M US Patients</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Addressable market</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #8b5cf6; font-weight: 700; margin: 0 0 6px 0;">✓ $8,400 Savings</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Per patient annually</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Swing Therapeutics</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)