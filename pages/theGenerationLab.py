"""
The Generation Lab - Longevity Diagnostics Platform  
SystemAge™ biological age testing across 19 organ systems
Built for The Generation Lab by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="The Generation Lab - Longevity", page_icon="🧬", layout="wide")

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🧬</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">The Generation Lab</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">SystemAge™ Longevity Diagnostics</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">19 organ systems • DNA methylation • Age reversal tracking</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🧬 SystemAge™ Test", "📊 Your Aging Profile", "💡 Interventions"])

with tab1:
    st.markdown("### Comprehensive Biological Age Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Patient Information**")
        
        name = st.text_input("Name", "Michael Chen")
        chronological_age = st.number_input("Chronological Age", 25, 85, 45)
        
        st.markdown("**Sample Collection**")
        
        sample_type = st.radio("Test Type", ["Blood Draw", "Cheek Swab"])
        collection_date = st.date_input("Collection Date", datetime.now())
        
        st.markdown("**Lifestyle Factors**")
        
        exercise = st.select_slider("Exercise Frequency", 
                                    options=["Sedentary", "1-2x/week", "3-4x/week", "5+ x/week"],
                                    value="3-4x/week")
        
        diet_quality = st.select_slider("Diet Quality",
                                       options=["Poor", "Fair", "Good", "Excellent"],
                                       value="Good")
        
        sleep_hours = st.slider("Avg Sleep (hours)", 4, 10, 7)
        
        supplements = st.multiselect("Current Interventions",
                                    ["NAD+ precursors", "Metformin", "Rapamycin", "Resveratrol", "None"],
                                    ["NAD+ precursors"])
        
        analyze_btn = st.button("🧬 Analyze SystemAge™", type="primary", use_container_width=True)
    
    with col2:
        if analyze_btn:
            st.markdown("**Your SystemAge™ Report**")
            
            import time
            with st.spinner("Analyzing 460 biomarkers across 19 organ systems..."):
                time.sleep(1.5)
            
            st.success("✅ SystemAge™ analysis complete!")
            
            # Calculate biological age (younger for healthier)
            bio_age = chronological_age - 3.2
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Overall Biological Age</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Chronological Age</p>
                        <p style="font-size: 36px; color: white; font-weight: 900; margin: 0;">{chronological_age}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Biological Age</p>
                        <p style="font-size: 36px; color: white; font-weight: 900; margin: 0;">{bio_age:.1f}</p>
                        <p style="font-size: 14px; color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">↓ 3.2 years younger!</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Organ System Breakdown")
            
            systems_data = pd.DataFrame({
                'System': ['Cardiovascular', 'Immune', 'Metabolic', 'Brain/Cognitive', 'Reproductive', 'Liver', 'Kidney', 'Musculoskeletal'],
                'Bio Age': [f'{chronological_age - 5.2:.1f}', f'{chronological_age - 2.1:.1f}', f'{chronological_age + 1.8:.1f}', 
                           f'{chronological_age - 4.3:.1f}', f'{chronological_age - 6.1:.1f}', f'{chronological_age:.1f}',
                           f'{chronological_age - 1.5:.1f}', f'{chronological_age + 2.3:.1f}'],
                'Status': ['🟢 Excellent', '🟢 Good', '🟡 Monitor', '🟢 Excellent', '🟢 Excellent', '🟢 Good', '🟢 Good', '🟡 Monitor']
            })
            
            st.dataframe(systems_data, hide_index=True, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Best System", "Reproductive", "-6.1 years")
            col2.metric("Needs Attention", "Musculoskeletal", "+2.3 years")
            col3.metric("Biomarkers", "460", "Analyzed")
            col4.metric("Overall", "Healthy", "🟢")

with tab2:
    st.markdown("### Detailed Aging Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">275+</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Clinic Partners</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">19</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Organ Systems Measured</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0e7490 0%, #155e75 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">460</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Biomarkers Analyzed</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Personalized Interventions")
    
    st.markdown("""
    <div style="background: #dcfce7; padding: 20px; border-radius: 12px; border-left: 5px solid #22c55e;">
        <h4 style="margin: 0 0 12px 0; color: #166534;">Recommended Actions</h4>
        <ul style="margin: 0; padding-left: 20px; color: #15803d;">
            <li>Increase resistance training 2x/week for musculoskeletal system</li>
            <li>Consider NAD+ supplementation for metabolic support</li>
            <li>Maintain current cardiovascular exercise routine</li>
            <li>Monitor metabolic markers quarterly</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #cffafe 0%, #a5f3fc 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #155e75; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #06b6d4; font-weight: 700; margin: 0 0 6px 0;">✓ 19 Organ Systems</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Comprehensive analysis</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #06b6d4; font-weight: 700; margin: 0 0 6px 0;">✓ 460 Biomarkers</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">DNA methylation</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #06b6d4; font-weight: 700; margin: 0 0 6px 0;">✓ $15M Funded</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Accel-led</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #06b6d4; font-weight: 700; margin: 0 0 6px 0;">✓ 275+ Clinics</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Worldwide partners</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for The Generation Lab</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)