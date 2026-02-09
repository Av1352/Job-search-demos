"""
Superpower - Preventative Health Super App
100+ biomarker testing with AI-powered insights
Built for Superpower by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Superpower - Health Super App", page_icon="⚡", layout="wide")

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">⚡</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Superpower</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Your Personal Health OS</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">100+ biomarkers • AI insights • $499/year</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["⚡ Dashboard", "🧪 Lab Results", "💡 AI Insights"])

with tab1:
    st.markdown("### Your Health Command Center")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">87</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Superpower Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">38.5</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Biological Age</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">102</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Biomarkers Tracked</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">3</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Action Items</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Comprehensive Biomarker Analysis")
    
    categories = ['Hormones', 'Metabolic', 'Cardiovascular', 'Immune', 'Inflammation', 'Liver', 'Kidney', 'Nutrients']
    values = [92, 85, 88, 90, 78, 94, 91, 82]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker=dict(color='#6366f1')
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### AI-Powered Health Recommendations")
    
    st.markdown("""
    <div style="background: #dcfce7; padding: 20px; border-radius: 12px; border-left: 5px solid #22c55e;">
        <h4 style="margin: 0 0 12px 0; color: #166534;">✓ Strengths</h4>
        <ul style="margin: 0; padding-left: 20px; color: #15803d;">
            <li>Excellent cardiovascular health</li>
            <li>Optimal hormone balance</li>
            <li>Strong immune function</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #fef3c7; padding: 20px; border-radius: 12px; border-left: 5px solid #f59e0b; margin-top: 15px;">
        <h4 style="margin: 0 0 12px 0; color: #92400e;">⚠️ Areas to Improve</h4>
        <ul style="margin: 0; padding-left: 20px; color: #78350f;">
            <li>Inflammation markers slightly elevated - reduce processed foods</li>
            <li>Vitamin D low - consider 2000 IU daily supplementation</li>
            <li>Sleep quality suboptimal - aim for 8 hours consistently</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #3730a3; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #6366f1; font-weight: 700; margin: 0 0 6px 0;">✓ 100+ Biomarkers</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Twice yearly</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #6366f1; font-weight: 700; margin: 0 0 6px 0;">✓ $30M Funded</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Forerunner-led</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #6366f1; font-weight: 700; margin: 0 0 6px 0;">✓ 24/7 AI Concierge</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Hybrid AI + human</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #6366f1; font-weight: 700; margin: 0 0 6px 0;">✓ 150K Waitlist</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Now live</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Superpower</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)