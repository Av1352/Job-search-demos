"""
Bedrock Robotics - Autonomous Construction Equipment
Ex-Waymo engineers building self-driving excavators
Built for Bedrock Robotics by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Bedrock Robotics", page_icon="🏗️", layout="wide")

# Equipment types
EQUIPMENT_TYPES = {
    'Excavator': {'capacity': 'CAT 336', 'precision': '±2cm', 'uptime': '24/7', 'productivity': '+45%'},
    'Bulldozer': {'capacity': 'CAT D6', 'precision': '±3cm', 'uptime': '24/7', 'productivity': '+38%'},
    'Loader': {'capacity': 'CAT 980', 'precision': '±2.5cm', 'uptime': '24/7', 'productivity': '+42%'}
}

# Sensor suite
SENSOR_SUITE = {
    'LiDAR': '360° environment mapping, obstacle detection',
    'HD Cameras (8x)': 'Visual perception, object recognition',
    'GPS + IMU': 'Precise positioning, orientation tracking',
    'On-board Compute': 'Real-time ML inference, path planning'
}

# Project metrics
PROJECT_METRICS = {
    'Site Size': '130 acres',
    'Material Moved': '1.2M cubic yards',
    'Timeline Reduction': '35%',
    'Cost Savings': '40%',
    'Safety Incidents': '0',
    'Operating Hours': '24/7'
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #eab308 0%, #ca8a04 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🏗️</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Bedrock Robotics</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Autonomous Construction Equipment</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">$350M raised • Ex-Waymo • 24/7 autonomous ops</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Bedrock Operator", "📊 Live Operations", "📈 Project Impact", "💡 Technology"])

with tab1:
    st.markdown("### Autonomous Excavation System")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Project Configuration**")
        
        project_name = st.text_input("Project Name", "Data Center Site Preparation")
        site_size = st.number_input("Site Size (acres)", 1, 500, 130)
        location = st.text_input("Location", "Texas")
        
        st.markdown("**Equipment Fleet**")
        
        equipment_type = st.selectbox("Primary Equipment", list(EQUIPMENT_TYPES.keys()))
        equipment_count = st.number_input("Number of Machines", 1, 20, 3)
        
        st.markdown("**Autonomous Configuration**")
        
        autonomy_mode = st.radio("Operating Mode", 
                                ["Supervised (human oversight)", 
                                 "Semi-Autonomous (remote monitoring)",
                                 "Fully Autonomous (2026 target)"])
        
        operating_hours = st.selectbox("Operating Schedule", 
                                      ["Daytime only (8am-6pm)", 
                                       "Extended shift (16 hours)", 
                                       "24/7 continuous"])
        
        material_type = st.selectbox("Material", ["Dirt/Soil", "Rock", "Mixed"])
        
        start_btn = st.button("🏗️ Start Autonomous Operation", type="primary", use_container_width=True)
    
    with col2:
        if start_btn:
            st.markdown("**Live Autonomous Operations Dashboard**")
            
            import time
            with st.spinner("Initializing Bedrock Operator system..."):
                time.sleep(1.2)
            
            st.success("✅ Autonomous excavation active - All systems operational!")
            
            equipment_data = EQUIPMENT_TYPES[equipment_type]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #eab308 0%, #ca8a04 100%); padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Real-Time Fleet Status</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Machines Active</p>
                        <p style="font-size: 36px; color: white; font-weight: 900; margin: 0;">{equipment_count}/{equipment_count}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Operating Mode</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Autonomous</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Uptime</p>
                        <p style="font-size: 36px; color: white; font-weight: 900; margin: 0;">24/7</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Material Moved</p>
                        <p style="font-size: 28px; color: white; font-weight: 900; margin: 0;">1,247 yd³</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Precision</p>
                        <p style="font-size: 28px; color: white; font-weight: 900; margin: 0;">{equipment_data['precision']}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Safety Events</p>
                        <p style="font-size: 36px; color: white; font-weight: 900; margin: 0;">0</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Equipment Details")
            
            equipment_details = pd.DataFrame({
                'Machine': [f'{equipment_type} #1', f'{equipment_type} #2', f'{equipment_type} #3'],
                'Status': ['🟢 Active', '🟢 Active', '🟢 Active'],
                'Battery': ['87%', '92%', '79%'],
                'Material Moved': ['423 yd³', '398 yd³', '426 yd³'],
                'Operating Hours': ['12.3 hrs', '11.8 hrs', '12.5 hrs']
            })
            
            st.dataframe(equipment_details, hide_index=True, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Timeline", "-35%", "vs manual")
            col2.metric("Productivity", f"+{equipment_data['productivity']}", "Improvement")
            col3.metric("Cost", "-40%", "Labor savings")
            col4.metric("Safety", "100%", "Zero incidents")

with tab2:
    st.markdown("### Live Site Operations")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Funding", "$350M", "Series B")
    col2.metric("Active Projects", "4", "Partners")
    col3.metric("States", "4", "CA, TX, AZ, AR")
    col4.metric("2026 Target", "Operator-less", "Fully autonomous")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Daily Productivity")
        
        hours = list(range(0, 24))
        productivity = [420, 410, 425, 430, 435, 440, 445, 450, 465, 480, 495, 505, 510, 505, 498, 490, 485, 480, 470, 460, 450, 445, 435, 428]
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=hours,
            y=productivity,
            mode='lines',
            line=dict(color='#eab308', width=3),
            fill='tozeroy',
            fillcolor='rgba(234, 179, 8, 0.1)'
        ))
        
        fig1.update_layout(
            xaxis_title='Hour of Day',
            yaxis_title='Cubic Yards Moved',
            height=250,
            title='24/7 Continuous Operation'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### Partner Projects")
        
        partners = pd.DataFrame({
            'Partner': ['Sundt Construction', 'Zachry Construction', 'Champion Site Prep', 'Capitol Aggregates'],
            'Location': ['Arizona', 'Texas', 'Texas', 'Texas'],
            'Project Type': ['Manufacturing facility', 'Industrial', 'Site prep', 'Materials']
        })
        
        st.dataframe(partners, hide_index=True, use_container_width=True)

with tab3:
    st.markdown("### Project Impact & Economics")
    
    # Key impact metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #eab308 0%, #ca8a04 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">35%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Timeline Acceleration</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ca8a04 0%, #a16207 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">40%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Cost Reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a16207 0%, #854d0e 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">0</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Safety Incidents</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ROI Analysis")
        
        roi_data = {
            'Category': ['Labor cost reduction', 'Extended operating hours', 'Precision (less rework)', 'Safety improvements'],
            'Annual Value': ['$2.8M', '$1.9M', '$850K', '$640K']
        }
        
        st.dataframe(pd.DataFrame(roi_data), hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; margin-top: 15px;">
            <h4 style="margin: 0 0 10px 0; color: #92400e;">💰 Total Project Savings</h4>
            <p style="font-size: 32px; font-weight: 900; color: #92400e; margin: 0;">$6.2M</p>
            <p style="margin: 8px 0 0 0; color: #78350f; font-size: 14px;">Per large construction project annually</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Industry Crisis")
        
        st.markdown("""
        <div style="background: #fee2e2; padding: 20px; border-radius: 12px; border-left: 5px solid #dc2626;">
            <h4 style="margin: 0 0 12px 0; color: #991b1b;">⚠️ Labor Shortage</h4>
            <p style="margin: 8px 0; color: #7f1d1d; font-size: 14px;">
            • <strong>800,000 workers</strong> needed over next 2 years<br>
            • <strong>8-month</strong> project backlogs (Dec 2025)<br>
            • Retirements widening the gap<br>
            • $13 trillion global construction industry
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Bedrock Solution")
        
        st.markdown("""
        <div style="background: #dcfce7; padding: 20px; border-radius: 12px; border-left: 5px solid #22c55e; margin-top: 15px;">
            <h4 style="margin: 0 0 12px 0; color: #166534;">✓ Autonomous Supplement</h4>
            <p style="margin: 0; color: #15803d; font-size: 14px;">
            Retrofit existing equipment with autonomy—no need to buy new machines. Machines work 24/7 with superhuman precision, supplementing (not replacing) workforce.
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("### Bedrock Operator Technology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sensor Suite**")
        
        for sensor, description in SENSOR_SUITE.items():
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #eab308;">
                <p style="margin: 0; font-weight: 700; color: #92400e;">{sensor}</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**Installation**")
        st.markdown("""
        - ✅ Retrofit kit (few hours to install)
        - ✅ No permanent modifications
        - ✅ Works with existing equipment
        - ✅ Caterpillar, Deere, etc compatible
        - ✅ Removable/transferable
        """)
    
    with col2:
        st.markdown("**Founding Team (Ex-Waymo)**")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 20px; border-radius: 12px;">
            <h4 style="margin: 0 0 15px 0; color: #92400e;">🚗 Waymo → Construction</h4>
            <p style="margin: 8px 0; color: #78350f; font-size: 14px;">
            <strong>Boris Sofman (CEO):</strong><br>
            • Ex-Waymo (Head of Trucking, 4 years)<br>
            • Co-founder Anki Robotics (Cozmo robot)
            </p>
            <p style="margin: 15px 0 8px 0; color: #78350f; font-size: 14px;">
            <strong>Kevin Peterson (CTO):</strong><br>
            • Ex-Waymo (Head of Perception for Via, 3 years)<br>
            • Founded Marble Robot (acquired by Caterpillar 2020)
            </p>
            <p style="margin: 15px 0 0 0; color: #78350f; font-size: 14px;">
            <strong>Team:</strong> 7 years at Waymo (Ajay Gummalla), ex-Segment engineering
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Funding & Investors**")
        
        funding_data = {
            'Round': ['Seed + Series A', 'Series B', 'Total'],
            'Amount': ['$80M', '$270M', '$350M'],
            'Lead': ['Eclipse + 8VC', 'CapitalG + Valor', '—']
        }
        
        st.dataframe(pd.DataFrame(funding_data), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #92400e; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #eab308; font-weight: 700; margin: 0 0 6px 0;">✓ $350M Total Funding</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">CapitalG + Nvidia</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #eab308; font-weight: 700; margin: 0 0 6px 0;">✓ 24/7 Operations</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Continuous autonomous</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #eab308; font-weight: 700; margin: 0 0 6px 0;">✓ Ex-Waymo Team</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">World-class autonomy</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #eab308; font-weight: 700; margin: 0 0 6px 0;">✓ ±2cm Precision</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Superhuman accuracy</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #eab308 0%, #ca8a04 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Bedrock Robotics</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)