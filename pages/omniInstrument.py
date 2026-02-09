"""
Omni Instrument - Autonomous Manufacturing Tools
Edge AI robotics for manufacturing automation
Built for Omni Instrument by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Omni Instrument", page_icon="🤖", layout="wide")

# Edge AI capabilities
EDGE_AI_CAPABILITIES = {
    'On-Device Inference': 'Real-time ML without cloud latency (<10ms)',
    'Computer Vision': 'Object detection, quality inspection, pose estimation',
    'Adaptive Control': 'Real-time adjustment based on sensor feedback',
    'Sensor Fusion': 'Combines vision, force, position data',
    'Autonomous Decision': 'Task planning without human intervention',
    'Zero Downtime': 'Continuous operation with edge processing'
}

# Manufacturing applications
MANUFACTURING_APPS = {
    'Assembly': {'automation': 92, 'precision': '±0.1mm', 'throughput': '2400/hr'},
    'Quality Inspection': {'automation': 98, 'precision': '99.5%', 'throughput': '5000/hr'},
    'Material Handling': {'automation': 95, 'precision': '±2mm', 'throughput': '1800/hr'},
    'Packaging': {'automation': 94, 'precision': '±1mm', 'throughput': '3200/hr'}
}

# Edge AI advantages
EDGE_ADVANTAGES = {
    'Latency': {'edge': '<10ms', 'cloud': '50-200ms'},
    'Reliability': {'edge': '99.9%', 'cloud': '95% (network dependent)'},
    'Data Privacy': {'edge': 'Local processing', 'cloud': 'Cloud transmission'},
    'Operating Cost': {'edge': 'One-time hardware', 'cloud': 'Ongoing fees'}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🤖</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Omni Instrument</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Autonomous Manufacturing Tools</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Edge AI • Robotics • Shield Capital + Afore</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🤖 Autonomous Tools", "📊 Manufacturing Apps", "📈 Performance", "💡 Edge AI"])

with tab1:
    st.markdown("### Edge AI-Powered Manufacturing Automation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Manufacturing Task**")
        
        task_type = st.selectbox("Task Type", list(MANUFACTURING_APPS.keys()))
        
        st.markdown("**Configuration**")
        
        part_type = st.text_input("Part/Product", "Electronic component")
        target_rate = st.number_input("Target Rate (units/hr)", 100, 10000, 2400)
        quality_threshold = st.slider("Quality Threshold (%)", 90, 100, 99)
        
        st.markdown("**Environment**")
        
        lighting = st.selectbox("Lighting Conditions", ["Optimal", "Variable", "Low-light"])
        vibration = st.checkbox("High vibration environment", value=False)
        
        st.markdown("**Edge AI Settings**")
        
        inference_device = st.selectbox("Edge Device", 
                                       ["NVIDIA Jetson AGX", "NVIDIA Jetson Orin", "Intel Movidius"])
        
        model_optimization = st.multiselect("Optimizations",
                                           ["Quantization", "Pruning", "Knowledge distillation"],
                                           ["Quantization"])
        
        deploy_btn = st.button("🤖 Deploy Autonomous Tool", type="primary", use_container_width=True)
    
    with col2:
        if deploy_btn:
            st.markdown("**Autonomous System Status**")
            
            import time
            with st.spinner("Deploying edge AI model..."):
                time.sleep(1.0)
            
            st.success("✅ Autonomous manufacturing tool operational!")
            
            app_data = MANUFACTURING_APPS[task_type]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%); padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Live Operations</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Automation Rate</p>
                        <p style="font-size: 32px; color: white; font-weight: 900; margin: 0;">{app_data['automation']}%</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Precision</p>
                        <p style="font-size: 24px; color: white; font-weight: 900; margin: 0;">{app_data['precision']}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Throughput</p>
                        <p style="font-size: 24px; color: white; font-weight: 900; margin: 0;">{app_data['throughput']}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Inference Latency</p>
                        <p style="font-size: 32px; color: white; font-weight: 900; margin: 0;">8ms</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Edge Device</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{inference_device.split()[1]}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Uptime</p>
                        <p style="font-size: 32px; color: white; font-weight: 900; margin: 0;">99.9%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Real-Time Performance")
            
            performance = pd.DataFrame({
                'Metric': ['Parts Processed', 'Quality Pass Rate', 'Defects Detected', 'Cycle Time', 'Downtime'],
                'Current Shift': ['1,847 units', '99.7%', '6 units', '1.5 sec/unit', '0 min'],
                'Target': ['2,400/shift', '>99.5%', '<10', '<1.5 sec', '<30 min']
            })
            
            st.dataframe(performance, hide_index=True, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Latency", "8ms", "Edge AI")
            col2.metric("Accuracy", "99.7%", "High")
            col3.metric("Uptime", "99.9%", "Reliable")
            col4.metric("Cost", "$0", "No cloud fees")

with tab2:
    st.markdown("### Manufacturing Application Coverage")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Applications", "4", "Core use cases")
    col2.metric("Avg Automation", "94.8%", "High")
    col3.metric("Avg Throughput", "3,100/hr", "Fast")
    col4.metric("Quality", "99.5%+", "Consistent")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Application Performance")
        
        app_data = []
        for app, data in MANUFACTURING_APPS.items():
            app_data.append({
                'Application': app,
                'Automation': f"{data['automation']}%",
                'Precision': data['precision'],
                'Throughput': data['throughput']
            })
        
        st.dataframe(pd.DataFrame(app_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### Automation Rate by Application")
        
        apps = list(MANUFACTURING_APPS.keys())
        automation_rates = [MANUFACTURING_APPS[app]['automation'] for app in apps]
        
        fig1 = go.Figure(data=[go.Bar(
            x=apps,
            y=automation_rates,
            marker_color='#14b8a6',
            text=[f"{rate}%" for rate in automation_rates],
            textposition='auto'
        )])
        
        fig1.update_layout(
            yaxis=dict(range=[85, 100], title='Automation Rate (%)'),
            height=250,
            showlegend=False
        )
        
        st.plotly_chart(fig1, use_container_width=True)

with tab3:
    st.markdown("### Edge AI Performance Metrics")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;"><10ms</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Inference Latency</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">99.9%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Uptime Reliability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0f766e 0%, #115e59 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">$0</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Cloud Fees</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Edge vs Cloud AI")
        
        advantage_data = []
        for metric, values in EDGE_ADVANTAGES.items():
            advantage_data.append({
                'Metric': metric,
                'Edge AI': values['edge'],
                'Cloud AI': values['cloud']
            })
        
        st.dataframe(pd.DataFrame(advantage_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### Cost Comparison")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ccfbf1 0%, #99f6e4 100%); padding: 20px; border-radius: 12px;">
            <h4 style="margin: 0 0 15px 0; color: #134e4a;">💰 Edge AI Economics</h4>
            <p style="margin: 8px 0; color: #0f766e; font-size: 14px;">
            <strong>Edge AI (Omni):</strong><br>
            • One-time hardware: $2,500 - $5,000<br>
            • Ongoing cost: $0 (no cloud fees)<br>
            • 5-year TCO: $5,000
            </p>
            <p style="margin: 15px 0 8px 0; color: #0f766e; font-size: 14px;">
            <strong>Cloud AI:</strong><br>
            • Hardware: $500<br>
            • Cloud fees: $200/month<br>
            • 5-year TCO: $12,500
            </p>
            <p style="margin: 15px 0 0 0; color: #115e59; font-size: 15px; font-weight: 700;">
            Edge AI saves $7,500 over 5 years per robot
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("### Edge AI Technology Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Core Capabilities**")
        
        for capability, description in EDGE_AI_CAPABILITIES.items():
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #14b8a6;">
                <p style="margin: 0; font-weight: 700; color: #0f766e;">{capability}</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Hardware Platform**")
        st.markdown("""
        - 🎮 NVIDIA Jetson (AGX/Orin)
        - 📷 Industrial cameras (stereo vision)
        - 🔧 Force/torque sensors
        - 📍 Precision encoders
        - 🤖 Robotic actuators
        - ⚡ Low-power edge compute
        """)
        
        st.markdown("**Model Optimization**")
        st.markdown("""
        - ✅ INT8 quantization (4x speedup)
        - ✅ Model pruning (50% size reduction)
        - ✅ TensorRT optimization
        - ✅ Custom CUDA kernels
        - ✅ Hardware-aware NAS
        """)
        
        st.markdown("**Investors**")
        
        st.markdown("""
        <div style="background: #dcfce7; padding: 15px; border-radius: 10px; margin-top: 15px;">
            <p style="margin: 0; color: #166534; font-size: 14px;">
            <strong>Backed by:</strong><br>
            • Shield Capital (defense tech focus)<br>
            • Afore Capital (deep tech)
            </p>
        </div>
        """, unsafe_allow_html=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #ccfbf1 0%, #99f6e4 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #134e4a; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #14b8a6; font-weight: 700; margin: 0 0 6px 0;">✓ <10ms Latency</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Edge AI inference</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #14b8a6; font-weight: 700; margin: 0 0 6px 0;">✓ 99.9% Uptime</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">No cloud dependency</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #14b8a6; font-weight: 700; margin: 0 0 6px 0;">✓ Shield + Afore</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Top deep tech VCs</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #14b8a6; font-weight: 700; margin: 0 0 6px 0;">✓ 94.8% Automation</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Across applications</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Omni Instrument</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)