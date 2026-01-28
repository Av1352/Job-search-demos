"""
Modal - ML Infrastructure (Serverless GPUs)
Run ML workloads on serverless GPUs with simple Python
Built for Modal by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Modal", page_icon="⚡", layout="wide")

# Workload types
WORKLOAD_TYPES = {
    'Training': {'gpu': 'A100 (80GB)', 'duration': '2.5 hrs', 'cost': '$12.50'},
    'Inference': {'gpu': 'T4', 'duration': '500ms/req', 'cost': '$0.0003/req'},
    'Fine-tuning': {'gpu': 'A100 (40GB)', 'duration': '45 min', 'cost': '$5.25'},
    'Batch Processing': {'gpu': '8x A100', 'duration': '15 min', 'cost': '$8.00'},
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #6366f1 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">⚡</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Modal</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Serverless ML Infrastructure</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Run ML workloads on serverless GPUs • Zero DevOps • Pay per second</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["⚡ Deploy ML Model", "📊 Cost Comparison", "🚀 Workload Examples", "💡 Platform Features"])

with tab1:
    st.markdown("### Deploy ML Model with Modal")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Model Configuration**")
        
        model_type = st.selectbox(
            "Model Type",
            ["Image Classification (ResNet50)", "LLM Inference (Llama-2-7B)", "Object Detection (YOLOv8)", "Fine-tuning (BERT)"]
        )
        
        gpu_type = st.selectbox("GPU Type", ["A100 (80GB)", "A100 (40GB)", "T4", "V100"])
        replicas = st.slider("Replicas", 1, 8, 2)
        
        st.markdown("**Deployment Code**")
        st.code("""
import modal

stub = modal.Stub("ml-inference")

@stub.function(
    gpu="A100",
    memory=32768,
    timeout=300
)
def run_inference(image):
    import torch
    model = load_model()
    return model(image)

@stub.local_entrypoint()
def main():
    result = run_inference.remote(img)
    print(result)
""", language="python")
        
        deploy_btn = st.button("⚡ Deploy to Modal", type="primary", use_container_width=True)
    
    with col2:
        if deploy_btn:
            st.markdown("**Deployment Status**")
            
            with st.spinner("Provisioning GPUs..."):
                import time
                time.sleep(1.5)
            
            st.success("✅ Model deployed successfully!")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #6366f1 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Deployment Info</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Endpoint</p>
                        <p style="font-size: 16px; color: white; font-weight: 700; margin: 0;">modal.run/ml-inference</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Status</p>
                        <p style="font-size: 16px; color: white; font-weight: 700; margin: 0;">🟢 Running</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">GPU</p>
                        <p style="font-size: 16px; color: white; font-weight: 700; margin: 0;">2x A100 (80GB)</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Cold Start</p>
                        <p style="font-size: 16px; color: white; font-weight: 700; margin: 0;">2.3s</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Latency", "45ms", "-30ms")
            col2.metric("Throughput", "250 req/s", "+150")
            col3.metric("Cost/1K req", "$0.30", "-$2.70")
            col4.metric("Uptime", "99.9%", "✓")

with tab2:
    st.markdown("### Cost Comparison: Modal vs Traditional Cloud")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Training Workload (ResNet50, 100 epochs)**")
        
        cost_comparison = {
            'Provider': ['Modal (Serverless)', 'AWS EC2 (p4d.24xlarge)', 'GCP (a2-ultragpu-8g)', 'Azure (ND96amsr_A100_v4)'],
            'GPU Hours': ['2.5 hrs', '2.5 hrs', '2.5 hrs', '2.5 hrs'],
            'Total Cost': ['$12.50', '$243.75', '$198.50', '$215.00'],
            'DevOps Time': ['0 hrs', '4 hrs', '3 hrs', '3.5 hrs']
        }
        st.dataframe(pd.DataFrame(cost_comparison), hide_index=True, use_container_width=True)
        
        st.markdown("**💰 Modal Savings: $186-$202 per training run**")
    
    with col2:
        st.markdown("**Cost Breakdown**")
        
        fig1 = go.Figure(data=[go.Bar(
            x=['Modal', 'AWS', 'GCP', 'Azure'],
            y=[12.50, 243.75, 198.50, 215.00],
            marker=dict(color=['#10b981', '#ef4444', '#ef4444', '#ef4444']),
            text=['$12.50', '$243.75', '$198.50', '$215.00'],
            textposition='auto'
        )])
        fig1.update_layout(
            yaxis_title='Cost ($)',
            height=300
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("**Why Modal is Cheaper**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **No Idle Costs**
        - Pay per second of usage
        - Auto-shutdown when done
        - No reserved instances
        """)
    with col2:
        st.markdown("""
        **Zero DevOps**
        - No cluster management
        - No Kubernetes setup
        - No infrastructure code
        """)
    with col3:
        st.markdown("""
        **Optimized Pricing**
        - Bulk GPU purchasing
        - Efficient scheduling
        - Spot instance usage
        """)

with tab3:
    st.markdown("### Example Workloads")
    
    for workload, details in WORKLOAD_TYPES.items():
        with st.expander(f"**{workload}** - {details['gpu']}", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("GPU Type", details['gpu'])
            col2.metric("Duration", details['duration'])
            col3.metric("Cost", details['cost'])
            
            if workload == 'Training':
                st.code("""
@stub.function(gpu="A100", memory=65536, timeout=7200)
def train_model(dataset_url):
    model = ResNet50()
    train_loader = load_data(dataset_url)
    
    for epoch in range(100):
        for batch in train_loader:
            loss = model.train_step(batch)
    
    return model.save()
""", language="python")
            
            elif workload == 'Inference':
                st.code("""
@stub.function(gpu="T4", memory=8192, keep_warm=5)
def predict(image_bytes):
    model = load_cached_model()
    image = preprocess(image_bytes)
    return model.predict(image)

@stub.webhook(method="POST")
def inference_endpoint(item: dict):
    return predict.remote(item["image"])
""", language="python")

with tab4:
    st.markdown("### Platform Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Serverless Features**")
        st.markdown("""
        - ✅ Auto-scaling (0 to 1000s of GPUs)
        - ✅ Cold start < 3 seconds
        - ✅ Pay per second usage
        - ✅ No cluster management
        - ✅ Automatic retries
        - ✅ Built-in monitoring
        """)
        
        st.markdown("**GPU Options**")
        st.markdown("""
        - ✅ A100 (80GB) - $5.00/hr
        - ✅ A100 (40GB) - $3.00/hr
        - ✅ T4 - $0.60/hr
        - ✅ V100 - $2.00/hr
        - ✅ H100 - $7.00/hr (preview)
        """)
    
    with col2:
        st.markdown("**Developer Experience**")
        st.markdown("""
        - ✅ Pure Python (no YAML)
        - ✅ Local dev → Cloud deploy
        - ✅ Version control friendly
        - ✅ Built-in CI/CD
        - ✅ Secrets management
        - ✅ Logs & metrics
        """)
        
        st.markdown("**Use Cases**")
        st.markdown("""
        - ✅ Model training & fine-tuning
        - ✅ Batch inference
        - ✅ Real-time API serving
        - ✅ Data processing pipelines
        - ✅ Research experiments
        """)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #3730a3; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #4f46e5; font-weight: 700; margin: 0 0 6px 0;">✓ Serverless GPUs</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Scale 0 to 1000s instantly</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #4f46e5; font-weight: 700; margin: 0 0 6px 0;">✓ Pay Per Second</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">No idle costs, 95% savings</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #4f46e5; font-weight: 700; margin: 0 0 6px 0;">✓ < 3s Cold Start</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Fast GPU provisioning</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #4f46e5; font-weight: 700; margin: 0 0 6px 0;">✓ Zero DevOps</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Pure Python, no YAML</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #6366f1 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Modal</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)