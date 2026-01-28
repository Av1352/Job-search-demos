"""
Fireworks AI - AI Infrastructure Platform
Fast and affordable LLM inference at scale
Built for Fireworks AI by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Fireworks AI", page_icon="🔥", layout="wide")

# Model catalog
MODELS = {
    'Llama-3-70B': {'latency': 45, 'cost': 0.0009, 'throughput': 2400, 'quality': 94.5},
    'Llama-3-8B': {'latency': 12, 'cost': 0.0002, 'throughput': 8000, 'quality': 89.2},
    'Mistral-7B': {'latency': 15, 'cost': 0.0002, 'throughput': 7200, 'quality': 87.8},
    'CodeLlama-34B': {'latency': 38, 'cost': 0.0008, 'throughput': 2800, 'quality': 92.3},
    'Mixtral-8x7B': {'latency': 35, 'cost': 0.0005, 'throughput': 3200, 'quality': 91.7}
}

# Infrastructure metrics
INFRA_METRICS = {
    'GPU Utilization': 94.2,
    'Cache Hit Rate': 87.5,
    'Batch Efficiency': 96.8,
    'Network Latency': 8.3,
    'Memory Usage': 78.4
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #f97316 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🔥</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Fireworks AI</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Fast & Affordable LLM Inference</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">10x faster • 5x cheaper • Production-ready infrastructure</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Deploy Model", "⚡ Performance", "💰 Cost Analysis", "💡 Infrastructure"])

with tab1:
    st.markdown("### Deploy LLM with Fireworks")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Model Selection**")
        
        selected_model = st.selectbox("Choose Model", list(MODELS.keys()))
        
        st.markdown("**Configuration**")
        
        max_tokens = st.slider("Max Tokens", 128, 4096, 512)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        replicas = st.slider("Replicas", 1, 10, 3)
        
        st.markdown("**Optimization**")
        enable_cache = st.checkbox("Enable KV Cache", value=True)
        enable_batching = st.checkbox("Dynamic Batching", value=True)
        enable_quantization = st.checkbox("INT8 Quantization", value=True)
        
        deploy_btn = st.button("🔥 Deploy Model", type="primary", use_container_width=True)
    
    with col2:
        if deploy_btn:
            st.markdown("**Deployment Status**")
            
            with st.spinner("Provisioning infrastructure..."):
                import time
                time.sleep(1.5)
            
            st.success("✅ Model deployed successfully!")
            
            model_data = MODELS[selected_model]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f97316 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Deployment Info</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Model</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{selected_model}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Status</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">🟢 Running</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Endpoint</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">fireworks.ai/api/...</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Replicas</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{replicas}x</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Latency", f"{model_data['latency']}ms", "-60ms")
            col2.metric("Throughput", f"{model_data['throughput']} tok/s", "+1200")
            col3.metric("Cost/1M tok", f"${model_data['cost']*1000:.2f}", "-80%")
            col4.metric("Quality", f"{model_data['quality']}%", "✓")
            
            st.markdown("**Quick Test**")
            
            test_prompt = st.text_area("Test Prompt", "Explain quantum computing in simple terms")
            
            if st.button("▶️ Run Inference"):
                with st.spinner("Generating..."):
                    time.sleep(1)
                st.success("✅ Generated in 45ms")
                st.text_area("Response", "Quantum computing uses quantum mechanics principles like superposition and entanglement to perform calculations that would be impossible for classical computers...", height=100)

with tab2:
    st.markdown("### Performance Benchmarks")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Latency Comparison**")
        
        latency_comparison = {
            'Provider': ['Fireworks AI', 'OpenAI', 'Anthropic', 'AWS Bedrock', 'Azure OpenAI'],
            'Llama-3-70B (ms)': [45, 180, 120, 240, 200],
            'Mixtral-8x7B (ms)': [35, 150, 95, 190, 175]
        }
        st.dataframe(pd.DataFrame(latency_comparison), hide_index=True, use_container_width=True)
        
        st.markdown("**🔥 Fireworks is 3-5x faster**")
    
    with col2:
        st.markdown("**Latency Benchmark (ms)**")
        
        fig1 = go.Figure()
        providers = latency_comparison['Provider']
        llama_latency = latency_comparison['Llama-3-70B (ms)']
        
        fig1.add_trace(go.Bar(
            x=providers,
            y=llama_latency,
            marker=dict(color=['#10b981', '#ef4444', '#ef4444', '#ef4444', '#ef4444']),
            text=llama_latency,
            textposition='auto'
        ))
        fig1.update_layout(
            yaxis_title='Latency (ms)',
            height=300
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("**Infrastructure Optimization**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**GPU Utilization**")
        st.metric("", "94.2%", "+12.3%")
        st.progress(0.942)
    
    with col2:
        st.markdown("**Cache Hit Rate**")
        st.metric("", "87.5%", "+15.8%")
        st.progress(0.875)
    
    with col3:
        st.markdown("**Batch Efficiency**")
        st.metric("", "96.8%", "+8.4%")
        st.progress(0.968)

with tab3:
    st.markdown("### Cost Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Pricing Comparison (per 1M tokens)**")
        
        pricing = {
            'Provider': ['Fireworks AI', 'OpenAI GPT-4', 'Anthropic Claude-3', 'AWS Bedrock', 'Azure OpenAI'],
            'Input': ['$0.90', '$30.00', '$15.00', '$35.00', '$30.00'],
            'Output': ['$0.90', '$60.00', '$75.00', '$70.00', '$60.00'],
            'Total (avg)': ['$0.90', '$45.00', '$45.00', '$52.50', '$45.00']
        }
        st.dataframe(pd.DataFrame(pricing), hide_index=True, use_container_width=True)
        
        st.markdown("**💰 Fireworks saves 98% on costs**")
        
        st.markdown("**Monthly Cost Projection (10M tokens/day)**")
        monthly_costs = {
            'Provider': ['Fireworks AI', 'OpenAI', 'Anthropic', 'AWS', 'Azure'],
            'Monthly Cost': ['$2,700', '$135,000', '$135,000', '$157,500', '$135,000']
        }
        st.dataframe(pd.DataFrame(monthly_costs), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Cost Savings Visualization**")
        
        costs = [2700, 135000, 135000, 157500, 135000]
        providers = ['Fireworks', 'OpenAI', 'Anthropic', 'AWS', 'Azure']
        
        fig2 = go.Figure(data=[go.Bar(
            x=providers,
            y=costs,
            marker=dict(color=['#10b981', '#ef4444', '#ef4444', '#ef4444', '#ef4444']),
            text=[f"${c:,}" for c in costs],
            textposition='auto'
        )])
        fig2.update_layout(
            yaxis_title='Monthly Cost ($)',
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("**ROI Calculator**")
        
        monthly_volume = st.slider("Monthly Volume (M tokens)", 1, 100, 10)
        
        fireworks_cost = monthly_volume * 0.9
        competitor_cost = monthly_volume * 45
        savings = competitor_cost - fireworks_cost
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Fireworks Cost", f"${fireworks_cost:,.0f}")
        col2.metric("Competitor Cost", f"${competitor_cost:,.0f}")
        col3.metric("Monthly Savings", f"${savings:,.0f}", f"98%")

with tab4:
    st.markdown("### Infrastructure & Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Infrastructure Features**")
        st.markdown("""
        - ✅ Custom CUDA kernels (10x faster)
        - ✅ KV cache optimization
        - ✅ Dynamic batching
        - ✅ INT8 quantization (2x throughput)
        - ✅ Multi-GPU parallelism
        - ✅ Speculative decoding
        - ✅ Flash Attention
        - ✅ Continuous batching
        """)
        
        st.markdown("**Model Library**")
        st.markdown("""
        - ✅ Llama 3 (8B, 70B, 405B)
        - ✅ Mixtral 8x7B & 8x22B
        - ✅ Mistral 7B & Nemo
        - ✅ CodeLlama 34B
        - ✅ Custom fine-tuned models
        - ✅ Vision models (LLaVA)
        """)
    
    with col2:
        st.markdown("**Performance Optimizations**")
        
        optimizations = {
            'Optimization': ['Custom CUDA Kernels', 'KV Cache', 'Dynamic Batching', 'INT8 Quantization', 'Flash Attention'],
            'Speedup': ['10x', '3x', '4x', '2x', '2x'],
            'Status': ['✅ Enabled', '✅ Enabled', '✅ Enabled', '✅ Enabled', '✅ Enabled']
        }
        st.dataframe(pd.DataFrame(optimizations), hide_index=True, use_container_width=True)
        
        st.markdown("**Infrastructure Metrics**")
        
        metrics_data = []
        for metric, value in INFRA_METRICS.items():
            metrics_data.append({
                'Metric': metric,
                'Value': f"{value}%" if 'Rate' in metric or 'Utilization' in metric or 'Efficiency' in metric or 'Usage' in metric else f"{value}ms",
                'Status': '✅ Optimal'
            })
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #7c2d12; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #ea580c; font-weight: 700; margin: 0 0 6px 0;">✓ 10x Faster</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">45ms latency (vs 180ms)</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #ea580c; font-weight: 700; margin: 0 0 6px 0;">✓ 98% Cheaper</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">$0.90 per 1M tokens</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #ea580c; font-weight: 700; margin: 0 0 6px 0;">✓ Custom CUDA Kernels</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">10x inference speedup</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #ea580c; font-weight: 700; margin: 0 0 6px 0;">✓ 94% GPU Utilization</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Optimized infrastructure</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #f97316 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Fireworks AI</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)