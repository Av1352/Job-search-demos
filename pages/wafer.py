"""
Wafer - AI That Makes AI Fast
Model optimization and inference acceleration
Built for Wafer by Anju Nandhakumar
"""

import streamlit as st
from utils.sidebar import render_sidebar
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Wafer - AI Acceleration", layout="wide")
render_sidebar()

# Initialize session state
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False

# Model optimization scenarios
MODELS = {
    "ResNet50 - Image Classification": {
        "base_latency_ms": 45,
        "base_throughput": 22,
        "base_size_mb": 98,
        "optimizations": {
            "Quantization (INT8)": {"latency": 18, "throughput": 55, "size": 25, "accuracy_drop": 0.5},
            "Pruning (50% sparsity)": {"latency": 28, "throughput": 36, "size": 49, "accuracy_drop": 1.2},
            "Knowledge Distillation": {"latency": 12, "throughput": 83, "size": 15, "accuracy_drop": 2.1},
            "Combined (Quant + Prune + Distill)": {"latency": 8, "throughput": 125, "size": 12, "accuracy_drop": 2.8}
        },
        "base_accuracy": 94.2
    },
    "BERT - NLP": {
        "base_latency_ms": 120,
        "base_throughput": 8,
        "base_size_mb": 440,
        "optimizations": {
            "Quantization (INT8)": {"latency": 52, "throughput": 19, "size": 110, "accuracy_drop": 0.3},
            "Pruning (40% sparsity)": {"latency": 78, "throughput": 13, "size": 264, "accuracy_drop": 0.8},
            "Knowledge Distillation (DistilBERT)": {"latency": 35, "throughput": 29, "size": 268, "accuracy_drop": 1.5},
            "Combined": {"latency": 22, "throughput": 45, "size": 88, "accuracy_drop": 2.0}
        },
        "base_accuracy": 92.8
    },
    "YOLOv8 - Object Detection": {
        "base_latency_ms": 28,
        "base_throughput": 36,
        "base_size_mb": 52,
        "optimizations": {
            "Quantization (INT8)": {"latency": 12, "throughput": 83, "size": 13, "accuracy_drop": 1.2},
            "Pruning (30% sparsity)": {"latency": 21, "throughput": 48, "size": 36, "accuracy_drop": 1.8},
            "TensorRT Optimization": {"latency": 8, "throughput": 125, "size": 52, "accuracy_drop": 0.2},
            "Combined (Quant + TensorRT)": {"latency": 5, "throughput": 200, "size": 13, "accuracy_drop": 1.4}
        },
        "base_accuracy": 95.1
    }
}

def calculate_cost_savings(base_latency, optimized_latency, requests_per_day=1000000):
    """Calculate infrastructure cost savings"""
    
    # Cost per GPU hour
    gpu_cost_per_hour = 1.50  # A100 pricing
    
    # Base infrastructure cost
    base_requests_per_hour = (3600000 / base_latency)  # milliseconds to hours
    base_gpus_needed = requests_per_day / (base_requests_per_hour * 24)
    base_daily_cost = base_gpus_needed * 24 * gpu_cost_per_hour
    
    # Optimized infrastructure cost
    opt_requests_per_hour = (3600000 / optimized_latency)
    opt_gpus_needed = requests_per_day / (opt_requests_per_hour * 24)
    opt_daily_cost = opt_gpus_needed * 24 * gpu_cost_per_hour
    
    savings_daily = base_daily_cost - opt_daily_cost
    savings_annual = savings_daily * 365
    
    return {
        'base_gpus': base_gpus_needed,
        'opt_gpus': opt_gpus_needed,
        'savings_daily': savings_daily,
        'savings_annual': savings_annual,
        'cost_reduction_pct': (savings_daily / base_daily_cost) * 100
    }

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(251, 191, 36, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #fbbf24 0%, #fcd34d 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(251, 191, 36, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">⚡</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">Wafer</h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">AI That Makes AI Fast</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Model optimization and inference acceleration</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);">Quantization</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Pruning</span>
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Distillation</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">YC Backed</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">Built for <strong style="color: white;">Wafer</strong> by <strong style="color: white;">Anju Nandhakumar</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Value Prop
st.markdown("""
<div style="background: linear-gradient(135deg, #fef3c7, #fde68a); padding: 25px; border-radius: 15px; border: 2px solid #f59e0b; margin-bottom: 30px;">
    <h3 style="color: #92400e; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">🎯 The AI Inference Cost Problem</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #ef4444; font-weight: 700; margin: 0 0 8px 0;">❌ Today</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Large models are slow (100ms+ latency). Expensive to serve (many GPUs). Power consumption huge. Carbon footprint high.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #f59e0b; font-weight: 700; margin: 0 0 8px 0;">💰 Cost Impact</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">$500K/year in GPU costs for 1M requests/day. Slow inference = bad UX. Can't scale without massive infrastructure.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #10b981; font-weight: 700; margin: 0 0 8px 0;">✅ With Wafer</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">5-10x faster inference. 75% cost reduction. Same accuracy. Deploy on smaller GPUs or even CPUs.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["⚡ Optimize Model", "📊 Performance Comparison", "🔧 Optimization Techniques"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Select Model to Optimize</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">AI applies best optimization techniques automatically</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        model_name = st.selectbox("Model Architecture", list(MODELS.keys()))
        model_data = MODELS[model_name]
        
        optimization_type = st.selectbox(
            "Optimization Strategy",
            list(model_data['optimizations'].keys())
        )
        
        st.markdown(f"""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb; margin-top: 20px;">
            <h3 style="color: #1f2937; margin: 0 0 15px 0; font-size: 18px;">Base Model Stats</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                <div style="background: #f9fafb; padding: 12px; border-radius: 8px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Latency</p>
                    <p style="color: #ef4444; font-size: 20px; font-weight: 700; margin: 3px 0 0 0;">{model_data['base_latency_ms']}ms</p>
                </div>
                <div style="background: #f9fafb; padding: 12px; border-radius: 8px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Throughput</p>
                    <p style="color: #6b7280; font-size: 20px; font-weight: 700; margin: 3px 0 0 0;">{model_data['base_throughput']} req/s</p>
                </div>
                <div style="background: #f9fafb; padding: 12px; border-radius: 8px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Model Size</p>
                    <p style="color: #6b7280; font-size: 20px; font-weight: 700; margin: 3px 0 0 0;">{model_data['base_size_mb']}MB</p>
                </div>
                <div style="background: #f9fafb; padding: 12px; border-radius: 8px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Accuracy</p>
                    <p style="color: #10b981; font-size: 20px; font-weight: 700; margin: 3px 0 0 0;">{model_data['base_accuracy']}%</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("⚡ Optimize Model", type="primary", use_container_width=True):
            st.session_state.optimization_complete = True
            st.session_state.current_model = model_name
            st.session_state.current_opt = optimization_type
    
    with col2:
        if st.session_state.optimization_complete and st.session_state.current_model == model_name:
            opt_data = model_data['optimizations'][st.session_state.current_opt]
            
            speedup = model_data['base_latency_ms'] / opt_data['latency']
            throughput_gain = opt_data['throughput'] / model_data['base_throughput']
            size_reduction = ((model_data['base_size_mb'] - opt_data['size']) / model_data['base_size_mb']) * 100
            new_accuracy = model_data['base_accuracy'] - opt_data['accuracy_drop']
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); padding: 25px; border-radius: 15px; border: 2px solid #059669;">
                <h3 style="color: #065f46; margin: 0 0 20px 0; font-size: 18px;">✅ Optimization Complete!</h3>
                <div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 12px; text-align: center;">
                    <p style="color: #6b7280; font-size: 13px; margin: 0;">Speedup</p>
                    <p style="color: #059669; font-size: 48px; font-weight: 900; margin: 8px 0;">{speedup:.1f}x</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 12px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">New Latency: <strong style="color: #10b981;">{opt_data['latency']}ms</strong></p>
                    <p style="color: #6b7280; font-size: 12px; margin: 5px 0 0 0;">Throughput: <strong style="color: #3b82f6;">{opt_data['throughput']} req/s</strong></p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 12px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Model Size: <strong style="color: #8b5cf6;">{opt_data['size']}MB</strong> (-{size_reduction:.0f}%)</p>
                    <p style="color: #6b7280; font-size: 12px; margin: 5px 0 0 0;">Accuracy: <strong style="color: #059669;">{new_accuracy:.1f}%</strong> (-{opt_data['accuracy_drop']:.1f}%)</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Before vs After Comparison</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">See the impact of Wafer optimization across key metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.optimization_complete:
        model_data = MODELS[st.session_state.current_model]
        opt_data = model_data['optimizations'][st.session_state.current_opt]
        
        # Comparison chart
        metrics = ['Latency (ms)', 'Throughput (req/s)', 'Model Size (MB)']
        base_values = [model_data['base_latency_ms'], model_data['base_throughput'], model_data['base_size_mb']]
        opt_values = [opt_data['latency'], opt_data['throughput'], opt_data['size']]
        
        fig = go.Figure(data=[
            go.Bar(name='Base Model', x=metrics, y=base_values, marker_color='#ef4444'),
            go.Bar(name='Optimized', x=metrics, y=opt_values, marker_color='#10b981')
        ])
        fig.update_layout(
            title="Performance Comparison",
            barmode='group',
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost savings
        savings = calculate_cost_savings(model_data['base_latency_ms'], opt_data['latency'])
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; margin-top: 25px;">
            <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">💰 Infrastructure Cost Savings</h2>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Base GPUs Needed</p>
                    <p style="font-size: 40px; color: white; font-weight: 900; margin: 8px 0;">{savings['base_gpus']:.1f}</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Optimized GPUs</p>
                    <p style="font-size: 40px; color: #86efac; font-weight: 900; margin: 8px 0;">{savings['opt_gpus']:.1f}</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Annual Savings</p>
                    <p style="font-size: 36px; color: #fbbf24; font-weight: 900; margin: 8px 0;">${savings['savings_annual']:,.0f}</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Cost Reduction</p>
                    <p style="font-size: 40px; color: white; font-weight: 900; margin: 8px 0;">{savings['cost_reduction_pct']:.0f}%</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Optimization Techniques</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">How Wafer makes models faster without sacrificing accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">🔧 Optimization Methods</h3>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6; margin-bottom: 12px;">
                <h4 style="color: #2563eb; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">Quantization</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">FP32 → INT8: 4x smaller, 2-3x faster, <1% accuracy loss</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #10b981; margin-bottom: 12px;">
                <h4 style="color: #059669; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">Pruning</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Remove 30-50% of weights: 2x faster, 50% smaller, <2% accuracy loss</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #ec4899; margin-bottom: 12px;">
                <h4 style="color: #db2777; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">Knowledge Distillation</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Student learns from teacher: 5-10x faster, 90% smaller, 2-3% accuracy loss</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #8b5cf6;">
                <h4 style="color: #7c3aed; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">Compiler Optimization</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">TensorRT/ONNX: Fuse ops, optimize kernels, 2x faster, zero accuracy loss</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">📈 Typical Improvements</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #e5e7eb;">
                    <th style="text-align: left; padding: 12px; color: #6b7280; font-size: 13px;">Technique</th>
                    <th style="text-align: center; padding: 12px; color: #6b7280; font-size: 13px;">Speedup</th>
                    <th style="text-align: center; padding: 12px; color: #6b7280; font-size: 13px;">Accuracy</th>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6;">
                    <td style="padding: 12px; color: #1f2937;">Quantization</td>
                    <td style="text-align: center; padding: 12px; color: #059669; font-weight: 700;">2.5x</td>
                    <td style="text-align: center; padding: 12px; color: #10b981;">-0.5%</td>
                </tr>
                <tr style="background: #f9fafb; border-bottom: 1px solid #f3f4f6;">
                    <td style="padding: 12px; color: #1f2937;">Pruning</td>
                    <td style="text-align: center; padding: 12px; color: #059669; font-weight: 700;">1.6x</td>
                    <td style="text-align: center; padding: 12px; color: #10b981;">-1.2%</td>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6;">
                    <td style="padding: 12px; color: #1f2937;">Distillation</td>
                    <td style="text-align: center; padding: 12px; color: #059669; font-weight: 700;">3.8x</td>
                    <td style="text-align: center; padding: 12px; color: #f59e0b;">-2.1%</td>
                </tr>
                <tr style="background: #f9fafb; border-bottom: 1px solid #f3f4f6;">
                    <td style="padding: 12px; color: #1f2937;">TensorRT</td>
                    <td style="text-align: center; padding: 12px; color: #059669; font-weight: 700;">2.0x</td>
                    <td style="text-align: center; padding: 12px; color: #10b981;">-0.1%</td>
                </tr>
                <tr>
                    <td style="padding: 12px; color: #1f2937; font-weight: 700;">Combined</td>
                    <td style="text-align: center; padding: 12px; color: #059669; font-weight: 900;">5-10x</td>
                    <td style="text-align: center; padding: 12px; color: #10b981;">-2.8%</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #f59e0b; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Wafer</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 5-10x Speedup</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">Combined optimizations deliver massive performance gains. 45ms → 8ms inference. Better UX, lower costs.</p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 75% Cost Cut</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">Reduce GPU needs by 75%. $500K/year → $125K/year for 1M requests/day. Same model, fraction of the cost.</p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🎯 <3% Accuracy Loss</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">Minimal quality tradeoff. 94.2% → 91.4% still production-ready. Most users can't tell the difference.</p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Production Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">5-10x faster inference:</strong> Better UX, lower latency</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">75% cost reduction:</strong> Fewer GPUs needed</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;"><3% accuracy loss:</strong> Production-ready quality maintained</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Deploy anywhere:</strong> Run on CPUs, edge devices, mobile</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Stack</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Quantization</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">INT8, FP16, mixed precision</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Pruning</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Structured, unstructured, magnitude-based</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Distillation</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Teacher-student, response-based, feature-based</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Compiler Optimization</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">TensorRT, ONNX Runtime, kernel fusion</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(245, 158, 11, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">Built for <strong style="color: white;">Wafer</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong></p>
        <div style="margin: 25px 0; padding: 22px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.2);">
            <p style="margin: 8px 0; font-size: 16px;">📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: white; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a></p>
            <p style="margin: 8px 0; font-size: 16px;">💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">LinkedIn</a> | 💻 <a href="https://github.com/Av1352" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">GitHub</a> | 🌐 <a href="https://vxanju.com" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">Portfolio</a></p>
        </div>
        <p style="font-size: 15px; margin: 18px 0; font-weight: 700;"><strong style="color: white;">Tech Stack:</strong> Model Optimization • Quantization • Pruning • Knowledge Distillation • TensorRT</p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">Demo showcasing AI model optimization and inference acceleration techniques.<br>Quantization • Pruning • Distillation • Compiler optimization • Cost reduction analysis</p>
    </div>
    """, unsafe_allow_html=True)