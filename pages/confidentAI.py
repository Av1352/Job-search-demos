"""
Confident AI - LLM Eval and Observability Platform
Monitor, evaluate, and debug LLM applications in production
Built for Confident AI by Anju Nandhakumar
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
st.set_page_config(page_title="Confident AI", page_icon="📊", layout="wide")

# LLM models
LLM_MODELS = {
    'GPT-4': {'cost_per_1k': 0.03, 'latency_avg': 1200, 'accuracy': 94.5},
    'GPT-3.5-Turbo': {'cost_per_1k': 0.002, 'latency_avg': 450, 'accuracy': 89.2},
    'Claude-3-Opus': {'cost_per_1k': 0.015, 'latency_avg': 980, 'accuracy': 93.8},
    'Claude-3-Sonnet': {'cost_per_1k': 0.003, 'latency_avg': 520, 'accuracy': 90.5},
    'Llama-3-70B': {'cost_per_1k': 0.001, 'latency_avg': 380, 'accuracy': 87.3}
}

# Generate sample evaluation data
def generate_eval_data(model_name, num_samples=100):
    np.random.seed(42)
    base_accuracy = LLM_MODELS[model_name]['accuracy']
    
    data = {
        'timestamp': [datetime.now() - timedelta(minutes=i*5) for i in range(num_samples)],
        'accuracy': np.random.normal(base_accuracy, 2, num_samples),
        'latency_ms': np.random.normal(LLM_MODELS[model_name]['latency_avg'], 100, num_samples),
        'cost': np.random.exponential(LLM_MODELS[model_name]['cost_per_1k'], num_samples),
        'hallucination_rate': np.random.beta(2, 50, num_samples) * 100,
        'toxicity_score': np.random.beta(1, 100, num_samples) * 100
    }
    return pd.DataFrame(data)

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #8b5cf6 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">📊</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Confident AI</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">LLM Eval and Observability Platform</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Monitor performance • Detect drift • Debug failures • Ensure quality</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-Time Monitoring", "🔍 Evaluation Dashboard", "🚨 Drift Detection", "💡 System Features"])

with tab1:
    st.markdown("### Live Model Performance")
    
    # Model selector
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_model = st.selectbox("Select LLM Model", list(LLM_MODELS.keys()))
        time_window = st.selectbox("Time Window", ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"])
    
    # Generate data
    df = generate_eval_data(selected_model)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Accuracy", f"{df['accuracy'].mean():.1f}%", f"{np.random.uniform(-1, 1):.1f}%")
    col2.metric("Avg Latency", f"{df['latency_ms'].mean():.0f}ms", f"{np.random.uniform(-50, 50):.0f}ms")
    col3.metric("Hallucination Rate", f"{df['hallucination_rate'].mean():.2f}%", f"-{np.random.uniform(0.1, 0.5):.2f}%")
    col4.metric("Avg Cost/1K", f"${df['cost'].mean():.4f}", f"+${np.random.uniform(0.0001, 0.001):.4f}")
    
    # Accuracy over time
    st.markdown("**Accuracy Trend**")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['accuracy'],
        mode='lines',
        name='Accuracy',
        line=dict(color='#10b981', width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    fig1.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Threshold (90%)")
    fig1.update_layout(
        xaxis_title='Time',
        yaxis_title='Accuracy (%)',
        height=300,
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Latency distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Latency Distribution**")
        fig2 = go.Figure(data=[go.Histogram(
            x=df['latency_ms'],
            nbinsx=30,
            marker=dict(color='#3b82f6')
        )])
        fig2.update_layout(
            xaxis_title='Latency (ms)',
            yaxis_title='Count',
            height=250
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("**Cost vs Latency**")
        fig3 = go.Figure(data=[go.Scatter(
            x=df['latency_ms'],
            y=df['cost'],
            mode='markers',
            marker=dict(
                color=df['accuracy'],
                colorscale='Viridis',
                size=8,
                colorbar=dict(title="Accuracy")
            )
        )])
        fig3.update_layout(
            xaxis_title='Latency (ms)',
            yaxis_title='Cost ($)',
            height=250
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown("### Model Evaluation Metrics")
    
    # Evaluation categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Performance Metrics**")
        
        perf_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'BLEU Score', 'ROUGE-L'],
            'Score': [
                f"{np.random.uniform(88, 95):.1f}%",
                f"{np.random.uniform(86, 93):.1f}%",
                f"{np.random.uniform(85, 92):.1f}%",
                f"{np.random.uniform(87, 94):.1f}%",
                f"{np.random.uniform(0.6, 0.85):.2f}",
                f"{np.random.uniform(0.65, 0.88):.2f}"
            ],
            'Status': ['✅ Pass', '✅ Pass', '✅ Pass', '✅ Pass', '✅ Pass', '✅ Pass']
        }
        st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Safety Metrics**")
        safety_data = {
            'Metric': ['Hallucination Rate', 'Toxicity Score', 'Bias Detection', 'PII Leakage'],
            'Score': [
                f"{np.random.uniform(1.2, 2.8):.2f}%",
                f"{np.random.uniform(0.3, 1.5):.2f}%",
                f"{np.random.uniform(2.1, 4.8):.2f}%",
                f"{np.random.uniform(0.1, 0.5):.2f}%"
            ],
            'Status': ['✅ Low', '✅ Low', '✅ Low', '✅ Low']
        }
        st.dataframe(pd.DataFrame(safety_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Test Suite Results**")
        
        # Test categories
        test_results = {
            'Category': ['Correctness', 'Hallucination', 'Toxicity', 'Bias', 'Context Relevance'],
            'Tests Run': [156, 89, 134, 78, 112],
            'Passed': [152, 87, 132, 76, 109],
            'Failed': [4, 2, 2, 2, 3]
        }
        test_df = pd.DataFrame(test_results)
        test_df['Pass Rate'] = (test_df['Passed'] / test_df['Tests Run'] * 100).round(1).astype(str) + '%'
        st.dataframe(test_df, hide_index=True, use_container_width=True)
        
        # Visual test results
        fig4 = go.Figure(data=[
            go.Bar(name='Passed', x=test_results['Category'], y=test_results['Passed'], marker_color='#10b981'),
            go.Bar(name='Failed', x=test_results['Category'], y=test_results['Failed'], marker_color='#ef4444')
        ])
        fig4.update_layout(
            barmode='stack',
            height=250,
            xaxis_title='Test Category',
            yaxis_title='Count'
        )
        st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.markdown("### Drift Detection & Alerts")
    
    # Drift detection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Model Performance Drift**")
        
        # Generate drift data
        days = 30
        baseline_acc = 92.5
        drift_data = {
            'day': list(range(days)),
            'accuracy': [baseline_acc - i*0.3 + np.random.uniform(-1, 1) for i in range(days)],
            'baseline': [baseline_acc] * days
        }
        drift_df = pd.DataFrame(drift_data)
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=drift_df['day'],
            y=drift_df['accuracy'],
            mode='lines+markers',
            name='Current',
            line=dict(color='#3b82f6', width=2)
        ))
        fig5.add_trace(go.Scatter(
            x=drift_df['day'],
            y=drift_df['baseline'],
            mode='lines',
            name='Baseline',
            line=dict(color='#10b981', width=2, dash='dash')
        ))
        fig5.add_hline(y=85, line_dash="dot", line_color="red", annotation_text="Critical Threshold")
        fig5.update_layout(
            xaxis_title='Days',
            yaxis_title='Accuracy (%)',
            height=300
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        st.markdown("**Active Alerts**")
        
        alerts = [
            {"severity": "🟡", "type": "Warning", "message": "Accuracy dropped 3.2%"},
            {"severity": "🟢", "type": "Info", "message": "Latency improved 15ms"},
            {"severity": "🔴", "type": "Critical", "message": "Hallucination spike +2.1%"},
            {"severity": "🟡", "type": "Warning", "message": "Cost increased 12%"}
        ]
        
        for alert in alerts:
            st.markdown(f"""
            <div style="background: white; border-left: 4px solid {'#f59e0b' if alert['severity']=='🟡' else '#10b981' if alert['severity']=='🟢' else '#ef4444'}; padding: 12px; margin: 8px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 0; font-size: 14px;"><strong>{alert['severity']} {alert['type']}</strong></p>
                <p style="margin: 4px 0 0 0; font-size: 13px; color: #6b7280;">{alert['message']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Drift summary
    st.markdown("**Drift Analysis Summary**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Drift Magnitude", "3.2%", "⚠️ Moderate")
    col2.metric("Days Since Baseline", "30 days", "+30")
    col3.metric("Retraining Recommended", "Yes", "In 5 days")

with tab4:
    st.markdown("### Platform Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Real-Time Monitoring**")
        st.markdown("""
        - ✅ Live performance dashboards
        - ✅ Accuracy, latency, cost tracking
        - ✅ Custom metric definitions
        - ✅ Multi-model comparison
        - ✅ API response time monitoring
        - ✅ Token usage analytics
        """)
        
        st.markdown("**Evaluation & Testing**")
        st.markdown("""
        - ✅ Automated test suites (578 tests)
        - ✅ Hallucination detection
        - ✅ Toxicity & bias checking
        - ✅ Context relevance scoring
        - ✅ BLEU, ROUGE metrics
        - ✅ Custom evaluators
        """)
    
    with col2:
        st.markdown("**Drift Detection**")
        st.markdown("""
        - ✅ Statistical drift analysis
        - ✅ Performance degradation alerts
        - ✅ Distribution shift tracking
        - ✅ Baseline comparison
        - ✅ Automatic retraining triggers
        """)
        
        st.markdown("**Debug & Trace**")
        st.markdown("""
        - ✅ Request/response logging
        - ✅ Error replay & analysis
        - ✅ Prompt versioning
        - ✅ A/B test framework
        - ✅ Root cause identification
        """)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #5b21b6; font-size: 24px; font-weight: 800;">💡 System Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Monitoring</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Track accuracy, latency, cost 24/7</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ Hallucination Detection</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Catch false information before users</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ Automated Testing</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">578 tests run continuously</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ Debug Tools</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Trace failures, replay interactions</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #8b5cf6 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Confident AI</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)