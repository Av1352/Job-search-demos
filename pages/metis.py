"""
Metis - Infrastructure to Build Reliable Agents
Agent monitoring, testing, and reliability platform
Built for Metis by Anju Nandhakumar
"""

import streamlit as st
from utils.sidebar import render_sidebar
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Metis - Reliable Agents", layout="wide")
render_sidebar()

# Initialize session state
if 'agent_deployed' not in st.session_state:
    st.session_state.agent_deployed = False

# Agent reliability data
AGENTS = {
    "Customer Support Agent": {
        "type": "Conversational",
        "interactions_daily": 2400,
        "uptime": 99.7,
        "success_rate": 94.2,
        "avg_response_time": 1.8,
        "error_rate": 2.3,
        "hallucination_rate": 1.2,
        "escalation_rate": 8.5
    },
    "Sales Qualification Agent": {
        "type": "Decision-Making",
        "interactions_daily": 850,
        "uptime": 99.9,
        "success_rate": 91.5,
        "avg_response_time": 3.2,
        "error_rate": 3.1,
        "hallucination_rate": 2.5,
        "escalation_rate": 12.3
    },
    "Code Review Agent": {
        "type": "Analysis",
        "interactions_daily": 320,
        "uptime": 99.5,
        "success_rate": 88.7,
        "avg_response_time": 8.5,
        "error_rate": 4.2,
        "hallucination_rate": 3.8,
        "escalation_rate": 15.2
    }
}

def generate_reliability_metrics(agent_data):
    """Calculate agent reliability metrics"""
    
    # Calculate composite reliability score
    weights = {
        'uptime': 0.25,
        'success_rate': 0.30,
        'error_rate': 0.20,
        'response_time': 0.15,
        'hallucination_rate': 0.10
    }
    
    uptime_score = agent_data['uptime'] / 100
    success_score = agent_data['success_rate'] / 100
    error_score = 1 - (agent_data['error_rate'] / 100)
    response_score = max(0, 1 - (agent_data['avg_response_time'] / 10))
    hallucination_score = 1 - (agent_data['hallucination_rate'] / 100)
    
    reliability_score = (
        uptime_score * weights['uptime'] +
        success_score * weights['success_rate'] +
        error_score * weights['error_rate'] +
        response_score * weights['response_time'] +
        hallucination_score * weights['hallucination_rate']
    ) * 100
    
    # Classify reliability tier
    if reliability_score >= 95:
        tier = "Production-Ready"
        tier_color = "#10b981"
    elif reliability_score >= 85:
        tier = "Needs Tuning"
        tier_color = "#f59e0b"
    else:
        tier = "Not Ready"
        tier_color = "#ef4444"
    
    return {
        'score': reliability_score,
        'tier': tier,
        'tier_color': tier_color,
        'monthly_interactions': agent_data['interactions_daily'] * 30,
        'monthly_errors': int(agent_data['interactions_daily'] * 30 * agent_data['error_rate'] / 100),
        'monthly_escalations': int(agent_data['interactions_daily'] * 30 * agent_data['escalation_rate'] / 100)
    }

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #059669 0%, #73BA9B 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(115, 186, 155, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #14b8a6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🛡️</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">Metis</h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Infrastructure for Reliable Agents</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Build, monitor, and debug AI agents at scale</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);">Monitoring</span>
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Testing</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Reliability</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">YC Backed</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">Built for <strong style="color: white;">Metis</strong> by <strong style="color: white;">Anju Nandhakumar</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Value Prop
st.markdown("""
<div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); padding: 25px; border-radius: 15px; border: 2px solid #059669; margin-bottom: 30px;">
    <h3 style="color: #065f46; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">🎯 The Agent Reliability Problem</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #ef4444; font-weight: 700; margin: 0 0 8px 0;">❌ Today</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Agents fail in production. Hallucinations break trust. No visibility into errors. Hard to debug black box failures.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #f59e0b; font-weight: 700; margin: 0 0 8px 0;">💰 Cost Impact</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Failed interactions cost revenue. Engineer time debugging. Customer churn from bad experiences. Can't ship with confidence.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #10b981; font-weight: 700; margin: 0 0 8px 0;">✅ With Metis</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Real-time monitoring, automated testing, hallucination detection. 99.7% uptime. Debug failures instantly. Ship agents with confidence.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🛡️ Agent Dashboard", "🧪 Reliability Testing", "📊 Monitoring"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Select Agent to Monitor</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Real-time reliability metrics for your production agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    agent_name = st.selectbox("Agent", list(AGENTS.keys()))
    agent = AGENTS[agent_name]
    
    if st.button("📊 View Agent Metrics", type="primary", use_container_width=True):
        st.session_state.agent_deployed = True
        st.session_state.current_agent = agent_name
        st.session_state.reliability = generate_reliability_metrics(agent)
    
    if st.session_state.agent_deployed and st.session_state.current_agent == agent_name:
        reliability = st.session_state.reliability
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; margin-top: 25px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">🛡️ Reliability Score</h2>
            <div style="text-align: center; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 30px; margin-bottom: 20px;">
                <p style="font-size: 64px; color: white; font-weight: 900; margin: 0;">{reliability['score']:.1f}</p>
                <p style="font-size: 20px; color: {reliability['tier_color']}; font-weight: 700; margin: 10px 0 0 0;">{reliability['tier']}</p>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Uptime</p>
                    <p style="font-size: 32px; color: #86efac; font-weight: 900; margin: 8px 0;">{agent['uptime']}%</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Success Rate</p>
                    <p style="font-size: 32px; color: white; font-weight: 900; margin: 8px 0;">{agent['success_rate']}%</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Response Time</p>
                    <p style="font-size: 32px; color: white; font-weight: 900; margin: 8px 0;">{agent['avg_response_time']}s</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Error Rate</p>
                    <p style="font-size: 32px; color: #fbbf24; font-weight: 900; margin: 8px 0;">{agent['error_rate']}%</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"""
            <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
                <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">📊 Monthly Volume</h3>
                <div style="background: #f9fafb; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 15px;">
                    <p style="color: #6b7280; font-size: 13px; margin: 0;">Total Interactions</p>
                    <p style="color: #3b82f6; font-size: 36px; font-weight: 900; margin: 8px 0;">{reliability['monthly_interactions']:,}</p>
                </div>
                <div style="background: #fef2f2; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Errors: <strong style="color: #ef4444;">{reliability['monthly_errors']}</strong></p>
                </div>
                <div style="background: #fef3c7; padding: 15px; border-radius: 10px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Escalations: <strong style="color: #f59e0b;">{reliability['monthly_escalations']}</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div style="background: #fef2f2; padding: 25px; border-radius: 15px; border: 2px solid #ef4444;">
                <h3 style="color: #7f1d1d; margin: 0 0 20px 0; font-size: 20px;">⚠️ Risk Factors</h3>
                <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <p style="color: #6b7280; font-size: 13px; margin: 0;">Hallucination Rate</p>
                    <p style="color: #ef4444; font-size: 24px; font-weight: 700; margin: 5px 0 0 0;">{agent['hallucination_rate']}%</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <p style="color: #6b7280; font-size: 13px; margin: 0;">Error Rate</p>
                    <p style="color: #f59e0b; font-size: 24px; font-weight: 700; margin: 5px 0 0 0;">{agent['error_rate']}%</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 10px;">
                    <p style="color: #6b7280; font-size: 13px; margin: 0;">Escalation Rate</p>
                    <p style="color: #f59e0b; font-size: 24px; font-weight: 700; margin: 5px 0 0 0;">{agent['escalation_rate']}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Automated Agent Testing</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Continuous testing to catch failures before production</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
        <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">🧪 Test Suite Results</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 2px solid #e5e7eb;">
                <th style="text-align: left; padding: 12px; color: #6b7280; font-size: 13px;">Test Category</th>
                <th style="text-align: center; padding: 12px; color: #6b7280; font-size: 13px;">Tests Run</th>
                <th style="text-align: center; padding: 12px; color: #6b7280; font-size: 13px;">Pass Rate</th>
            </tr>
            <tr style="border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 12px; color: #1f2937;">Response Quality</td>
                <td style="text-align: center; padding: 12px; color: #6b7280;">247</td>
                <td style="text-align: center; padding: 12px; color: #059669; font-weight: 700;">96.4%</td>
            </tr>
            <tr style="background: #f9fafb; border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 12px; color: #1f2937;">Hallucination Detection</td>
                <td style="text-align: center; padding: 12px; color: #6b7280;">82</td>
                <td style="text-align: center; padding: 12px; color: #059669; font-weight: 700;">98.8%</td>
            </tr>
            <tr style="border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 12px; color: #1f2937;">Edge Cases</td>
                <td style="text-align: center; padding: 12px; color: #6b7280;">156</td>
                <td style="text-align: center; padding: 12px; color: #f59e0b; font-weight: 700;">84.6%</td>
            </tr>
            <tr style="background: #f9fafb; border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 12px; color: #1f2937;">Safety & Compliance</td>
                <td style="text-align: center; padding: 12px; color: #6b7280;">93</td>
                <td style="text-align: center; padding: 12px; color: #059669; font-weight: 700;">99.1%</td>
            </tr>
            <tr>
                <td style="padding: 12px; color: #1f2937; font-weight: 700;">Overall</td>
                <td style="text-align: center; padding: 12px; color: #6b7280; font-weight: 700;">578</td>
                <td style="text-align: center; padding: 12px; color: #059669; font-weight: 900;">95.2%</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Real-Time Monitoring</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Track agent performance across all metrics 24/7</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mock monitoring data
    hours = list(range(24))
    success_rates = [92 + np.random.normal(0, 2) for _ in hours]
    response_times = [1.8 + np.random.normal(0, 0.3) for _ in hours]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_success = go.Figure()
        fig_success.add_trace(go.Scatter(
            x=hours, y=success_rates,
            mode='lines+markers',
            line=dict(color='#059669', width=2),
            fill='tonexty',
            fillcolor='rgba(5, 150, 105, 0.1)'
        ))
        fig_success.add_hline(y=90, line_dash="dash", line_color="#ef4444", annotation_text="Threshold")
        fig_success.update_layout(
            title="Success Rate (Last 24 Hours)",
            xaxis_title="Hour",
            yaxis_title="Success %",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig_success, use_container_width=True)
    
    with col2:
        fig_latency = go.Figure()
        fig_latency.add_trace(go.Scatter(
            x=hours, y=response_times,
            mode='lines+markers',
            line=dict(color='#3b82f6', width=2),
            fill='tonexty',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        fig_latency.add_hline(y=3.0, line_dash="dash", line_color="#ef4444", annotation_text="SLA")
        fig_latency.update_layout(
            title="Response Time (Last 24 Hours)",
            xaxis_title="Hour",
            yaxis_title="Seconds",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig_latency, use_container_width=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #059669; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Metis</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🛡️ 99.7% Uptime</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">Real-time monitoring catches failures instantly. Automated testing prevents bugs. Hallucination detection before users see them.</p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🔍 Full Observability</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">See every interaction, trace failures, debug black boxes. Know exactly why agents fail. Fix issues in minutes, not days.</p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📈 95% Reliability</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">Composite score across uptime, success, errors, latency. Ship agents with confidence. Production-grade reliability.</p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Production Agent Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">99.7% uptime:</strong> Reliable enough for production</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">95% test pass rate:</strong> Catch issues before users</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">1.2% hallucination:</strong> Detection and prevention</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Real-time debugging:</strong> Trace failures instantly</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Monitoring</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Track success, latency, errors 24/7</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Hallucination Detection</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Catch false information before users</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Automated Testing</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">578 tests run continuously</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Debug Tools</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Trace failures, replay interactions</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #059669 0%, #73BA9B 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(5, 150, 105, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">Built for <strong style="color: white;">Metis</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong></p>
        <div style="margin: 25px 0; padding: 22px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.2);">
            <p style="margin: 8px 0; font-size: 16px;">📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: white; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a></p>
            <p style="margin: 8px 0; font-size: 16px;">💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">LinkedIn</a> | 💻 <a href="https://github.com/Av1352" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">GitHub</a> | 🌐 <a href="https://vxanju.com" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">Portfolio</a></p>
        </div>
        <p style="font-size: 15px; margin: 18px 0; font-weight: 700;"><strong style="color: white;">Tech Stack:</strong> Agent Monitoring • Testing Frameworks • Observability • Reliability Engineering</p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">Demo showcasing infrastructure for building reliable AI agents at scale.<br>Real-time monitoring • Automated testing • Hallucination detection • Debug tools • Reliability scoring</p>
    </div>
    """, unsafe_allow_html=True)