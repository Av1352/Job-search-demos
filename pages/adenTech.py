"""
Aden Technologies - AI Agent Observability Platform
Real-time monitoring and debugging for AI agents
Built for Aden Technologies by Anju Nandhakumar
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Aden Technologies Demo - Anju Vilashni",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px 24px;
}
.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px 32px;
    font-size: 16px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# Agent types and their characteristics
AGENT_TYPES = {
    "Research Agent": {
        "avg_latency": 3.5,
        "success_rate": 0.92,
        "cost_per_call": 0.08,
        "tools": ["web_search", "document_reader", "summarizer"]
    },
    "Code Agent": {
        "avg_latency": 5.2,
        "success_rate": 0.88,
        "cost_per_call": 0.12,
        "tools": ["code_interpreter", "file_system", "git_commands"]
    },
    "Data Agent": {
        "avg_latency": 2.8,
        "success_rate": 0.95,
        "cost_per_call": 0.05,
        "tools": ["sql_query", "data_transformer", "chart_generator"]
    },
    "Customer Service Agent": {
        "avg_latency": 1.5,
        "success_rate": 0.96,
        "cost_per_call": 0.03,
        "tools": ["knowledge_base", "ticket_creator", "email_sender"]
    }
}

def generate_agent_execution_trace(agent_type, success=True):
    """Generate a realistic agent execution trace"""
    
    agent_config = AGENT_TYPES[agent_type]
    tools = agent_config["tools"]
    
    trace = {
        "agent_id": f"agent_{random.randint(1000, 9999)}",
        "agent_type": agent_type,
        "timestamp": datetime.now().isoformat(),
        "status": "success" if success else "error",
        "total_latency": agent_config["avg_latency"] * random.uniform(0.8, 1.2),
        "total_cost": agent_config["cost_per_call"] * random.uniform(0.9, 1.1),
        "steps": []
    }
    
    # Generate execution steps
    num_steps = random.randint(3, 6)
    
    step_templates = {
        "planning": "Agent planning: Analyzing user request and determining action sequence",
        "tool_selection": f"Tool selection: Choosing {random.choice(tools)} for execution",
        "tool_execution": f"Executing {random.choice(tools)}",
        "result_processing": "Processing tool output and extracting relevant information",
        "response_generation": "Generating final response using LLM",
        "validation": "Validating output quality and completeness"
    }
    
    step_types = list(step_templates.keys())
    selected_steps = random.sample(step_types, min(num_steps, len(step_types)))
    
    for i, step_type in enumerate(selected_steps):
        step_latency = random.uniform(0.3, 1.5)
        
        step = {
            "step_number": i + 1,
            "step_type": step_type,
            "description": step_templates[step_type],
            "latency_ms": step_latency * 1000,
            "token_usage": random.randint(50, 500),
            "status": "success" if (success or i < len(selected_steps) - 1) else "error"
        }
        
        if not success and i == len(selected_steps) - 1:
            step["error"] = "TimeoutError: Agent execution exceeded maximum time limit"
        
        trace["steps"].append(step)
    
    return trace

import streamlit.components.v1 as components

# Header - Complete in one component
components.html("""
<div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
    <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; margin: 0 auto 25px auto; border: 5px solid white; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5);">
        <span style="font-size: 56px;">🤖</span>
    </div>
    
    <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        Aden Agent Observatory
    </h1>
    
    <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">AI-Native Observability for Agents</p>
    <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Real-time monitoring • Trace visualization • Error debugging</p>
    
    <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 800px; margin: 28px auto 0 auto;">
        <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Multi-Agent</span>
        <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Real-Time</span>
        <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Trace Analysis</span>
        <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Cost Tracking</span>
    </div>
    
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
        Built for <strong style="color: white;">Aden Technologies</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
    </p>
</div>
""", height=500)

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Execution Trace", "📊 Dashboard", "⚠️ Error Analysis"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Agent Execution Tracing</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Visualize step-by-step agent execution with latency breakdown</p>
    </div>
    """, unsafe_allow_html=True)
    
    agent_type = st.selectbox("Select Agent Type", list(AGENT_TYPES.keys()), key="trace_agent")
    
    if st.button("🔍 Generate Execution Trace", key="trace_btn"):
        success = random.random() > 0.1
        trace = generate_agent_execution_trace(agent_type, success)
        
        status_color = "#10b981" if trace["status"] == "success" else "#ef4444"
        status_emoji = "✅" if trace["status"] == "success" else "❌"
        
        # Trace header
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🔍 Agent Execution Trace</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Agent ID</p>
                <p style="font-size: 20px; color: #667eea; font-weight: 900; margin: 0; font-family: monospace;">{trace["agent_id"]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Status</p>
                <p style="font-size: 28px; color: {status_color}; font-weight: 900; margin: 0;">{status_emoji} {trace["status"].upper()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Total Latency</p>
                <p style="font-size: 32px; color: #667eea; font-weight: 900; margin: 0;">{trace["total_latency"]:.2f}s</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Cost</p>
                <p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">${trace["total_cost"]:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Execution steps
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📝 Execution Steps</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for step in trace["steps"]:
            step_color = "#10b981" if step["status"] == "success" else "#ef4444"
            
            step_html = f"""
            <div style="background: white; border-left: 5px solid {step_color}; border-radius: 12px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <span style="background: {step_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-right: 10px;">STEP {step["step_number"]}</span>
                        <span style="font-size: 16px; color: #1f2937; font-weight: 700;">{step["step_type"].replace('_', ' ').title()}</span>
                    </div>
                    <div style="text-align: right;">
                        <p style="font-size: 18px; color: #3b82f6; font-weight: 800; margin: 0;">{step["latency_ms"]:.0f}ms</p>
                        <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">{step["token_usage"]} tokens</p>
                    </div>
                </div>
                <p style="font-size: 14px; color: #6b7280; margin: 0; line-height: 1.6;">{step["description"]}</p>
            """
            
            if "error" in step:
                step_html += f"""
                <div style="background: #fee2e2; border: 2px solid #dc2626; border-radius: 8px; padding: 12px; margin-top: 10px;">
                    <p style="font-size: 13px; color: #991b1b; font-weight: 600; margin: 0;">❌ Error: {step["error"]}</p>
                </div>
                """
            
            step_html += "</div>"
            st.markdown(step_html, unsafe_allow_html=True)
        
        # Waterfall chart
        step_names = [f"Step {s['step_number']}: {s['step_type']}" for s in trace["steps"]]
        latencies = [s["latency_ms"] for s in trace["steps"]]
        
        fig_waterfall = go.Figure()
        fig_waterfall.add_trace(go.Waterfall(
            name="Latency",
            orientation="v",
            x=step_names,
            y=latencies,
            connector={"line": {"color": "#3b82f6"}},
            decreasing={"marker": {"color": "#ef4444"}},
            increasing={"marker": {"color": "#10b981"}},
            totals={"marker": {"color": "#8b5cf6"}}
        ))
        
        fig_waterfall.update_layout(
            title="Execution Latency Waterfall (ms)",
            yaxis_title="Latency (ms)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_waterfall, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Live Agent Monitoring</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Real-time metrics across all agents in your system</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Load Dashboard", key="dashboard_btn"):
        # Dashboard header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Live Agent Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Total Executions</p>
                <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">12,547</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Last 24 hours</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(16, 185, 129, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Success Rate</p>
                <p style="font-size: 48px; color: #10b981; font-weight: 900; margin: 0;">94.2%</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">11,823 successful</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: rgba(59, 130, 246, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(59, 130, 246, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Avg Latency</p>
                <p style="font-size: 48px; color: #3b82f6; font-weight: 900; margin: 0;">2.8s</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">P95: 4.2s</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: rgba(251, 191, 36, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(251, 191, 36, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Total Cost</p>
                <p style="font-size: 48px; color: #f59e0b; font-weight: 900; margin: 0;">$847</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">$0.067 per exec</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Agent breakdown
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #6b21a8; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🤖 Agent Performance Breakdown</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for agent_name, config in AGENT_TYPES.items():
            executions = random.randint(2000, 4000)
            success_rate = config["success_rate"] + random.uniform(-0.03, 0.03)
            avg_latency = config["avg_latency"] + random.uniform(-0.5, 0.5)
            
            color = "#10b981" if success_rate >= 0.93 else "#f59e0b" if success_rate >= 0.85 else "#ef4444"
            
            st.markdown(f"""
            <div style="background: white; border-radius: 14px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 6px 0;">{agent_name}</p>
                        <p style="font-size: 13px; color: #6b7280; margin: 0;">Tools: {', '.join(config['tools'])}</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="font-size: 28px; color: {color}; font-weight: 900; margin: 0;">{success_rate:.1%}</p>
                        <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">{executions:,} runs</p>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 12px;">
                    <div style="background: #f9fafb; border-radius: 8px; padding: 12px;">
                        <p style="font-size: 11px; color: #6b7280; margin: 0 0 4px 0;">Avg Latency</p>
                        <p style="font-size: 18px; color: #3b82f6; font-weight: 700; margin: 0;">{avg_latency:.2f}s</p>
                    </div>
                    <div style="background: #f9fafb; border-radius: 8px; padding: 12px;">
                        <p style="font-size: 11px; color: #6b7280; margin: 0 0 4px 0;">Cost/Run</p>
                        <p style="font-size: 18px; color: #f59e0b; font-weight: 700; margin: 0;">${config['cost_per_call']:.3f}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        hours = list(range(24))
        success_rates = [0.92 + random.uniform(-0.05, 0.05) for _ in hours]
        
        fig_success = go.Figure()
        fig_success.add_trace(go.Scatter(
            x=hours,
            y=success_rates,
            mode='lines+markers',
            line=dict(color='#10b981', width=3),
            marker=dict(size=6),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        fig_success.add_hline(y=0.95, line_dash="dash", line_color="#059669", 
                              annotation_text="Target: 95%", annotation_position="right")
        fig_success.update_layout(
            title="Success Rate (Last 24 Hours)",
            xaxis_title="Hour",
            yaxis_title="Success Rate",
            yaxis_range=[0.8, 1.0],
            height=400
        )
        st.plotly_chart(fig_success, use_container_width=True)
        
        # Latency by agent
        agent_names = list(AGENT_TYPES.keys())
        latencies = [AGENT_TYPES[name]["avg_latency"] + random.uniform(-0.3, 0.3) for name in agent_names]
        
        fig_latency = go.Figure(data=[
            go.Bar(
                x=agent_names,
                y=latencies,
                marker_color=['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'],
                text=[f'{l:.2f}s' for l in latencies],
                textposition='outside'
            )
        ])
        fig_latency.update_layout(
            title="Average Latency by Agent Type",
            yaxis_title="Latency (seconds)",
            height=400
        )
        st.plotly_chart(fig_latency, use_container_width=True)
        
        # Cost breakdown
        costs = [AGENT_TYPES[name]["cost_per_call"] * random.randint(2000, 4000) for name in agent_names]
        
        fig_cost = go.Figure(data=[go.Pie(
            labels=agent_names,
            values=costs,
            marker=dict(colors=['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b']),
            hole=0.4,
            textinfo='label+percent'
        )])
        fig_cost.update_layout(title="Cost Distribution by Agent Type (24h)", height=400)
        st.plotly_chart(fig_cost, use_container_width=True)

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #991b1b; font-size: 22px; font-weight: 800; margin: 0;">Error Detection & Debugging</h3>
        <p style="color: #dc2626; font-size: 14px; margin: 8px 0 0 0;">Identify and resolve agent failures with actionable insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("⚠️ Analyze Errors", key="error_btn"):
        error_types = {
            "Timeout": {"count": 342, "severity": "High", "color": "#ef4444"},
            "Rate Limit": {"count": 187, "severity": "Medium", "color": "#f59e0b"},
            "Invalid Output": {"count": 124, "severity": "Medium", "color": "#f59e0b"},
            "Tool Failure": {"count": 89, "severity": "High", "color": "#ef4444"},
            "Validation Error": {"count": 67, "severity": "Low", "color": "#fbbf24"}
        }
        
        total_errors = sum(e["count"] for e in error_types.values())
        
        # Error header
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 25px;">
                <div style="background: #ef4444; width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 4px solid white;">
                    <span style="font-size: 36px;">⚠️</span>
                </div>
                <div>
                    <h2 style="color: #991b1b; font-size: 32px; font-weight: 900; margin: 0;">Error Analysis</h2>
                    <p style="color: #dc2626; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">{total_errors} errors detected in last 24 hours</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Error breakdown
        for error_type, data in error_types.items():
            pct = (data["count"] / total_errors) * 100
            
            st.markdown(f"""
            <div style="background: white; border-left: 5px solid {data['color']}; border-radius: 12px; padding: 18px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="background: {data['color']}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-right: 10px;">{data['severity'].upper()}</span>
                        <span style="font-size: 18px; color: #1f2937; font-weight: 700;">{error_type}</span>
                    </div>
                    <div style="text-align: right;">
                        <p style="font-size: 28px; color: {data['color']}; font-weight: 900; margin: 0;">{data['count']}</p>
                        <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">{pct:.1f}%</p>
                    </div>
                </div>
                <div style="background: #e5e7eb; border-radius: 8px; height: 8px; overflow: hidden;">
                    <div style="background: {data['color']}; height: 100%; width: {pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; margin-top: 25px;">
            <h3 style="color: #1e40af; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">💡 Debugging Recommendations</h3>
            <div style="background: white; border-radius: 12px; padding: 20px;">
                <ul style="margin: 0; padding-left: 24px; line-height: 2.2;">
                    <li style="color: #1f2937; font-size: 15px; font-weight: 600;">🔧 <strong>Timeout Issues:</strong> Increase max_execution_time from 30s to 45s for Research Agent</li>
                    <li style="color: #1f2937; font-size: 15px; font-weight: 600;">⚡ <strong>Rate Limits:</strong> Implement exponential backoff with jitter (current: fixed 1s delay)</li>
                    <li style="color: #1f2937; font-size: 15px; font-weight: 600;">✓ <strong>Output Validation:</strong> Add JSON schema validation before parsing LLM responses</li>
                    <li style="color: #1f2937; font-size: 15px; font-weight: 600;">🛠️ <strong>Tool Failures:</strong> Add fallback tools when primary tool returns error</li>
                    <li style="color: #1f2937; font-size: 15px; font-weight: 600;">📊 <strong>Monitoring:</strong> Set up alerts when error rate exceeds 10% in 5-minute window</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Error trend
        hours = list(range(24))
        error_trend = [random.randint(20, 60) for _ in hours]
        
        fig_errors = go.Figure()
        fig_errors.add_trace(go.Scatter(
            x=hours,
            y=error_trend,
            mode='lines+markers',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=6),
            fill='tonexty',
            fillcolor='rgba(239, 68, 68, 0.1)'
        ))
        fig_errors.update_layout(
            title="Error Rate Over Time (Last 24 Hours)",
            xaxis_title="Hour",
            yaxis_title="Number of Errors",
            height=400
        )
        st.plotly_chart(fig_errors, use_container_width=True)

# Footer
st.markdown("<hr style='border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Aden Technologies</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: rgba(102, 126, 234, 0.1); border-radius: 16px; padding: 24px; margin-top: 20px; text-align: center;">
    <p style="margin: 8px 0; font-size: 16px;">
        📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: #667eea; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a>
    </p>
    <p style="margin: 8px 0; font-size: 16px;">
        💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: #667eea; font-weight: 700; text-decoration: none;">LinkedIn</a> | 
        💻 <a href="https://github.com/Av1352" target="_blank" style="color: #667eea; font-weight: 700; text-decoration: none;">GitHub</a> | 
        🌐 <a href="https://vxanju.com" target="_blank" style="color: #667eea; font-weight: 700; text-decoration: none;">Portfolio</a>
    </p>
    <p style="font-size: 15px; margin: 18px 0 0 0; font-weight: 700; color: #1f2937;">
        <strong>Tech Stack:</strong> Python • Streamlit • Plotly • Agent Tracing • Cost Analytics
    </p>
</div>
""", unsafe_allow_html=True)