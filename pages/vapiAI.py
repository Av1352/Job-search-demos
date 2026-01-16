"""
Vapi AI - Voice AI Platform for Developers
Build voice agents with API-first approach
Built for Vapi AI by Anju Nandhakumar
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="Vapi Voice AI", page_icon="🎙️", layout="wide")

# Voice agent configurations
AGENT_CONFIGS = {
    "Customer Support Agent": {
        "voice": "Friendly Female",
        "language": "English",
        "response_time": 850,
        "accuracy": 0.94,
        "use_cases": ["FAQs", "Order status", "Returns", "Account help"],
        "avg_call_duration": 3.2
    },
    "Appointment Scheduler": {
        "voice": "Professional Male",
        "language": "English",
        "response_time": 650,
        "accuracy": 0.97,
        "use_cases": ["Book appointments", "Reschedule", "Cancellations", "Reminders"],
        "avg_call_duration": 1.8
    },
    "Lead Qualification": {
        "voice": "Energetic Female",
        "language": "English",
        "response_time": 920,
        "accuracy": 0.91,
        "use_cases": ["Qualify leads", "Gather info", "Schedule demos", "Route to sales"],
        "avg_call_duration": 4.5
    },
    "Survey Collection": {
        "voice": "Neutral Male",
        "language": "English",
        "response_time": 720,
        "accuracy": 0.96,
        "use_cases": ["CSAT surveys", "NPS collection", "Feedback", "Market research"],
        "avg_call_duration": 2.4
    }
}

# Sample API code snippets
API_EXAMPLES = {
    "Create Agent": """import vapi

# Initialize Vapi client
client = vapi.Client(api_key="your_api_key")

# Create voice agent
agent = client.agents.create(
    name="Customer Support",
    voice="en-US-Neural2-F",
    model="gpt-4",
    first_message="Hi! How can I help you today?",
    system_prompt="You are a helpful customer support agent...",
    functions=[
        {
            "name": "lookup_order",
            "description": "Look up order status by order number",
            "parameters": {
                "order_number": {"type": "string"}
            }
        }
    ]
)

print(f"Agent created: {agent.id}")""",

    "Make Call": """# Initiate outbound call
call = client.calls.create(
    agent_id="agent_abc123",
    phone_number="+1-555-123-4567",
    customer_data={
        "name": "John Smith",
        "account_id": "ACCT-789"
    }
)

# Call status: "queued" | "ringing" | "in-progress" | "completed"
print(f"Call status: {call.status}")""",

    "Listen to Webhook": """from flask import Flask, request

app = Flask(__name__)

@app.route("/vapi/webhook", methods=["POST"])
def handle_vapi_webhook():
    event = request.json
    
    if event["type"] == "call-started":
        print(f"Call started: {event['call_id']}")
    
    elif event["type"] == "function-call":
        # Handle function execution
        function_name = event["function"]["name"]
        args = event["function"]["arguments"]
        
        # Execute function and return result
        result = execute_function(function_name, args)
        return {"result": result}
    
    elif event["type"] == "call-ended":
        print(f"Call ended. Duration: {event['duration']}s")
        # Log to analytics, update CRM, etc.
    
    return {"status": "ok"}"""
}

def simulate_voice_agent(agent_type, num_calls):
    """Simulate voice agent performance"""
    
    agent = AGENT_CONFIGS[agent_type]
    
    # Generate call results
    successful_calls = int(num_calls * agent['accuracy'])
    failed_calls = num_calls - successful_calls
    
    total_duration = num_calls * agent['avg_call_duration']
    total_cost = num_calls * 0.08
    
    # Build use case badges
    use_case_badges = []
    for uc in agent['use_cases']:
        badge_html = f'<span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 8px 16px; border-radius: 16px; font-size: 13px; font-weight: 700; box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);">{uc}</span>'
        use_case_badges.append(badge_html)
    
    # Build complete HTML
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🎙️ Voice Agent Performance</h2>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Calls</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{num_calls:,}</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Success Rate</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{agent['accuracy']:.0%}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{successful_calls:,} successful</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Avg Duration</p>
                <p style="font-size: 40px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{agent['avg_call_duration']:.1f}m</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Avg Latency</p>
                <p style="font-size: 40px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{agent['response_time']}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">milliseconds</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Cost</p>
                <p style="font-size: 40px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${total_cost:.2f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">$0.08 per call</p>
            </div>
        </div>
    </div>    
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🤖 Agent Configuration</h3>
        <div style="background: white; border-radius: 14px; padding: 22px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 15px;">
                <div style="background: #f9fafb; border-radius: 8px; padding: 12px;">
                    <p style="font-size: 12px; color: #6b7280; margin: 0 0 4px 0;">Voice Type</p>
                    <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0;">{agent['voice']}</p>
                </div>
                <div style="background: #f9fafb; border-radius: 8px; padding: 12px;">
                    <p style="font-size: 12px; color: #6b7280; margin: 0 0 4px 0;">Language</p>
                    <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0;">{agent['language']}</p>
                </div>
                <div style="background: #f9fafb; border-radius: 8px; padding: 12px;">
                    <p style="font-size: 12px; color: #6b7280; margin: 0 0 4px 0;">Response Time</p>
                    <p style="font-size: 16px; color: #3b82f6; font-weight: 700; margin: 0;">{agent['response_time']}ms</p>
                </div>
            </div>
            <h4 style="color: #1f2937; font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">Use Cases:</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                {''.join(use_case_badges)}
            </div>
        </div>
    </div>    
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2);">
        <h3 style="color: #065f46; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">📊 Performance Breakdown</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: white; border-radius: 12px; padding: 20px;">
                <h4 style="color: #1f2937; font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">✅ Successful Calls</h4>
                <p style="font-size: 42px; color: #10b981; font-weight: 900; margin: 0;">{successful_calls:,}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0;">Tasks completed successfully</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 20px;">
                <h4 style="color: #1f2937; font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">❌ Failed Calls</h4>
                <p style="font-size: 42px; color: #ef4444; font-weight: 900; margin: 0;">{failed_calls}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0;">Required human escalation</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 20px;">
                <h4 style="color: #1f2937; font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">⏱️ Total Call Time</h4>
                <p style="font-size: 42px; color: #3b82f6; font-weight: 900; margin: 0;">{total_duration:.0f}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0;">minutes of conversations</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 20px;">
                <h4 style="color: #1f2937; font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">💰 Total Cost</h4>
                <p style="font-size: 42px; color: #f59e0b; font-weight: 900; margin: 0;">${total_cost:.2f}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0;">vs ${num_calls * 2.5:.2f} human cost</p>
            </div>
        </div>
    </div>
    """
    
    # Create charts
    hours = list(range(24))
    success_rates = [agent['accuracy'] + random.uniform(-0.03, 0.03) for _ in hours]
    
    fig_success = go.Figure()
    
    fig_success.add_trace(go.Scatter(
        x=hours,
        y=success_rates,
        mode='lines+markers',
        line=dict(color='#10b981', width=3),
        marker=dict(size=6),
        fill='tonexty',
        fillcolor='rgba(16, 185, 129, 0.1)',
        name='Success Rate'
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
    
    latencies = [agent['response_time'] + random.gauss(0, 100) for _ in range(100)]
    
    fig_latency = go.Figure(data=[go.Histogram(
        x=latencies,
        nbinsx=20,
        marker_color='#3b82f6',
        name='Response Time'
    )])
    
    fig_latency.update_layout(
        title="Response Time Distribution",
        xaxis_title="Latency (ms)",
        yaxis_title="Frequency",
        height=400
    )
    
    return summary_html, fig_success, fig_latency

def show_api_documentation(endpoint):
    """Show API code examples"""
    
    code = API_EXAMPLES.get(endpoint, "")
    
    doc_html = f"""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(168, 85, 247, 0.2); margin-bottom: 25px;">
        <h3 style="color: #6b21a8; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📚 API Example: {endpoint}</h3>
        <div style="background: #1f2937; border-radius: 12px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); font-family: 'Courier New', monospace; color: #d1d5db; font-size: 13px; line-height: 1.6; overflow-x: auto; white-space: pre;">
{code}
        </div>
        <div style="background: white; border-radius: 12px; padding: 20px; margin-top: 18px;">
            <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 12px 0;">Key Features</h4>
            <ul style="margin: 0; padding-left: 24px; line-height: 2;">
                <li style="color: #6b7280; font-size: 14px;">Simple REST API - integrate in 5 minutes</li>
                <li style="color: #6b7280; font-size: 14px;">Real-time webhooks for call events</li>
                <li style="color: #6b7280; font-size: 14px;">Function calling for custom business logic</li>
                <li style="color: #6b7280; font-size: 14px;">Built-in conversation memory and context</li>
            </ul>
        </div>
    </div>
    """
    
    return doc_html

# Header
st.markdown("""
<div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
    <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
        <span style="font-size: 56px;">🎙️</span>
    </div>
    <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        Vapi Voice AI
    </h1>
    <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Voice AI Platform for Developers</p>
    <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">API-first voice agents • Build in minutes • Deploy anywhere</p>
    <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
        <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Voice AI</span>
        <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">API-First</span>
        <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Developer Tools</span>
        <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Real-Time</span>
    </div>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
        Built for <strong style="color: white;">Vapi AI</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🎙️ Agent Performance", "📚 API Docs"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Voice Agent Simulation</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Test voice agent performance at scale</p>
    </div>
    """, unsafe_allow_html=True)
    
    agent_dropdown = st.selectbox(
        "Agent Type",
        list(AGENT_CONFIGS.keys())
    )
    
    num_calls = st.slider(
        "Number of Calls to Simulate",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )
    
    if st.button("📞 Run Call Simulation", type="primary"):
        simulation_html, success_chart, latency_chart = simulate_voice_agent(agent_dropdown, num_calls)
        
        st.markdown(simulation_html, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(success_chart, use_container_width=True)
        with col2:
            st.plotly_chart(latency_chart, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Vapi API Documentation</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Build voice agents with simple REST API calls</p>
    </div>
    """, unsafe_allow_html=True)
    
    api_dropdown = st.selectbox(
        "Select API Endpoint",
        list(API_EXAMPLES.keys())
    )
    
    if st.button("📖 View Code Example", type="primary"):
        docs_html = show_api_documentation(api_dropdown)
        st.markdown(docs_html, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">

<div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
    <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Vapi AI</h2>    
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
        <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
            <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🚀 Developer First</h4>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                Simple REST API. Build voice agents in 5 minutes with 10 lines of code. No complex SDKs, no vendor lock-in.
            </p>
        </div>        
        <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
            <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ Ultra-Low Latency</h4>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                650-920ms response time. Feels natural, not robotic. Optimized infrastructure for real-time conversations.
            </p>
        </div>        
        <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
            <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Cost Efficient</h4>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                $0.08 per call vs $2.50 human agent. 97% cost reduction. Scale to millions of calls without hiring.
            </p>
        </div>
    </div>    
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
        <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ REST API</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Simple HTTP calls, no SDKs required</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Webhooks</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Get call events instantly</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Function Calling</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Execute custom business logic</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-Language</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Support 20+ languages</p>
            </div>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Vapi AI</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
    </p>
    <div style="margin: 25px 0; padding: 22px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.2);">
        <p style="margin: 8px 0; font-size: 16px;">
            📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: white; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a>
        </p>
        <p style="margin: 8px 0; font-size: 16px;">
            💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">LinkedIn</a> | 
            💻 <a href="https://github.com/Av1352" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">GitHub</a> | 
            🌐 <a href="https://vxanju.com" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">Portfolio</a>
        </p>
    </div>
    <p style="font-size: 15px; margin: 18px 0; font-weight: 700;">
        <strong style="color: white;">Tech Stack:</strong> Python • Streamlit • Voice AI API • Real-Time Analytics
    </p>
    <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
    <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
        Demo showcasing API-first voice AI platform for developers.<br>
        Simple integration • Real-time performance • Developer-friendly • Production-ready
    </p>
</div>
""", unsafe_allow_html=True)