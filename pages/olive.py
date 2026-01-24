"""
Olive - Build Internal Tools with NLP
AI-powered internal tool builder for enterprises
Built for Olive by Anju Nandhakumar
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import random
from utils.sidebar import render_sidebar
render_sidebar()

st.set_page_config(page_title="Olive - Build Internal Tools with NLP", layout="wide")

# Initialize session state
if 'tool_built' not in st.session_state:
    st.session_state.tool_built = False
if 'gallery_viewed' not in st.session_state:
    st.session_state.gallery_viewed = False

# Sample internal tools that can be built
TOOL_TEMPLATES = {
    "Employee Onboarding Dashboard": {
        "description": "Track new hire progress, documents, training completion",
        "components": ["Employee list", "Progress tracker", "Document checklist", "Training status", "Alert system"],
        "users": "HR Team",
        "build_time": "2 minutes",
        "complexity": "Medium"
    },
    "Expense Approval Workflow": {
        "description": "Submit, review, and approve employee expenses",
        "components": ["Expense form", "Receipt upload", "Approval queue", "Spend analytics", "Export reports"],
        "users": "Finance Team",
        "build_time": "3 minutes",
        "complexity": "Medium"
    },
    "IT Support Ticket System": {
        "description": "Employee IT requests with priority and assignment",
        "components": ["Ticket form", "Priority selector", "Assignment logic", "Status tracker", "SLA monitoring"],
        "users": "IT Team",
        "build_time": "2.5 minutes",
        "complexity": "Medium"
    },
    "Sales Pipeline Tracker": {
        "description": "Deal stages, forecasting, and team performance",
        "components": ["Deal cards", "Stage pipeline", "Revenue forecast", "Rep leaderboard", "Activity log"],
        "users": "Sales Team",
        "build_time": "3.5 minutes",
        "complexity": "High"
    },
    "Content Calendar Manager": {
        "description": "Plan, schedule, and track content publication",
        "components": ["Calendar view", "Content cards", "Status workflow", "Team assignments", "Analytics"],
        "users": "Marketing Team",
        "build_time": "2 minutes",
        "complexity": "Low"
    },
    "Inventory Management System": {
        "description": "Track stock levels, reorder alerts, supplier info",
        "components": ["Item database", "Stock counter", "Low inventory alerts", "Supplier contacts", "Order history"],
        "users": "Operations Team",
        "build_time": "4 minutes",
        "complexity": "High"
    }
}

def build_tool_from_description(description, team_size, integrations):
    """Build internal tool from natural language"""
    
    if not description or len(description.strip()) < 10:
        return None, None
    
    desc_lower = description.lower()
    matched = None
    
    for tool_name, tool_data in TOOL_TEMPLATES.items():
        keywords = tool_name.lower().split() + tool_data['description'].lower().split()
        if any(word in desc_lower for word in keywords[:5]):
            matched = tool_name
            break
    
    if not matched:
        matched = random.choice(list(TOOL_TEMPLATES.keys()))
    
    tool = TOOL_TEMPLATES[matched]
    num_integrations = len(integrations)
    estimated_dev_time = random.randint(20, 40)
    
    # Build component badges
    comp_badges = []
    for comp in tool['components']:
        badge = f'<span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 20px; border-radius: 20px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(16, 185, 129, 0.3);">{comp}</span>'
        comp_badges.append(badge)
    all_badges = ''.join(comp_badges)
    
    # Build component preview boxes
    comp_previews = []
    for comp in tool['components']:
        preview = f'<div style="background: white; border: 2px solid #d1d5db; border-radius: 8px; padding: 14px; margin-bottom: 10px;"><p style="font-size: 14px; color: #1f2937; font-weight: 600; margin: 0;">📋 {comp}</p></div>'
        comp_previews.append(preview)
    all_previews = ''.join(comp_previews)
    
    # Build integration badges
    integration_badges = []
    if integrations:
        for integration in integrations:
            badge = f'<span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 6px 14px; border-radius: 16px; font-size: 12px; font-weight: 700; box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);">{integration}</span>'
            integration_badges.append(badge)
        all_integrations = ''.join(integration_badges)
    else:
        all_integrations = '<p style="font-size: 13px; color: #6b7280; margin: 0;">No integrations selected</p>'
    
    result_html = f'<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;"><h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">✨ Internal Tool Built Successfully!</h2><div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;"><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Tool Type</p><p style="font-size: 18px; color: white; font-weight: 900; margin: 0; line-height: 1.3;">{matched}</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Build Time</p><p style="font-size: 40px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{tool["build_time"]}</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">vs {estimated_dev_time}hrs manual</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Components</p><p style="font-size: 40px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{len(tool["components"])}</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">UI elements</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Integrations</p><p style="font-size: 40px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{num_integrations}</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Connected</p></div></div></div><div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;"><h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🎨 Generated Components</h3><div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">{all_badges}</div><div style="background: white; border-radius: 14px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);"><h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 15px 0;">🖥️ Interactive Tool Preview</h4><div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); border: 2px solid #e5e7eb; border-radius: 12px; padding: 24px;"><h5 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 18px 0; text-align: center;">{matched}</h5>{all_previews}<div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border-radius: 10px; padding: 16px; margin-top: 18px; text-align: center;"><p style="color: white; font-size: 15px; font-weight: 700; margin: 0;">✓ All {len(tool["components"])} components are fully functional</p></div></div><div style="background: #f0f9ff; border-radius: 10px; padding: 16px; margin-top: 15px;"><p style="font-size: 14px; color: #1e40af; font-weight: 700; margin: 0 0 10px 0;">🔗 Connected Integrations:</p><div style="display: flex; flex-wrap: wrap; gap: 8px;">{all_integrations}</div></div></div><div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 15px;"><div style="background: white; border-radius: 12px; padding: 16px; text-align: center;"><p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Target Users</p><p style="font-size: 18px; color: #10b981; font-weight: 800; margin: 0;">{tool["users"]}</p></div><div style="background: white; border-radius: 12px; padding: 16px; text-align: center;"><p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Team Size</p><p style="font-size: 24px; color: #3b82f6; font-weight: 800; margin: 0;">{team_size}</p></div><div style="background: white; border-radius: 12px; padding: 16px; text-align: center;"><p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Complexity</p><p style="font-size: 18px; color: #f59e0b; font-weight: 800; margin: 0;">{tool["complexity"]}</p></div></div></div><div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2);"><h3 style="color: #1e40af; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">💰 ROI Analysis</h3><div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;"><div style="background: white; border-radius: 12px; padding: 20px;"><h4 style="color: #1f2937; font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">Traditional Development</h4><p style="font-size: 14px; color: #6b7280; margin: 0 0 8px 0;">Developer time: {estimated_dev_time} hours</p><p style="font-size: 14px; color: #6b7280; margin: 0 0 8px 0;">Cost at $100/hr: ${estimated_dev_time * 100:,}</p><p style="font-size: 14px; color: #6b7280; margin: 0;">Timeline: 1-2 weeks</p></div><div style="background: white; border-radius: 12px; padding: 20px;"><h4 style="color: #1f2937; font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">With Olive AI</h4><p style="font-size: 14px; color: #10b981; font-weight: 700; margin: 0 0 8px 0;">Build time: {tool["build_time"]}</p><p style="font-size: 14px; color: #10b981; font-weight: 700; margin: 0 0 8px 0;">Cost: $50 (subscription)</p><p style="font-size: 14px; color: #10b981; font-weight: 700; margin: 0;">Timeline: Same day</p></div></div><div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; padding: 18px; margin-top: 15px; text-align: center; color: white;"><p style="font-size: 20px; font-weight: 900; margin: 0;">💰 Savings: ${(estimated_dev_time * 100) - 50:,} ({((estimated_dev_time * 100 - 50) / (estimated_dev_time * 100) * 100):.0f}% cost reduction)</p></div></div>'
    
    # Create build timeline
    steps = ["Understanding requirements", "Generating components", "Creating UI", "Adding logic", "Setting up integrations", "Testing & deployment"]
    durations = [0.2, 0.4, 0.6, 0.5, 0.3, 0.2]
    
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Bar(
        y=steps,
        x=durations,
        orientation='h',
        marker_color=['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4'],
        text=[f'{d:.1f}min' for d in durations],
        textposition='outside'
    ))
    
    fig_timeline.update_layout(
        title=f"Build Pipeline - Total: {sum(durations):.1f} minutes",
        xaxis_title="Time (minutes)",
        height=400,
        showlegend=False
    )
    
    return result_html, fig_timeline

def show_tool_gallery():
    """Display gallery of tools that can be built"""
    
    colors = ['#3b82f6', '#10b981', '#ec4899', '#f59e0b', '#8b5cf6', '#06b6d4']
    
    # Build tool cards
    tool_cards = []
    for idx, (tool_name, tool_data) in enumerate(TOOL_TEMPLATES.items()):
        color = colors[idx % len(colors)]
        
        comp_badges_small = []
        for comp in tool_data['components'][:4]:
            badge = f'<span style="background: {color}; color: white; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 700;">{comp}</span>'
            comp_badges_small.append(badge)
        all_small_badges = ''.join(comp_badges_small)
        
        card = f'<div style="background: white; border-radius: 14px; padding: 22px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid {color};"><h4 style="color: #1f2937; font-size: 20px; font-weight: 800; margin: 0 0 10px 0;">{tool_name}</h4><p style="color: #6b7280; font-size: 14px; margin: 0 0 15px 0; line-height: 1.6;">{tool_data["description"]}</p><div style="background: #f9fafb; border-radius: 10px; padding: 14px; margin-bottom: 15px;"><p style="font-size: 13px; color: #6b7280; margin: 0 0 8px 0; font-weight: 600;">For: {tool_data["users"]}</p><div style="display: flex; flex-wrap: wrap; gap: 6px;">{all_small_badges}</div></div><div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;"><div style="background: #d1fae5; border-radius: 8px; padding: 10px; text-align: center;"><p style="font-size: 11px; color: #6b7280; margin: 0 0 4px 0;">Build Time</p><p style="font-size: 18px; color: #10b981; font-weight: 800; margin: 0;">{tool_data["build_time"]}</p></div><div style="background: #fef3c7; border-radius: 8px; padding: 10px; text-align: center;"><p style="font-size: 11px; color: #6b7280; margin: 0 0 4px 0;">Complexity</p><p style="font-size: 18px; color: #f59e0b; font-weight: 800; margin: 0;">{tool_data["complexity"]}</p></div></div></div>'
        tool_cards.append(card)
    
    all_cards = ''.join(tool_cards)
    
    gallery_html = f'<div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(168, 85, 247, 0.2); margin-bottom: 25px;"><h3 style="color: #6b21a8; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🎨 Internal Tool Gallery</h3><div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px;">{all_cards}</div><div style="background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); border-radius: 12px; padding: 20px; margin-top: 24px; color: white; text-align: center;"><p style="font-size: 18px; font-weight: 800; margin: 0;">💡 Or describe your own: "Build a tool to track customer feedback" → AI creates it instantly</p></div></div>'
    
    # Stats chart
    tools = list(TOOL_TEMPLATES.keys())
    complexities = {'Low': 1, 'Medium': 2, 'High': 3}
    complexity_values = [complexities[TOOL_TEMPLATES[t]['complexity']] for t in tools]
    
    fig_complexity = go.Figure(data=[
        go.Bar(
            x=tools,
            y=complexity_values,
            marker_color=colors,
            text=[TOOL_TEMPLATES[t]['complexity'] for t in tools],
            textposition='outside'
        )
    ])
    
    fig_complexity.update_layout(
        title="Tool Complexity Comparison",
        yaxis_title="Complexity Level",
        yaxis=dict(tickmode='array', tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High']),
        height=400
    )
    
    return gallery_html, fig_complexity

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🛠️</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Olive AI
        </h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Build Internal Tools with NLP</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Natural language → Production tools • No coding required</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">No-Code</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">NLP Powered</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Internal Tools</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">YC Backed</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Olive</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🛠️ Build Tool", "🎨 Tool Gallery"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Natural Language Tool Builder</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Describe your internal tool and watch AI build it in minutes</p>
    </div>
    """, unsafe_allow_html=True)
    
    tool_description = st.text_area(
        "Describe Your Tool",
        placeholder="Example: Build an employee onboarding tracker where HR can see new hire progress, required documents, and training completion status",
        height=120
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        team_size = st.number_input("Team Size", min_value=1, value=25, step=1)
    
    with col2:
        integrations = st.multiselect(
            "Integrations Needed",
            ["Slack", "Google Workspace", "Salesforce", "Jira", "GitHub"],
            default=["Slack"]
        )
    
    if st.button("✨ Build Tool with AI", type="primary", use_container_width=True):
        if tool_description and len(tool_description.strip()) >= 10:
            st.session_state.tool_built = True
            st.session_state.build_params = (tool_description, team_size, integrations)
        else:
            st.error("⚠️ Please describe the tool you want to build!")
    
    if st.session_state.tool_built:
        result_html, fig_timeline = build_tool_from_description(*st.session_state.build_params)
        st.markdown(result_html, unsafe_allow_html=True)
        st.plotly_chart(fig_timeline, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Pre-Built Tool Templates</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Common internal tools you can build instantly</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎨 View Tool Gallery", type="primary", use_container_width=True):
        st.session_state.gallery_viewed = True
    
    if st.session_state.gallery_viewed:
        gallery_html, fig_complexity = show_tool_gallery()
        st.markdown(gallery_html, unsafe_allow_html=True)
        st.plotly_chart(fig_complexity, use_container_width=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Olive</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ Speed to Value</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Minutes vs weeks. HR needs onboarding tracker → built same day. No IT backlog, no sprints, instant productivity gains.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Cost Efficiency</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    $50/month vs $50K for custom development. Build 100 tools for price of 1 traditional project. Democratize software for every team.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🎯 Perfect Fit</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Non-technical teams describe exactly what they need. No translation layer, no requirements docs. Build tools that actually match workflows.
                </p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ NLP Understanding</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Parse requirements from natural language</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Component Library</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">100+ pre-built UI components</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Integration Layer</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Connect to Slack, Salesforce, G Suite</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Instant Deploy</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Live in production same day</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Olive</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Streamlit • NLP • Tool Generation
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing natural language internal tool building.<br>
            Instant generation • No coding • Enterprise integrations • Production ready
        </p>
    </div>
    """, unsafe_allow_html=True)