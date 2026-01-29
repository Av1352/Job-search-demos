"""
Weave - AI to Understand Engineering Work
Analyze code, track velocity, understand engineering workflows
Built for Weave by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Weave", page_icon="🔧", layout="wide")

# Engineering metrics
VELOCITY_METRICS = {
    'Code Quality': 94.2,
    'Review Thoroughness': 91.5,
    'Deployment Frequency': 88.7,
    'MTTR': 96.3,
    'Test Coverage': 89.4
}

# Code complexity analysis
CODE_METRICS = {
    'Total Lines': 45823,
    'Functions': 1247,
    'Classes': 389,
    'Complexity Score': 7.2,
    'Technical Debt': 23,  # days
    'Test Coverage': 84.5
}

# Team members
TEAM_MEMBERS = {
    'Alice Chen': {'commits': 234, 'prs': 45, 'reviews': 67, 'velocity': 92.3},
    'Bob Smith': {'commits': 189, 'prs': 38, 'reviews': 52, 'velocity': 88.5},
    'Carol Lee': {'commits': 267, 'prs': 52, 'reviews': 71, 'velocity': 95.1},
    'David Park': {'commits': 156, 'prs': 31, 'reviews': 43, 'velocity': 85.2}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #10b981 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🔧</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Weave</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">AI to Understand Engineering Work</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Code analysis • Velocity tracking • Team insights • Engineering intelligence</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Team Velocity", "🔍 Code Analysis", "👥 Engineer Insights", "💡 Platform Features"])

with tab1:
    st.markdown("### Engineering Velocity Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sprint Velocity", "92.3", "+4.2")
    col2.metric("PRs This Week", "166", "+23")
    col3.metric("Code Reviews", "233", "+31")
    col4.metric("Deployment Freq", "12/day", "+2")
    
    st.markdown("**Velocity Trends**")
    
    # Generate velocity trend
    weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
    velocity = [85.3, 88.7, 90.5, 92.3]
    commits = [623, 687, 742, 846]
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=weeks, y=velocity,
        mode='lines+markers',
        name='Velocity Score',
        line=dict(color='#10b981', width=3),
        yaxis='y'
    ))
    fig1.add_trace(go.Bar(
        x=weeks, y=commits,
        name='Total Commits',
        marker=dict(color='rgba(16, 185, 129, 0.3)'),
        yaxis='y2'
    ))
    
    fig1.update_layout(
        yaxis=dict(title='Velocity Score', side='left', range=[80, 100]),
        yaxis2=dict(title='Commits', side='right', overlaying='y'),
        height=300
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Engineering Metrics**")
        
        metrics_data = []
        for metric, value in VELOCITY_METRICS.items():
            metrics_data.append({
                'Metric': metric,
                'Score': f"{value}%",
                'Status': '✅ Excellent' if value > 90 else '✅ Good'
            })
        
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Metric Distribution**")
        
        fig2 = go.Figure(data=[go.Bar(
            x=list(VELOCITY_METRICS.keys()),
            y=list(VELOCITY_METRICS.values()),
            marker=dict(color='#10b981'),
            text=[f"{v}%" for v in VELOCITY_METRICS.values()],
            textposition='auto'
        )])
        fig2.update_layout(yaxis=dict(range=[80, 100]), height=250)
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown("### Codebase Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Code Metrics**")
        
        code_data = []
        for metric, value in CODE_METRICS.items():
            if 'Coverage' in metric or 'Score' in metric:
                display = f"{value}%" if 'Coverage' in metric else f"{value}/10"
            elif 'Debt' in metric:
                display = f"{value} days"
            else:
                display = f"{value:,}"
            
            code_data.append({
                'Metric': metric,
                'Value': display
            })
        
        st.dataframe(pd.DataFrame(code_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Complexity Analysis**")
        
        complexity = {
            'File': ['auth.py', 'models.py', 'api.py', 'utils.py', 'handlers.py'],
            'Lines': [847, 1234, 923, 456, 712],
            'Complexity': [8.2, 9.5, 7.3, 4.1, 6.8],
            'Status': ['⚠️ High', '⚠️ High', '✅ Good', '✅ Good', '✅ Good']
        }
        st.dataframe(pd.DataFrame(complexity), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Technical Debt Trends**")
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
        debt_days = [35, 31, 28, 25, 23]
        
        fig3 = go.Figure(data=[go.Scatter(
            x=months, y=debt_days,
            mode='lines+markers',
            line=dict(color='#10b981', width=3),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)'
        )])
        fig3.update_layout(
            yaxis_title='Technical Debt (days)',
            height=200
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("**Code Quality Score**")
        
        quality_breakdown = {
            'Category': ['Maintainability', 'Reliability', 'Security', 'Performance', 'Testability'],
            'Score': [92, 88, 96, 89, 85],
            'Grade': ['A', 'B+', 'A+', 'B+', 'B']
        }
        st.dataframe(pd.DataFrame(quality_breakdown), hide_index=True, use_container_width=True)

with tab3:
    st.markdown("### Individual Engineer Insights")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Team Performance**")
        
        team_data = []
        for engineer, data in TEAM_MEMBERS.items():
            team_data.append({
                'Engineer': engineer,
                'Commits': data['commits'],
                'PRs': data['prs'],
                'Reviews': data['reviews'],
                'Velocity': f"{data['velocity']:.1f}"
            })
        
        st.dataframe(pd.DataFrame(team_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Velocity Comparison**")
        
        fig4 = go.Figure(data=[go.Bar(
            x=list(TEAM_MEMBERS.keys()),
            y=[TEAM_MEMBERS[e]['velocity'] for e in TEAM_MEMBERS.keys()],
            marker=dict(
                color=[TEAM_MEMBERS[e]['velocity'] for e in TEAM_MEMBERS.keys()],
                colorscale='RdYlGn',
                cmin=80,
                cmax=100
            ),
            text=[f"{TEAM_MEMBERS[e]['velocity']:.1f}" for e in TEAM_MEMBERS.keys()],
            textposition='auto'
        )])
        fig4.update_layout(
            yaxis=dict(range=[80, 100]),
            yaxis_title='Velocity Score',
            height=250
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("**Activity Heatmap**")
    
    # Generate activity data
    engineers = list(TEAM_MEMBERS.keys())
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    
    activity_matrix = np.random.randint(10, 50, size=(len(engineers), len(days)))
    
    fig5 = go.Figure(data=go.Heatmap(
        z=activity_matrix,
        x=days,
        y=engineers,
        colorscale='Greens',
        text=activity_matrix,
        texttemplate='%{text}',
        textfont={"size": 12}
    ))
    fig5.update_layout(height=250)
    st.plotly_chart(fig5, use_container_width=True)

with tab4:
    st.markdown("### Platform Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Code Intelligence**")
        st.markdown("""
        - ✅ Codebase analysis (complexity, quality)
        - ✅ Technical debt tracking
        - ✅ Dependency mapping
        - ✅ Code review insights
        - ✅ Refactoring suggestions
        - ✅ Pattern detection
        """)
        
        st.markdown("**Velocity Tracking**")
        st.markdown("""
        - ✅ Sprint velocity calculation
        - ✅ Cycle time analysis
        - ✅ Deployment frequency
        - ✅ MTTR (Mean Time to Resolve)
        - ✅ Lead time for changes
        - ✅ Change failure rate
        """)
    
    with col2:
        st.markdown("**Team Analytics**")
        st.markdown("""
        - ✅ Individual productivity metrics
        - ✅ Collaboration patterns
        - ✅ Code ownership mapping
        - ✅ Review load balancing
        - ✅ Skill gap identification
        - ✅ Mentorship tracking
        """)
        
        st.markdown("**Integration**")
        st.markdown("""
        - ✅ GitHub, GitLab, Bitbucket
        - ✅ Jira, Linear, Asana
        - ✅ Slack, Discord
        - ✅ CI/CD pipelines
        - ✅ Cloud platforms
        - ✅ Custom webhooks
        """)
    
    st.markdown("**AI Features**")
    
    ai_features = {
        'Feature': ['Code Review AI', 'Bug Prediction', 'Refactor Suggestions', 'Dependency Analysis', 'Performance Insights'],
        'Accuracy': ['94.2%', '87.5%', '91.3%', '96.8%', '89.7%'],
        'Usage': ['2,340/week', '1,567/week', '892/week', '3,456/week', '1,234/week']
    }
    st.dataframe(pd.DataFrame(ai_features), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #065f46; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Code Intelligence</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Analyze 45K+ lines automatically</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ 94.2% Accuracy</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">AI code review quality</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Velocity Tracking</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Sprint & cycle time analysis</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Team Insights</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Productivity & collaboration</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #10b981 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Weave</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)