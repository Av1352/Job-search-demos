"""
ThirdLayer - Dex AI Browser Copilot
AI assistant for web browsing, research, and productivity
Built for ThirdLayer by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="ThirdLayer - Dex", page_icon="🌐", layout="wide")

# Copilot features
COPILOT_FEATURES = {
    'Smart Summarization': {'usage': 2847, 'time_saved': '4.2 hrs', 'rating': 4.8},
    'Research Assistant': {'usage': 1923, 'time_saved': '6.5 hrs', 'rating': 4.9},
    'Tab Management': {'usage': 3456, 'time_saved': '2.1 hrs', 'rating': 4.6},
    'Auto-Fill Forms': {'usage': 1234, 'time_saved': '1.8 hrs', 'rating': 4.7},
    'Price Tracking': {'usage': 892, 'time_saved': '3.3 hrs', 'rating': 4.8}
}

# Research topics
RESEARCH_TOPICS = [
    'AI Infrastructure Market Analysis',
    'Competitor Product Features',
    'Customer Pain Points',
    'Industry Trends 2025',
    'Technology Stack Comparison'
]

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #14b8a6 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🌐</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Dex</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">AI Browser Copilot by ThirdLayer</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Smart research • Tab management • Auto-summarization • Productivity boost</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Research Assistant", "📊 Productivity Dashboard", "⚡ Quick Actions", "💡 Features"])

with tab1:
    st.markdown("### AI-Powered Research")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Research Query**")
        
        research_query = st.text_area(
            "What do you want to research?",
            "Find the top 5 AI infrastructure companies, their funding rounds, and key products",
            height=100
        )
        
        st.markdown("**Settings**")
        
        depth = st.select_slider(
            "Research Depth",
            options=["Quick", "Standard", "Deep", "Comprehensive"],
            value="Deep"
        )
        
        sources = st.slider("Number of Sources", 5, 50, 20)
        
        include_citations = st.checkbox("Include Citations", value=True)
        auto_summarize = st.checkbox("Auto-Summarize Findings", value=True)
        
        research_btn = st.button("🔍 Start Research", type="primary", use_container_width=True)
    
    with col2:
        if research_btn:
            st.markdown("**Research Progress**")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            stages = [
                ("Searching web sources...", 0.2),
                ("Analyzing 20 websites...", 0.4),
                ("Extracting key information...", 0.6),
                ("Cross-referencing data...", 0.8),
                ("Generating summary...", 1.0)
            ]
            
            for stage, progress in stages:
                status_text.text(stage)
                progress_bar.progress(progress)
                time.sleep(0.5)
            
            st.success("✅ Research complete!")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #14b8a6 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Research Summary</h3>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                    <p style="color: white; margin: 0; line-height: 1.8; font-size: 15px;">
                        <strong>Top 5 AI Infrastructure Companies:</strong><br><br>
                        1. <strong>Modal</strong> - Serverless ML platform, Series B, $150M raised. Product: GPU-as-a-service.<br>
                        2. <strong>Fireworks AI</strong> - Fast LLM inference, Series C, $250M raised. Product: Optimized inference engine.<br>
                        3. <strong>Anyscale</strong> - Distributed computing, Series C, $260M raised. Product: Ray framework.<br>
                        4. <strong>Weights & Biases</strong> - MLOps platform, Series C, $200M raised. Product: Experiment tracking.<br>
                        5. <strong>Databricks</strong> - Data + AI platform, Public, $38B valuation. Product: Lakehouse architecture.
                    </p>
                </div>
                <div style="margin-top: 15px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 10px; padding: 15px; text-align: center;">
                        <p style="font-size: 13px; color: rgba(255,255,255,0.8); margin: 0 0 6px 0;">Sources</p>
                        <p style="font-size: 20px; color: white; font-weight: 700; margin: 0;">20</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 10px; padding: 15px; text-align: center;">
                        <p style="font-size: 13px; color: rgba(255,255,255,0.8); margin: 0 0 6px 0;">Time</p>
                        <p style="font-size: 20px; color: white; font-weight: 700; margin: 0;">8.5s</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 10px; padding: 15px; text-align: center;">
                        <p style="font-size: 13px; color: rgba(255,255,255,0.8); margin: 0 0 6px 0;">Confidence</p>
                        <p style="font-size: 20px; color: white; font-weight: 700; margin: 0;">96.5%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Time Saved", "45 min", "vs manual research")
            col2.metric("Sources Analyzed", "20", "Cross-verified")
            col3.metric("Accuracy", "96.5%", "High confidence")

with tab2:
    st.markdown("### Productivity Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Daily Active Users", "12,847", "+1,234")
    col2.metric("Time Saved Today", "18.2 hrs", "+2.1 hrs")
    col3.metric("Tasks Automated", "5,423", "+847")
    col4.metric("User Satisfaction", "4.8/5", "+0.3")
    
    st.markdown("**Feature Usage**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        usage_data = []
        for feature, data in COPILOT_FEATURES.items():
            usage_data.append({
                'Feature': feature,
                'Usage': data['usage'],
                'Time Saved': data['time_saved'],
                'Rating': f"{data['rating']}/5"
            })
        
        st.dataframe(pd.DataFrame(usage_data), hide_index=True, use_container_width=True)
    
    with col2:
        fig1 = go.Figure(data=[go.Bar(
            x=list(COPILOT_FEATURES.keys()),
            y=[COPILOT_FEATURES[f]['usage'] for f in COPILOT_FEATURES.keys()],
            marker=dict(color='#14b8a6'),
            text=[COPILOT_FEATURES[f]['usage'] for f in COPILOT_FEATURES.keys()],
            textposition='auto'
        )])
        fig1.update_layout(
            yaxis_title='Daily Usage',
            height=300
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("**Time Saved Analysis**")
    
    # Time saved trend
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours_saved = [16.2, 17.5, 18.2, 16.8, 19.1, 12.4, 10.8]
    
    fig2 = go.Figure(data=[go.Scatter(
        x=days,
        y=hours_saved,
        mode='lines+markers',
        line=dict(color='#14b8a6', width=3),
        fill='tozeroy',
        fillcolor='rgba(20, 184, 166, 0.1)'
    )])
    fig2.update_layout(
        xaxis_title='Day',
        yaxis_title='Hours Saved',
        height=250
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Quick Copilot Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Instant Actions**")
        
        if st.button("📝 Summarize This Page", use_container_width=True):
            st.success("✅ Page summarized in 1.2s")
            st.info("**Summary:** This page demonstrates Dex's AI browser copilot capabilities including research assistance, tab management, and productivity features.")
        
        if st.button("🔍 Research This Topic", use_container_width=True):
            st.success("✅ Research complete - 12 sources analyzed")
        
        if st.button("📋 Extract Data from Page", use_container_width=True):
            st.success("✅ Extracted 47 data points")
        
        if st.button("💾 Save All Tabs", use_container_width=True):
            st.success("✅ Saved 15 tabs to reading list")
        
        if st.button("🎯 Find Similar Content", use_container_width=True):
            st.success("✅ Found 8 related articles")
    
    with col2:
        st.markdown("**Tab Management**")
        
        open_tabs = {
            'Tab': ['Gmail', 'GitHub PR #234', 'Stack Overflow', 'LinkedIn', 'Google Docs', 'Slack'],
            'Category': ['Email', 'Work', 'Development', 'Social', 'Work', 'Work'],
            'Time Open': ['2.5 hrs', '45 min', '1.2 hrs', '15 min', '3.1 hrs', '1.8 hrs']
        }
        st.dataframe(pd.DataFrame(open_tabs), hide_index=True, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📁 Group by Category", use_container_width=True):
                st.success("✅ Tabs grouped into 4 categories")
        with col2:
            if st.button("🗑️ Close Inactive Tabs", use_container_width=True):
                st.success("✅ Closed 3 inactive tabs")

with tab4:
    st.markdown("### Copilot Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Research & Analysis**")
        st.markdown("""
        - ✅ Multi-source research (20+ sites)
        - ✅ Automatic summarization
        - ✅ Citation management
        - ✅ Fact verification
        - ✅ Competitive intelligence
        - ✅ Trend analysis
        """)
        
        st.markdown("**Productivity Tools**")
        st.markdown("""
        - ✅ Tab organization & grouping
        - ✅ Reading list management
        - ✅ Bookmark intelligence
        - ✅ Session recovery
        - ✅ Distraction blocking
        - ✅ Focus mode
        """)
    
    with col2:
        st.markdown("**Smart Automation**")
        st.markdown("""
        - ✅ Form auto-fill
        - ✅ Price tracking & alerts
        - ✅ Meeting scheduling
        - ✅ Email drafting
        - ✅ Data extraction
        - ✅ Screenshot & annotation
        """)
        
        st.markdown("**Privacy & Security**")
        st.markdown("""
        - ✅ Local processing (no cloud)
        - ✅ Encrypted storage
        - ✅ Ad blocking
        - ✅ Tracker prevention
        - ✅ Password manager integration
        - ✅ GDPR compliant
        """)
    
    st.markdown("**Performance Metrics**")
    
    metrics = {
        'Metric': ['Time Saved/Day', 'Tasks Automated', 'Pages Summarized', 'Research Queries', 'Forms Auto-Filled'],
        'Value': ['18.2 hours', '5,423', '2,847', '1,923', '1,234'],
        'Trend': ['+12%', '+23%', '+18%', '+15%', '+8%']
    }
    st.dataframe(pd.DataFrame(metrics), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #ccfbf1 0%, #99f6e4 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #134e4a; font-size: 24px; font-weight: 800;">💡 System Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0d9488; font-weight: 700; margin: 0 0 6px 0;">✓ AI Research</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Multi-source analysis in 8.5s</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0d9488; font-weight: 700; margin: 0 0 6px 0;">✓ 18.2 Hours Saved</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Per day per user</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0d9488; font-weight: 700; margin: 0 0 6px 0;">✓ 4.8/5 Rating</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">High user satisfaction</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0d9488; font-weight: 700; margin: 0 0 6px 0;">✓ 5,423 Tasks/Day</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Automated productivity</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #14b8a6 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for ThirdLayer</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)