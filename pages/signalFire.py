"""
Signal Fire - AI Investment Intelligence Platform
VC analytics and deal sourcing powered by AI
Built for Signal Fire by Anju Nandhakumar
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

st.set_page_config(page_title="Signal Fire Intelligence", layout="wide")

# Initialize session state
if 'startup_analyzed' not in st.session_state:
    st.session_state.startup_analyzed = False
if 'portfolio_loaded' not in st.session_state:
    st.session_state.portfolio_loaded = False

# Sample startup database
SAMPLE_STARTUPS = {
    "HealthTech AI": {
        "sector": "Healthcare",
        "stage": "Series A",
        "arr": 2.5,
        "growth_rate": 185,
        "burn_rate": 200,
        "runway": 18,
        "team_size": 25,
        "founded": "2023",
        "valuation": 25,
        "previous_raise": 5,
        "score": 87
    },
    "FinTech Platform": {
        "sector": "Fintech",
        "stage": "Seed",
        "arr": 0.8,
        "growth_rate": 320,
        "burn_rate": 150,
        "runway": 14,
        "team_size": 12,
        "founded": "2024",
        "valuation": 12,
        "previous_raise": 2,
        "score": 92
    },
    "DevTools Startup": {
        "sector": "Developer Tools",
        "stage": "Series A",
        "arr": 4.2,
        "growth_rate": 140,
        "burn_rate": 280,
        "runway": 22,
        "team_size": 35,
        "founded": "2022",
        "valuation": 40,
        "previous_raise": 10,
        "score": 78
    },
    "AI Infrastructure": {
        "sector": "AI/ML",
        "stage": "Seed",
        "arr": 1.2,
        "growth_rate": 280,
        "burn_rate": 180,
        "runway": 16,
        "team_size": 18,
        "founded": "2023",
        "valuation": 18,
        "previous_raise": 3.5,
        "score": 85
    },
    "EdTech Platform": {
        "sector": "Education",
        "stage": "Series B",
        "arr": 8.5,
        "growth_rate": 95,
        "burn_rate": 450,
        "runway": 24,
        "team_size": 65,
        "founded": "2021",
        "valuation": 85,
        "previous_raise": 25,
        "score": 81
    }
}

def analyze_startup(startup_name):
    """Analyze a startup for investment potential"""
    
    startup = SAMPLE_STARTUPS[startup_name]
    
    ltv_cac = random.uniform(3.5, 6.0)
    magic_number = startup['arr'] / (startup['previous_raise'] / 2)
    months_to_profitability = startup['burn_rate'] / (startup['arr'] * 1000 / 12) if startup['arr'] > 0 else 999
    
    scores = {
        "Growth Rate": min(startup['growth_rate'] / 3, 100),
        "Unit Economics": ltv_cac * 15,
        "Market Size": random.uniform(75, 95),
        "Team Quality": random.uniform(80, 95),
        "Product-Market Fit": startup['score'],
        "Capital Efficiency": min(magic_number * 25, 100)
    }
    
    avg_score = sum(scores.values()) / len(scores)
    
    if avg_score >= 85:
        recommendation = "STRONG BUY"
        rec_color = "#10b981"
        rec_emoji = "🚀"
    elif avg_score >= 70:
        recommendation = "BUY"
        rec_color = "#3b82f6"
        rec_emoji = "✓"
    elif avg_score >= 55:
        recommendation = "HOLD"
        rec_color = "#f59e0b"
        rec_emoji = "⚠️"
    else:
        recommendation = "PASS"
        rec_color = "#ef4444"
        rec_emoji = "✗"
    
    # Create charts
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=list(scores.values()),
        theta=list(scores.keys()),
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#3b82f6', width=3)
    ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Investment Score Breakdown",
        height=500
    )
    
    months = list(range(12))
    current_arr = startup['arr']
    monthly_growth = (1 + startup['growth_rate']/100) ** (1/12) - 1
    
    projected_arr = [current_arr * ((1 + monthly_growth) ** m) for m in months]
    
    fig_revenue = go.Figure()
    
    fig_revenue.add_trace(go.Scatter(
        x=months,
        y=projected_arr,
        mode='lines+markers',
        line=dict(color='#10b981', width=3),
        marker=dict(size=6),
        fill='tonexty',
        fillcolor='rgba(16, 185, 129, 0.1)',
        name='Projected ARR'
    ))
    
    fig_revenue.update_layout(
        title="12-Month ARR Projection",
        xaxis_title="Months from Now",
        yaxis_title="ARR ($M)",
        height=400
    )
    
    # Build strengths and concerns
    strengths = []
    concerns = []
    
    if startup['growth_rate'] > 200:
        strengths.append(f"Exceptional {startup['growth_rate']}% YoY growth - top decile for stage")
    elif startup['growth_rate'] > 100:
        strengths.append(f"Strong {startup['growth_rate']}% growth trajectory")
    
    if ltv_cac >= 4:
        strengths.append(f"Excellent unit economics (LTV:CAC = {ltv_cac:.1f}x)")
    
    if startup['runway'] >= 18:
        strengths.append(f"Healthy {startup['runway']}-month runway provides execution flexibility")
    elif startup['runway'] < 12:
        concerns.append(f"Short {startup['runway']}-month runway - may need bridge round")
    
    if magic_number > 1.0:
        strengths.append(f"Capital efficient (Magic Number: {magic_number:.2f})")
    else:
        concerns.append(f"Capital efficiency below 1.0 - burning more than generating")
    
    if startup['sector'] in ['AI/ML', 'Healthcare', 'Fintech']:
        strengths.append(f"{startup['sector']} sector tailwinds - high growth category")
    
    return startup, avg_score, recommendation, rec_color, rec_emoji, ltv_cac, magic_number, scores, strengths, concerns, fig_radar, fig_revenue

def generate_portfolio_dashboard():
    """Generate portfolio-wide analytics"""
    
    # Create charts
    sectors = [s['sector'] for s in SAMPLE_STARTUPS.values()]
    sector_counts = pd.Series(sectors).value_counts()
    
    colors = ['#3b82f6', '#10b981', '#ec4899', '#f59e0b', '#8b5cf6']
    
    fig_sector = go.Figure(data=[go.Pie(
        labels=sector_counts.index.tolist(),
        values=sector_counts.values.tolist(),
        marker=dict(colors=colors),
        hole=0.4,
        textinfo='label+percent',
        textfont=dict(size=13, color='white', family='Arial Black')
    )])
    
    fig_sector.update_layout(
        title="Portfolio Sector Distribution",
        height=450
    )
    
    names = list(SAMPLE_STARTUPS.keys())
    growth = [s['growth_rate'] for s in SAMPLE_STARTUPS.values()]
    arr = [s['arr'] for s in SAMPLE_STARTUPS.values()]
    score = [s['score'] for s in SAMPLE_STARTUPS.values()]
    
    fig_scatter = go.Figure(data=[go.Scatter(
        x=growth,
        y=arr,
        mode='markers+text',
        marker=dict(
            size=[s/5 for s in score],
            color=score,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="AI Score")
        ),
        text=names,
        textposition="top center",
        textfont=dict(size=10)
    )])
    
    fig_scatter.update_layout(
        title="Portfolio Performance Matrix",
        xaxis_title="Growth Rate (%)",
        yaxis_title="ARR ($M)",
        height=500
    )
    
    return fig_sector, fig_scatter

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">📈</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Signal Fire Intelligence
        </h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">AI-Powered Investment Analytics</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Data-driven deal sourcing • Portfolio intelligence • Market insights</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">VC Analytics</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">AI Scoring</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Deal Sourcing</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Data-Driven</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Signal Fire</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🎯 Analyze Startup", "📊 Portfolio Dashboard"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">AI-Powered Startup Analysis</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Comprehensive investment assessment with data-driven recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    startup_name = st.selectbox("Select Startup to Analyze", list(SAMPLE_STARTUPS.keys()), index=1)
    
    if st.button("📊 Analyze Investment Opportunity", type="primary", use_container_width=True):
        st.session_state.startup_analyzed = True
        st.session_state.startup_name = startup_name
    
    if st.session_state.startup_analyzed:
        startup, avg_score, recommendation, rec_color, rec_emoji, ltv_cac, magic_number, scores, strengths, concerns, fig_radar, fig_revenue = analyze_startup(st.session_state.startup_name)
        
        st.markdown(f'<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;"><h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🎯 Investment Analysis: {st.session_state.startup_name}</h2><div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px;"><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">AI Score</p><p style="font-size: 48px; color: {rec_color}; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{avg_score:.0f}</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{rec_emoji} {recommendation}</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">ARR</p><p style="font-size: 40px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${startup["arr"]:.1f}M</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Annual revenue</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Growth Rate</p><p style="font-size: 40px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{startup["growth_rate"]}%</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">YoY growth</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Runway</p><p style="font-size: 40px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{startup["runway"]}m</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Cash remaining</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Valuation</p><p style="font-size: 40px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${startup["valuation"]}M</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Post-money</p></div></div></div>', unsafe_allow_html=True)
        
        st.markdown(f'<div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;"><h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📊 Key Investment Metrics</h3><div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px;"><div style="background: white; border-radius: 14px; padding: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);"><h4 style="color: #1f2937; font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">📈 Revenue Multiple</h4><p style="font-size: 36px; color: #3b82f6; font-weight: 900; margin: 0;">{startup["valuation"]/startup["arr"]:.1f}x</p><p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Valuation / ARR</p><div style="background: #f0f9ff; border-radius: 8px; padding: 8px; margin-top: 10px;"><p style="font-size: 12px; color: #3b82f6; margin: 0; font-weight: 600;">Market avg: 8-12x for {startup["stage"]}</p></div></div><div style="background: white; border-radius: 14px; padding: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);"><h4 style="color: #1f2937; font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">💰 Capital Efficiency</h4><p style="font-size: 36px; color: #10b981; font-weight: 900; margin: 0;">{magic_number:.2f}</p><p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Magic Number (ARR/Raise)</p><div style="background: #d1fae5; border-radius: 8px; padding: 8px; margin-top: 10px;"><p style="font-size: 12px; color: #059669; margin: 0; font-weight: 600;">{"Excellent" if magic_number > 1.0 else "Good" if magic_number > 0.75 else "Needs improvement"} - Target: >1.0</p></div></div><div style="background: white; border-radius: 14px; padding: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);"><h4 style="color: #1f2937; font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">⚡ LTV:CAC Ratio</h4><p style="font-size: 36px; color: #ec4899; font-weight: 900; margin: 0;">{ltv_cac:.1f}x</p><p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Customer economics</p><div style="background: #fdf4ff; border-radius: 8px; padding: 8px; margin-top: 10px;"><p style="font-size: 12px; color: #a855f7; margin: 0; font-weight: 600;">{"Strong" if ltv_cac >= 3 else "Acceptable"} - Target: >3.0x</p></div></div></div><div style="background: white; border-radius: 14px; padding: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);"><h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 15px 0;">🏢 Company Overview</h4><div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;"><div style="background: #f0f9ff; border-radius: 8px; padding: 12px; text-align: center;"><p style="font-size: 12px; color: #6b7280; margin: 0 0 4px 0;">Sector</p><p style="font-size: 16px; color: #3b82f6; font-weight: 800; margin: 0;">{startup["sector"]}</p></div><div style="background: #fef3c7; border-radius: 8px; padding: 12px; text-align: center;"><p style="font-size: 12px; color: #6b7280; margin: 0 0 4px 0;">Stage</p><p style="font-size: 16px; color: #f59e0b; font-weight: 800; margin: 0;">{startup["stage"]}</p></div><div style="background: #d1fae5; border-radius: 8px; padding: 12px; text-align: center;"><p style="font-size: 12px; color: #6b7280; margin: 0 0 4px 0;">Team</p><p style="font-size: 16px; color: #10b981; font-weight: 800; margin: 0;">{startup["team_size"]} people</p></div><div style="background: #fdf4ff; border-radius: 8px; padding: 12px; text-align: center;"><p style="font-size: 12px; color: #6b7280; margin: 0 0 4px 0;">Founded</p><p style="font-size: 16px; color: #a855f7; font-weight: 800; margin: 0;">{startup["founded"]}</p></div></div></div></div>', unsafe_allow_html=True)
        
        # Scoring breakdown
        score_cards = []
        for metric, score in scores.items():
            color = '#10b981' if score >= 80 else '#f59e0b' if score >= 60 else '#ef4444'
            card = f'<div style="background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;"><span style="font-size: 16px; color: #1f2937; font-weight: 700;">{metric}</span><span style="font-size: 22px; color: {color}; font-weight: 900;">{score:.0f}</span></div><div style="background: #e5e7eb; border-radius: 8px; height: 8px; overflow: hidden;"><div style="background: {color}; height: 100%; width: {score}%; transition: width 0.3s;"></div></div></div>'
            score_cards.append(card)
        
        all_score_cards = ''.join(score_cards)
        
        scoring_html = f'<div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2); margin-bottom: 25px;"><h3 style="color: #92400e; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🤖 AI Investment Score Breakdown</h3><div style="display: grid; gap: 10px;">{all_score_cards}</div><div style="background: linear-gradient(135deg, {rec_color} 0%, {rec_color}dd 100%); border-radius: 12px; padding: 22px; margin-top: 20px; color: white; text-align: center;"><p style="font-size: 28px; font-weight: 900; margin: 0;">{rec_emoji} RECOMMENDATION: {recommendation}</p><p style="font-size: 15px; margin: 10px 0 0 0; opacity: 0.95;">{"Exceptional opportunity - move fast to secure allocation" if recommendation == "STRONG BUY" else "Solid fundamentals - proceed with due diligence" if recommendation == "BUY" else "Monitor for next round" if recommendation == "HOLD" else "Focus resources on stronger opportunities"}</p></div></div>'
        
        st.markdown(scoring_html, unsafe_allow_html=True)
        
        # Investment thesis
        strength_items = [f'<li style="color: #1f2937; font-size: 14px; font-weight: 600;">{s}</li>' for s in strengths]
        strength_list = ''.join(strength_items)
        
        concern_section = ''
        if concerns:
            concern_items = [f'<li style="color: #1f2937; font-size: 14px; font-weight: 600;">{c}</li>' for c in concerns]
            concern_list = ''.join(concern_items)
            concern_section = f'<div style="background: white; border-radius: 14px; padding: 22px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);"><h4 style="color: #f59e0b; font-size: 18px; font-weight: 800; margin: 0 0 12px 0;">⚠️ Areas to Monitor</h4><ul style="margin: 0; padding-left: 24px; line-height: 2;">{concern_list}</ul></div>'
        
        thesis_html = f'<div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;"><h3 style="color: #065f46; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">💡 Investment Thesis</h3><div style="background: white; border-radius: 14px; padding: 22px; margin-bottom: 15px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);"><h4 style="color: #10b981; font-size: 18px; font-weight: 800; margin: 0 0 12px 0;">✅ Key Strengths</h4><ul style="margin: 0; padding-left: 24px; line-height: 2;">{strength_list}</ul></div>{concern_section}</div>'
        
        st.markdown(thesis_html, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_radar, use_container_width=True)
        with col2:
            st.plotly_chart(fig_revenue, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Portfolio-Wide Analytics</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Performance metrics across all investments</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Load Portfolio Dashboard", type="primary", use_container_width=True):
        st.session_state.portfolio_loaded = True
    
    if st.session_state.portfolio_loaded:
        fig_sector, fig_scatter = generate_portfolio_dashboard()
        
        dashboard_html = f'<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;"><h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Portfolio Performance</h2><div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;"><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Portfolio Companies</p><p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{len(SAMPLE_STARTUPS)}</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Active investments</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total AUM</p><p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">$180M</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Under management</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Avg Growth</p><p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">184%</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">YoY across portfolio</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">IRR</p><p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">47%</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Internal return rate</p></div></div></div>'
        
        st.markdown(dashboard_html, unsafe_allow_html=True)
        
        # Portfolio companies
        colors = ['#3b82f6', '#10b981', '#ec4899', '#f59e0b', '#8b5cf6']
        company_cards = []
        
        for idx, (name, data) in enumerate(SAMPLE_STARTUPS.items()):
            color = colors[idx % len(colors)]
            card = f'<div style="background: white; border-left: 5px solid {color}; border-radius: 12px; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;"><div><p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 4px 0;">{name}</p><p style="font-size: 13px; color: #6b7280; margin: 0;">{data["sector"]} • {data["stage"]}</p></div><div style="text-align: right;"><p style="font-size: 24px; color: {color}; font-weight: 900; margin: 0;">{data["score"]}</p><p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">AI Score</p></div></div><div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px;"><div style="background: #f9fafb; border-radius: 6px; padding: 8px; text-align: center;"><p style="font-size: 11px; color: #6b7280; margin: 0;">ARR</p><p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 4px 0 0 0;">${data["arr"]:.1f}M</p></div><div style="background: #f9fafb; border-radius: 6px; padding: 8px; text-align: center;"><p style="font-size: 11px; color: #6b7280; margin: 0;">Growth</p><p style="font-size: 16px; color: #10b981; font-weight: 700; margin: 4px 0 0 0;">{data["growth_rate"]}%</p></div><div style="background: #f9fafb; border-radius: 6px; padding: 8px; text-align: center;"><p style="font-size: 11px; color: #6b7280; margin: 0;">Runway</p><p style="font-size: 16px; color: #3b82f6; font-weight: 700; margin: 4px 0 0 0;">{data["runway"]}m</p></div></div></div>'
            company_cards.append(card)
        
        all_companies = ''.join(company_cards)
        
        portfolio_html = f'<div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(168, 85, 247, 0.2); margin-bottom: 25px;"><h3 style="color: #6b21a8; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🏢 Portfolio Companies</h3><div style="display: grid; gap: 12px;">{all_companies}</div></div>'
        
        st.markdown(portfolio_html, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_sector, use_container_width=True)
        with col2:
            st.plotly_chart(fig_scatter, use_container_width=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Signal Fire</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🤖 AI Deal Sourcing</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Analyze 10,000+ startups per day vs 10-20 manually. Never miss the next unicorn hidden in data noise.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📊 Data-Driven Decisions</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Remove gut feel from investing. Score companies on 6 quantitative metrics. Replicate success patterns of your best investments.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ Competitive Edge</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Move faster than traditional VCs. Evaluate deals in hours, not weeks. Win competitive rounds with speed and conviction.
                </p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Real-World Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">1000x deal flow:</strong> Analyze entire market, not just warm intros</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">3-5x better returns:</strong> Data-driven selection beats intuition</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">50% faster decisions:</strong> Days → Hours for initial screening</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Zero bias:</strong> Objective scoring, not pattern matching on founders</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-Factor Scoring</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">6 weighted metrics for comprehensive assessment</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Predictive Analytics</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Revenue projections, runway analysis</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Portfolio Intelligence</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Aggregate metrics, sector distribution</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Market Benchmarking</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Compare against cohort averages</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Signal Fire</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Streamlit • Plotly • Financial Analytics • ML Scoring
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered venture capital investment intelligence.<br>
            Startup scoring • Portfolio analytics • Data-driven insights • Market intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)