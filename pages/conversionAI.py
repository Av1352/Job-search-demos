"""
Conversion AI - Marketing Conversion Optimization Platform
AI-powered funnel analysis and optimization
Built for Conversion AI by Anju Nandhakumar
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import random

# Page config
st.set_page_config(
    page_title="Conversion AI Demo - Anju Vilashni",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main { background: white; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
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

# Marketing funnel stages
FUNNEL_STAGES = {
    "Awareness": {"visitors": 10000, "conversion_to_next": 0.45},
    "Interest": {"visitors": 4500, "conversion_to_next": 0.35},
    "Consideration": {"visitors": 1575, "conversion_to_next": 0.28},
    "Intent": {"visitors": 441, "conversion_to_next": 0.42},
    "Purchase": {"visitors": 185, "conversion_to_next": 1.0}
}

OPTIMIZATION_STRATEGIES = {
    "Landing Page Copy": {
        "impact_stage": "Awareness",
        "expected_lift": 0.15,
        "implementation": "Easy",
        "cost": "Low"
    },
    "CTA Button Optimization": {
        "impact_stage": "Interest",
        "expected_lift": 0.22,
        "implementation": "Easy",
        "cost": "Low"
    },
    "Email Nurture Sequence": {
        "impact_stage": "Consideration",
        "expected_lift": 0.35,
        "implementation": "Medium",
        "cost": "Medium"
    },
    "Retargeting Ads": {
        "impact_stage": "Intent",
        "expected_lift": 0.28,
        "implementation": "Medium",
        "cost": "High"
    },
    "Checkout Flow Simplification": {
        "impact_stage": "Purchase",
        "expected_lift": 0.18,
        "implementation": "Hard",
        "cost": "Medium"
    }
}

# Header
components.html(
    """
    <div style="
        text-align: center;
        padding: 20px 30px 70px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 25px;
        box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);
    ">
        <div style="
            width: 100px;
            height: 100px;
            background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
            border-radius: 50%;
            margin: 0 auto 25px auto;
            border: 5px solid white;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5);
        ">
            <span style="font-size: 56px;">📊</span>
        </div>

        <h1 style="
            font-size: 58px;
            font-weight: 900;
            color: white;
            margin: 0 0 18px 0;
            text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            Conversion AI
        </h1>

        <p style="
            font-size: 28px;
            color: rgba(255,255,255,0.95);
            font-weight: 700;
            margin: 15px 0;
        ">
            Marketing Conversion Optimization
        </p>

        <p style="
            font-size: 18px;
            color: rgba(255,255,255,0.85);
            font-weight: 500;
            margin-bottom: 25px;
        ">
            AI-powered funnel analysis • Automated optimization • Revenue growth
        </p>

        <div style="
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
            max-width: 850px;
            margin: 28px auto 0 auto;
        ">
            <span style="background:#ec4899;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Funnel Analysis</span>
            <span style="background:#8b5cf6;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">AI Optimization</span>
            <span style="background:#f59e0b;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">ROAS Tracking</span>
            <span style="background:#10b981;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Revenue Growth</span>
        </div>

        <p style="
            font-size: 16px;
            color: rgba(255,255,255,0.9);
            margin-top: 28px;
            font-weight: 600;
        ">
            Built for <strong style="color:white;">Conversion AI</strong>
            by <strong style="color:white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """,
    height=520,
)

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["🔻 Funnel Analysis", "📊 Campaign Dashboard"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">AI Funnel Analyzer</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Identify drop-off points and get AI-powered optimization recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Analyze Marketing Funnel", key="analyze"):
        total_visitors = FUNNEL_STAGES["Awareness"]["visitors"]
        total_purchases = FUNNEL_STAGES["Purchase"]["visitors"]
        overall_conversion = (total_purchases / total_visitors) * 100
        
        drop_offs = {}
        for stage, data in FUNNEL_STAGES.items():
            if data['conversion_to_next'] < 1.0:
                drop_off_rate = 1 - data['conversion_to_next']
                drop_offs[stage] = drop_off_rate
        
        biggest_drop = max(drop_offs, key=drop_offs.get)
        
        # Summary
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Funnel Analysis Complete</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Total Visitors</p>
                <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">{total_visitors:,}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Top of funnel</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(16, 185, 129, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Conversions</p>
                <p style="font-size: 48px; color: #10b981; font-weight: 900; margin: 0;">{total_purchases}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Purchases</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">CVR</p>
                <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">{overall_conversion:.1f}%</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Overall conversion</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(239, 68, 68, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Biggest Drop</p>
                <p style="font-size: 18px; color: #ef4444; font-weight: 900; margin: 0; line-height: 1.2;">{biggest_drop}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">{drop_offs[biggest_drop]:.0%} drop-off</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Funnel stages
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🔻 Conversion Funnel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
        
        for idx, (stage, data) in enumerate(FUNNEL_STAGES.items()):
            width_pct = (data['visitors'] / total_visitors) * 100
            
            st.markdown(f"""
            <div style="background: white; border-radius: 12px; padding: 16px; margin-bottom: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0;">{stage}</p>
                    <p style="font-size: 24px; color: {colors[idx]}; font-weight: 900; margin: 0;">{data['visitors']:,}</p>
                </div>
                <div style="background: #e5e7eb; border-radius: 8px; height: 12px; overflow: hidden;">
                    <div style="background: {colors[idx]}; height: 100%; width: {width_pct}%;"></div>
                </div>
                {f'<p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Conversion to next: {data["conversion_to_next"]:.0%}</p>' if data['conversion_to_next'] < 1.0 else ''}
            </div>
            """, unsafe_allow_html=True)
        
        # AI Recommendations
        sorted_strategies = sorted(OPTIMIZATION_STRATEGIES.items(), key=lambda x: x[1]['expected_lift'], reverse=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #065f46; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">💡 AI Optimization Recommendations</h3>
        </div>
        """, unsafe_allow_html=True)
        
        rank_colors = ['#10b981', '#3b82f6', '#8b5cf6']
        
        for idx, (strategy, data) in enumerate(sorted_strategies[:3]):
            color = rank_colors[idx]
            
            components.html(f"""
            <div style="background: white; border-left: 5px solid {color}; border-radius: 12px; padding: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <span style="background: {color}; color: white; padding: 6px 14px; border-radius: 16px; font-size: 12px; font-weight: 800; margin-right: 10px;">TOP #{idx + 1}</span>
                        <span style="font-size: 18px; color: #1f2937; font-weight: 800;">{strategy}</span>
                    </div>
                    <div style="text-align: right;">
                        <p style="font-size: 28px; color: {color}; font-weight: 900; margin: 0;">+{data['expected_lift']:.0%}</p>
                        <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">Expected lift</p>
                    </div>
                </div>                
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2px;">
                    <div style="background: #f9fafb; border-radius: 8px; padding: 10px; text-align: center;">
                        <p style="font-size: 11px; color: #6b7280; margin: 0;">Target Stage</p>
                        <p style="font-size: 14px; color: #1f2937; font-weight: 700; margin: 4px 0 0 0;">{data['impact_stage']}</p>
                    </div>
                    <div style="background: #f9fafb; border-radius: 8px; padding: 10px; text-align: center;">
                        <p style="font-size: 11px; color: #6b7280; margin: 0;">Difficulty</p>
                        <p style="font-size: 14px; color: #1f2937; font-weight: 700; margin: 4px 0 0 0;">{data['implementation']}</p>
                    </div>
                    <div style="background: #f9fafb; border-radius: 8px; padding: 10px; text-align: center;">
                        <p style="font-size: 11px; color: #6b7280; margin: 0;">Cost</p>
                        <p style="font-size: 14px; color: #1f2937; font-weight: 700; margin: 4px 0 0 0;">{data['cost']}</p>
                    </div>
                </div>
            </div>
            """, height=130)
        
        total_lift = sum([s[1]['expected_lift'] for s in sorted_strategies[:3]])
        additional_conversions = int(total_purchases * total_lift)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; padding: 20px; margin-top: 20px; color: white;">
            <p style="font-size: 18px; font-weight: 800; margin: 0 0 10px 0;">🎯 Quick Win: Implement Top 3 Recommendations</p>
            <p style="font-size: 14px; margin: 0;">Projected impact: {total_lift:.0%} combined lift → {additional_conversions} additional conversions/month</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Funnel chart
        stages = list(FUNNEL_STAGES.keys())
        visitors = [FUNNEL_STAGES[s]['visitors'] for s in stages]
        
        fig_funnel = go.Figure(go.Funnel(
            y=stages,
            x=visitors,
            textinfo="value+percent initial",
            marker=dict(color=colors)
        ))
        fig_funnel.update_layout(title="Marketing Funnel Visualization", height=500)
        st.plotly_chart(fig_funnel, use_container_width=True)
        
        # Optimization impact
        strategies_list = [s for s, _ in sorted_strategies[:5]]
        lifts = [data['expected_lift'] * 100 for _, data in sorted_strategies[:5]]
        
        fig_impact = go.Figure(data=[
            go.Bar(
                x=strategies_list,
                y=lifts,
                marker_color=['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ec4899'],
                text=[f'+{l:.0f}%' for l in lifts],
                textposition='outside'
            )
        ])
        fig_impact.update_layout(
            title="Expected Lift by Optimization Strategy",
            yaxis_title="Conversion Lift (%)",
            height=400
        )
        st.plotly_chart(fig_impact, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Multi-Channel Performance</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Track ROI across all marketing channels</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Load Campaign Analytics", key="dashboard"):
        # Dashboard metrics
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📈 Campaign Performance</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Active Campaigns</p>
                <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">12</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(251, 191, 36, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(251, 191, 36, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Total Spend</p>
                <p style="font-size: 40px; color: #f59e0b; font-weight: 900; margin: 0;">$45K</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">This month</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(16, 185, 129, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">ROAS</p>
                <p style="font-size: 48px; color: #10b981; font-weight: 900; margin: 0;">4.2x</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Return on ad spend</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">CVR Improvement</p>
                <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">+28%</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">vs last month</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Channel performance
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #92400e; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📢 Channel Performance</h3>
        </div>
        """, unsafe_allow_html=True)
        
        channels = {
            "Google Ads": {"spend": 18000, "conversions": 245, "roas": 4.8, "color": "#3b82f6"},
            "Facebook Ads": {"spend": 12000, "conversions": 180, "roas": 3.9, "color": "#8b5cf6"},
            "Email Marketing": {"spend": 5000, "conversions": 142, "roas": 6.2, "color": "#10b981"},
            "Organic Search": {"spend": 8000, "conversions": 95, "roas": 3.1, "color": "#f59e0b"},
            "Referral": {"spend": 2000, "conversions": 68, "roas": 8.5, "color": "#ec4899"}
        }
        
        for channel, data in channels.items():
            st.markdown(f"""
            <div style="background: white; border-left: 5px solid {data['color']}; border-radius: 12px; padding: 18px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 4px 0;">{channel}</p>
                        <p style="font-size: 13px; color: #6b7280; margin: 0;">{data['conversions']} conversions</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="font-size: 28px; color: {data['color']}; font-weight: 900; margin: 0;">{data['roas']:.1f}x</p>
                        <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">ROAS</p>
                    </div>
                </div>
                <div style="background: #f9fafb; border-radius: 8px; padding: 10px; margin-top: 10px;">
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Spend: <strong style="color: #1f2937;">${data['spend']:,}</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Trend chart
        days = list(range(30))
        cvr_trend = [1.5 + (i * 0.03) + random.uniform(-0.1, 0.1) for i in days]
        
        fig_trends = go.Figure()
        fig_trends.add_trace(go.Scatter(
            x=days,
            y=cvr_trend,
            mode='lines+markers',
            line=dict(color='#10b981', width=3),
            marker=dict(size=5),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        fig_trends.update_layout(
            title="Conversion Rate Trend (Last 30 Days)",
            xaxis_title="Days Ago",
            yaxis_title="Conversion Rate (%)",
            height=400
        )
        st.plotly_chart(fig_trends, use_container_width=True)

# Footer
st.markdown("<hr style='border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Conversion AI</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
        <strong>Tech Stack:</strong> Python • Streamlit • Marketing Analytics • Funnel Optimization
    </p>
</div>
""", unsafe_allow_html=True)