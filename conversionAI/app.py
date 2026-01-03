"""
Conversion AI - Marketing Conversion Optimization Platform
AI-powered funnel analysis and optimization
Built for Conversion AI by Anju Nandhakumar
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import random

# Marketing funnel stages
FUNNEL_STAGES = {
    "Awareness": {"visitors": 10000, "conversion_to_next": 0.45},
    "Interest": {"visitors": 4500, "conversion_to_next": 0.35},
    "Consideration": {"visitors": 1575, "conversion_to_next": 0.28},
    "Intent": {"visitors": 441, "conversion_to_next": 0.42},
    "Purchase": {"visitors": 185, "conversion_to_next": 1.0}
}

# Optimization strategies
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

def analyze_funnel():
    """Analyze marketing funnel with AI recommendations"""
    
    # Calculate overall conversion
    total_visitors = FUNNEL_STAGES["Awareness"]["visitors"]
    total_purchases = FUNNEL_STAGES["Purchase"]["visitors"]
    overall_conversion = (total_purchases / total_visitors) * 100
    
    # Identify biggest drop-off
    drop_offs = {}
    for stage, data in FUNNEL_STAGES.items():
        if data['conversion_to_next'] < 1.0:
            drop_off_rate = 1 - data['conversion_to_next']
            drop_offs[stage] = drop_off_rate
    
    biggest_drop = max(drop_offs, key=drop_offs.get)
    
    # Analysis summary
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Funnel Analysis Complete</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Visitors</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{total_visitors:,}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Top of funnel</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Conversions</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{total_purchases}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Purchases</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">CVR</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{overall_conversion:.1f}%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Overall conversion</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Biggest Drop</p>
                <p style="font-size: 18px; color: #fca5a5; font-weight: 900; margin: 0; line-height: 1.2; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{biggest_drop}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{drop_offs[biggest_drop]:.0%} drop-off</p>
            </div>
        </div>
    </div>
    """
    
    # Funnel visualization
    funnel_html = """
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🔻 Conversion Funnel</h3>
        
        <div style="display: grid; gap: 10px;">
    """
    
    colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
    
    for idx, (stage, data) in enumerate(FUNNEL_STAGES.items()):
        width_pct = (data['visitors'] / total_visitors) * 100
        
        funnel_html += f"""
        <div style="background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0;">{stage}</p>
                <p style="font-size: 24px; color: {colors[idx]}; font-weight: 900; margin: 0;">{data['visitors']:,}</p>
            </div>
            <div style="background: #e5e7eb; border-radius: 8px; height: 12px; overflow: hidden;">
                <div style="background: {colors[idx]}; height: 100%; width: {width_pct}%; transition: width 0.3s;"></div>
            </div>
            {f'<p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Conversion to next: {data["conversion_to_next"]:.0%}</p>' if data['conversion_to_next'] < 1.0 else ''}
        </div>
        """
    
    funnel_html += "</div></div>"
    
    # AI recommendations
    rec_html = """
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2);">
        <h3 style="color: #065f46; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">💡 AI Optimization Recommendations</h3>
        
        <div style="display: grid; gap: 12px;">
    """
    
    # Sort by expected lift
    sorted_strategies = sorted(OPTIMIZATION_STRATEGIES.items(), key=lambda x: x[1]['expected_lift'], reverse=True)
    
    for idx, (strategy, data) in enumerate(sorted_strategies[:3]):
        rank_colors = ['#10b981', '#3b82f6', '#8b5cf6']
        color = rank_colors[idx]
        
        rec_html += f"""
        <div style="background: white; border-left: 5px solid {color}; border-radius: 12px; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
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
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px;">
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
        """
    
    rec_html += """
        </div>
        
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; padding: 20px; margin-top: 20px; color: white;">
            <p style="font-size: 18px; font-weight: 800; margin: 0 0 10px 0;">🎯 Quick Win: Implement Top 3 Recommendations</p>
            <p style="font-size: 14px; margin: 0;">Projected impact: """ + f"{sum([s[1]['expected_lift'] for s in sorted_strategies[:3]]):.0%}" + """ combined lift → """ + f"{int(total_purchases * sum([s[1]['expected_lift'] for s in sorted_strategies[:3]]))}" + """ additional conversions/month</p>
        </div>
    </div>
    """
    
    # Create funnel chart
    stages = list(FUNNEL_STAGES.keys())
    visitors = [FUNNEL_STAGES[s]['visitors'] for s in stages]
    
    fig_funnel = go.Figure(go.Funnel(
        y=stages,
        x=visitors,
        textinfo="value+percent initial",
        marker=dict(color=colors)
    ))
    
    fig_funnel.update_layout(
        title="Marketing Funnel Visualization",
        height=500
    )
    
    # Create optimization impact chart
    strategies = [s for s, _ in sorted_strategies[:5]]
    lifts = [data['expected_lift'] * 100 for _, data in sorted_strategies[:5]]
    
    fig_impact = go.Figure(data=[
        go.Bar(
            x=strategies,
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
    
    return summary_html + funnel_html + rec_html, fig_funnel, fig_impact

def generate_campaign_analytics():
    """Generate campaign performance dashboard"""
    
    dashboard_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📈 Campaign Performance</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Active Campaigns</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">12</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Spend</p>
                <p style="font-size: 40px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">$45K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">This month</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">ROAS</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">4.2x</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Return on ad spend</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">CVR Improvement</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">+28%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">vs last month</p>
            </div>
        </div>
    </div>
    """
    
    # Channel performance
    channels_html = """
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2); margin-bottom: 25px;">
        <h3 style="color: #92400e; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📢 Channel Performance</h3>
        
        <div style="display: grid; gap: 12px;">
    """
    
    channels = {
        "Google Ads": {"spend": 18000, "conversions": 245, "roas": 4.8, "color": "#3b82f6"},
        "Facebook Ads": {"spend": 12000, "conversions": 180, "roas": 3.9, "color": "#8b5cf6"},
        "Email Marketing": {"spend": 5000, "conversions": 142, "roas": 6.2, "color": "#10b981"},
        "Organic Search": {"spend": 8000, "conversions": 95, "roas": 3.1, "color": "#f59e0b"},
        "Referral": {"spend": 2000, "conversions": 68, "roas": 8.5, "color": "#ec4899"}
    }
    
    for channel, data in channels.items():
        channels_html += f"""
        <div style="background: white; border-left: 5px solid {data['color']}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
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
        """
    
    channels_html += "</div></div>"
    
    # Create trend chart
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
        fillcolor='rgba(16, 185, 129, 0.1)',
        name='CVR'
    ))
    
    fig_trends.update_layout(
        title="Conversion Rate Trend (Last 30 Days)",
        xaxis_title="Days Ago",
        yaxis_title="Conversion Rate (%)",
        height=400
    )
    
    return dashboard_html + channels_html, fig_trends

custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">📊</span>
        </div>
        
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Conversion AI
        </h1>
        
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Marketing Conversion Optimization</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">AI-powered funnel analysis • Automated optimization • Revenue growth</p>
        
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Funnel Analysis</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">AI Optimization</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">ROAS Tracking</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Revenue Growth</span>
        </div>
        
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Conversion AI</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("🔻 Funnel Analysis"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">AI Funnel Analyzer</h3>
                <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Identify drop-off points and get AI-powered optimization recommendations</p>
            </div>
            """)
            
            analyze_btn = gr.Button("🔍 Analyze Marketing Funnel", variant="primary", size="lg")
            
            analysis_output = gr.HTML(label="Funnel Analysis")
            funnel_chart = gr.Plot(label="Funnel Visualization")
            impact_chart = gr.Plot(label="Optimization Impact")
            
            analyze_btn.click(
                fn=analyze_funnel,
                inputs=[],
                outputs=[analysis_output, funnel_chart, impact_chart]
            )
        
        with gr.Tab("📊 Campaign Dashboard"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Multi-Channel Performance</h3>
                <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Track ROI across all marketing channels</p>
            </div>
            """)
            
            dashboard_btn = gr.Button("📊 Load Campaign Analytics", variant="primary", size="lg")
            
            dashboard_output = gr.HTML(label="Dashboard")
            trend_chart = gr.Plot(label="CVR Trend")
            
            dashboard_btn.click(
                fn=generate_campaign_analytics,
                inputs=[],
                outputs=[dashboard_output, trend_chart]
            )
    
    gr.HTML("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Conversion AI</h2>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📈 Revenue Impact</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    28% CVR improvement = 28% more revenue with same traffic. For $1M/month business, that's $336K additional annual revenue.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🎯 Automated Insights</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    AI identifies drop-off points, suggests fixes, predicts impact. Marketing teams act on data, not guesses.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Better ROAS</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Optimize spend allocation across channels. Put budget where ROAS is highest. 4.2x average return vs industry 2.8x.
                </p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Funnel Analytics</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">5-stage conversion tracking</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ AI Recommendations</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Prioritized by expected impact</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-Channel ROI</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Track ROAS across 5+ channels</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Trend Analysis</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">30-day performance tracking</p>
                </div>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Conversion AI</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Gradio • Marketing Analytics • Funnel Optimization
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered conversion optimization for marketing teams.<br>
            Funnel analysis • Channel ROI • AI recommendations • Revenue growth
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()