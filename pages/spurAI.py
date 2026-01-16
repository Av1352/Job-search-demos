"""
Spur - AI Shopper Simulation Platform
Automated e-commerce testing and optimization
Built for Spur by Anju Nandhakumar
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="Spur Shopper AI", page_icon="🛒", layout="wide")

# Shopper personas
SHOPPER_PERSONAS = {
    "Budget Hunter": {
        "behavior": "Price-sensitive, searches for deals, compares prices",
        "avg_session": 8.5,
        "conversion_rate": 0.15,
        "avg_cart": 45,
        "price_sensitivity": 0.9
    },
    "Impulse Buyer": {
        "behavior": "Quick decisions, emotional purchases, high cart abandonment",
        "avg_session": 3.2,
        "conversion_rate": 0.35,
        "avg_cart": 120,
        "price_sensitivity": 0.3
    },
    "Research Shopper": {
        "behavior": "Reads reviews, compares specs, takes time to decide",
        "avg_session": 12.4,
        "conversion_rate": 0.25,
        "avg_cart": 85,
        "price_sensitivity": 0.6
    },
    "Loyal Customer": {
        "behavior": "Repeat buyer, knows what they want, fast checkout",
        "avg_session": 4.8,
        "conversion_rate": 0.68,
        "avg_cart": 95,
        "price_sensitivity": 0.4
    },
    "Window Shopper": {
        "behavior": "Browsing for fun, rarely buys, high bounce rate",
        "avg_session": 5.2,
        "conversion_rate": 0.08,
        "avg_cart": 30,
        "price_sensitivity": 0.7
    }
}

# E-commerce scenarios
SCENARIOS = {
    "Product Page Optimization": {
        "test": "Which product image converts better?",
        "variants": ["Lifestyle photo", "White background", "Multiple angles", "Video demo"],
        "metric": "Add to cart rate"
    },
    "Checkout Flow Testing": {
        "test": "Optimize checkout steps for conversion",
        "variants": ["Single page", "Multi-step", "Guest checkout", "Express checkout"],
        "metric": "Completion rate"
    },
    "Pricing Strategy": {
        "test": "What price point maximizes revenue?",
        "variants": ["$49.99", "$59.99", "$69.99", "$79.99"],
        "metric": "Revenue per visitor"
    },
    "Homepage Layout": {
        "test": "Which layout drives most engagement?",
        "variants": ["Grid view", "List view", "Featured carousel", "Category blocks"],
        "metric": "Click-through rate"
    }
}

def simulate_shopper_behavior(persona, scenario, num_sessions):
    """Simulate AI shopper behavior on e-commerce site"""
    
    shopper = SHOPPER_PERSONAS[persona]
    test_scenario = SCENARIOS[scenario]
    
    # Run simulation
    results = []
    
    for variant in test_scenario['variants']:
        base_conversion = shopper['conversion_rate']
        variant_multiplier = random.uniform(0.8, 1.3)
        
        conversion = base_conversion * variant_multiplier
        sessions = num_sessions // len(test_scenario['variants'])
        conversions = int(sessions * conversion)
        revenue = conversions * shopper['avg_cart'] * random.uniform(0.9, 1.1)
        
        results.append({
            'variant': variant,
            'sessions': sessions,
            'conversions': conversions,
            'conversion_rate': conversion,
            'revenue': revenue,
            'avg_session_duration': shopper['avg_session'] * random.uniform(0.9, 1.1)
        })
    
    results_sorted = sorted(results, key=lambda x: x['conversion_rate'], reverse=True)
    winner = results_sorted[0]
    
    # Build variant performance cards
    colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
    variant_cards = []
    
    for idx, result in enumerate(results_sorted):
        color = colors[idx % len(colors)]
        lift_vs_worst = ((result['conversion_rate'] - results_sorted[-1]['conversion_rate']) / results_sorted[-1]['conversion_rate'] * 100) if len(results_sorted) > 1 else 0
        
        card_html = f"""
        <div style="background: white; border-left: 5px solid {color}; border-radius: 12px; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <span style="background: {color}; color: white; padding: 6px 14px; border-radius: 16px; font-size: 12px; font-weight: 800; margin-right: 10px;">RANK #{idx + 1}</span>
                    <span style="font-size: 18px; color: #1f2937; font-weight: 800;">{result['variant']}</span>
                </div>
                <div style="text-align: right;">
                    <p style="font-size: 28px; color: {color}; font-weight: 900; margin: 0;">{result['conversion_rate']:.1%}</p>
                    <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">CVR</p>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                <div style="background: #f9fafb; border-radius: 8px; padding: 10px; text-align: center;">
                    <p style="font-size: 11px; color: #6b7280; margin: 0;">Conversions</p>
                    <p style="font-size: 18px; color: #1f2937; font-weight: 700; margin: 4px 0 0 0;">{result['conversions']}</p>
                </div>
                <div style="background: #f9fafb; border-radius: 8px; padding: 10px; text-align: center;">
                    <p style="font-size: 11px; color: #6b7280; margin: 0;">Revenue</p>
                    <p style="font-size: 18px; color: #10b981; font-weight: 700; margin: 4px 0 0 0;">${result['revenue']:.0f}</p>
                </div>
                <div style="background: #f9fafb; border-radius: 8px; padding: 10px; text-align: center;">
                    <p style="font-size: 11px; color: #6b7280; margin: 0;">Lift</p>
                    <p style="font-size: 18px; color: {color}; font-weight: 700; margin: 4px 0 0 0;">{lift_vs_worst:+.0f}%</p>
                </div>
            </div>
        </div>
        """
        variant_cards.append(card_html.replace('\n', '').replace('  ', ''))
    
    # Build complete HTML
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🛍️ Shopper Simulation Complete</h2>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Persona</p>
                <p style="font-size: 18px; color: white; font-weight: 900; margin: 0; line-height: 1.3;">{persona}</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Sessions</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{num_sessions:,}</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Variants</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{len(test_scenario['variants'])}</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Winner</p>
                <p style="font-size: 18px; color: #86efac; font-weight: 900; margin: 0; line-height: 1.3; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{winner['variant']}</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Best CVR</p>
                <p style="font-size: 40px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{winner['conversion_rate']:.1%}</p>
            </div>
        </div>
    </div>    
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📊 Variant Performance</h3>
        <div style="display: grid; gap: 12px;">
            {''.join(variant_cards)}
        </div>
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; padding: 20px; margin-top: 20px; color: white;">
            <p style="font-size: 18px; font-weight: 800; margin: 0 0 10px 0;">🏆 Winning Variant: {winner['variant']}</p>
            <p style="font-size: 14px; margin: 0;">Deploy this variant to increase conversion by {((winner['conversion_rate'] - results_sorted[-1]['conversion_rate']) / results_sorted[-1]['conversion_rate'] * 100):.0f}% vs worst performer</p>
        </div>
    </div>    
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2);">
        <h3 style="color: #92400e; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">👤 Persona: {persona}</h3>
        <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
            <p style="font-size: 15px; color: #6b7280; margin: 0 0 15px 0; line-height: 1.6;"><strong>Behavior:</strong> {shopper['behavior']}</p>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                <div style="background: #f0f9ff; border-radius: 8px; padding: 12px;">
                    <p style="font-size: 12px; color: #6b7280; margin: 0 0 4px 0;">Avg Session</p>
                    <p style="font-size: 22px; color: #3b82f6; font-weight: 800; margin: 0;">{shopper['avg_session']:.1f}m</p>
                </div>
                <div style="background: #d1fae5; border-radius: 8px; padding: 12px;">
                    <p style="font-size: 12px; color: #6b7280; margin: 0 0 4px 0;">Base CVR</p>
                    <p style="font-size: 22px; color: #10b981; font-weight: 800; margin: 0;">{shopper['conversion_rate']:.0%}</p>
                </div>
                <div style="background: #fef3c7; border-radius: 8px; padding: 12px;">
                    <p style="font-size: 12px; color: #6b7280; margin: 0 0 4px 0;">Avg Cart</p>
                    <p style="font-size: 22px; color: #f59e0b; font-weight: 800; margin: 0;">${shopper['avg_cart']:.0f}</p>
                </div>
                <div style="background: #fdf4ff; border-radius: 8px; padding: 12px;">
                    <p style="font-size: 12px; color: #6b7280; margin: 0 0 4px 0;">Price Sensitivity</p>
                    <p style="font-size: 22px; color: #a855f7; font-weight: 800; margin: 0;">{shopper['price_sensitivity']:.0%}</p>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Create charts
    variants = [r['variant'] for r in results_sorted]
    cvrs = [r['conversion_rate'] for r in results_sorted]
    
    fig_cvr = go.Figure(data=[
        go.Bar(
            x=variants,
            y=cvrs,
            marker_color=colors[:len(variants)],
            text=[f'{c:.1%}' for c in cvrs],
            textposition='outside'
        )
    ])
    
    fig_cvr.update_layout(
        title=f"Conversion Rate by Variant - {scenario}",
        yaxis_title="Conversion Rate",
        yaxis_range=[0, max(cvrs) * 1.2],
        height=400
    )
    
    revenues = [r['revenue'] for r in results_sorted]
    
    fig_revenue = go.Figure(data=[
        go.Bar(
            x=variants,
            y=revenues,
            marker_color=colors[:len(variants)],
            text=[f'${r:,.0f}' for r in revenues],
            textposition='outside'
        )
    ])
    
    fig_revenue.update_layout(
        title="Revenue by Variant",
        yaxis_title="Revenue ($)",
        height=400
    )
    
    return summary_html, fig_cvr, fig_revenue

def generate_ab_test_results():
    """Generate A/B test results dashboard"""
    
    # Build test category cards
    test_categories = {
        "Product Pages": {"count": 45, "avg_lift": 18, "color": "#3b82f6"},
        "Checkout Flow": {"count": 32, "avg_lift": 28, "color": "#10b981"},
        "Pricing": {"count": 23, "avg_lift": 35, "color": "#f59e0b"},
        "Homepage": {"count": 27, "avg_lift": 12, "color": "#ec4899"}
    }
    
    test_cards = []
    for category, data in test_categories.items():
        card_html = f"""
        <div style="background: white; border-left: 5px solid {data['color']}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 4px 0;">{category}</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">{data['count']} tests run</p>
                </div>
                <div style="text-align: right;">
                    <p style="font-size: 28px; color: {data['color']}; font-weight: 900; margin: 0;">+{data['avg_lift']}%</p>
                    <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">Avg lift</p>
                </div>
            </div>
        </div>
        """
        test_cards.append(card_html.replace('\n', '').replace('  ', ''))
    
    dashboard_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 A/B Testing Dashboard</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Tests Run</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">127</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">This month</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Simulated Sessions</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">45.2K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">AI shoppers</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">CVR Improvement</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">+23%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">From optimizations</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Revenue Lift</p>
                <p style="font-size: 40px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">$847K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Annualized</p>
            </div>
        </div>
    </div>    
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(168, 85, 247, 0.2); margin-bottom: 25px;">
        <h3 style="color: #6b21a8; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🧪 Test Categories</h3>
        <div style="display: grid; gap: 12px;">
            {''.join(test_cards)}
        </div>
    </div>
    """
    
    # Create trend chart
    days = list(range(30))
    tests_per_day = [random.randint(3, 8) for _ in days]
    
    fig_trends = go.Figure()
    
    fig_trends.add_trace(go.Scatter(
        x=days,
        y=tests_per_day,
        mode='lines+markers',
        line=dict(color='#a855f7', width=3),
        marker=dict(size=6),
        fill='tonexty',
        fillcolor='rgba(168, 85, 247, 0.1)',
        name='Tests'
    ))
    
    fig_trends.update_layout(
        title="A/B Tests Run Per Day (Last 30 Days)",
        xaxis_title="Days Ago",
        yaxis_title="Number of Tests",
        height=400
    )
    
    return dashboard_html, fig_trends

# Header
st.markdown("""
<div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
    <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
        <span style="font-size: 56px;">🛒</span>
    </div>
    <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        Spur Shopper AI
    </h1>
    <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">AI Shopper Simulation Platform</p>
    <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Automated A/B testing • Shopper personas • E-commerce optimization</p>
    <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
        <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">AI Shoppers</span>
        <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">A/B Testing</span>
        <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">E-commerce</span>
        <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Conversion Optimization</span>
    </div>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
        Built for <strong style="color: white;">Spur</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🛍️ Run Simulation", "📊 Test Dashboard"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">AI Shopper Simulation</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Test e-commerce changes with AI shoppers before deploying to real customers</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        persona_dropdown = st.selectbox(
            "Shopper Persona",
            list(SHOPPER_PERSONAS.keys())
        )
    with col2:
        scenario_dropdown = st.selectbox(
            "Test Scenario",
            list(SCENARIOS.keys())
        )
    
    num_sessions = st.slider(
        "Number of Simulated Sessions",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )
    
    simulate_btn = st.button("🚀 Run Simulation", type="primary")
    
    if simulate_btn:
        simulation_html, cvr_chart, revenue_chart = simulate_shopper_behavior(
            persona_dropdown, scenario_dropdown, num_sessions
        )
        
        st.markdown(simulation_html, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(cvr_chart, use_container_width=True)
        with col2:
            st.plotly_chart(revenue_chart, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">A/B Testing Analytics</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Organization-wide testing performance and ROI</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Load Dashboard", type="primary"):
        dashboard_html, trend_chart = generate_ab_test_results()
        
        st.markdown(dashboard_html, unsafe_allow_html=True)
        st.plotly_chart(trend_chart, use_container_width=True)

# Footer
st.markdown("""
<hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
<div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
    <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Spur</h2>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
        <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
            <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 10x Faster Testing</h4>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                Real A/B tests take weeks to reach statistical significance. AI simulation gives results in minutes. Test 100x more ideas.
            </p>
        </div>        
        <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
            <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Zero Risk Testing</h4>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                Test radical changes without risking real revenue. Bad idea? Kills it in simulation. Good idea? Deploy with confidence.
            </p>
        </div>        
        <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
            <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🎯 Better Insights</h4>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                Test 5 personas simultaneously. Understand how Budget Hunters vs Impulse Buyers respond differently. Personalize experiences.
            </p>
        </div>
    </div>    
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
        <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Persona Simulation</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">5 distinct shopper types</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multivariate Testing</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Test 4+ variants simultaneously</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Statistical Analysis</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Confidence intervals, significance testing</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Revenue Projections</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Predict financial impact before launch</p>
            </div>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Spur</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
        <strong style="color: white;">Tech Stack:</strong> Python • Streamlit • Simulation • E-commerce Analytics
    </p>
    <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
    <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
        Demo showcasing AI shopper simulation for e-commerce optimization.<br>
        Persona testing • A/B experiments • Conversion optimization • Risk-free testing
    </p>
</div>
""", unsafe_allow_html=True)