"""
Autonomous Technologies Group - AI Financial Advisor
Superintelligent investment and wealth management
Built for Autonomous Technologies Group by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from utils.sidebar import render_sidebar
render_sidebar()

st.set_page_config(page_title="Autonomous Tech - AI Financial Advisor", layout="wide")

# Initialize session state
if 'portfolio_analyzed' not in st.session_state:
    st.session_state.portfolio_analyzed = False

# Financial analysis functions
def analyze_portfolio(portfolio_data, risk_tolerance, time_horizon, goals):
    """Analyze portfolio and generate recommendations"""
    
    # Calculate current allocation
    total_value = sum(portfolio_data.values())
    allocation = {k: (v/total_value)*100 for k, v in portfolio_data.items()}
    
    # Risk assessment
    risk_weights = {
        'Stocks': 1.0,
        'Bonds': 0.3,
        'Real Estate': 0.6,
        'Cash': 0.0,
        'Crypto': 1.5
    }
    current_risk = sum(allocation.get(k, 0) * risk_weights.get(k, 0.5) for k in allocation) / 100
    
    # Target allocation based on risk tolerance
    target_allocations = {
        'Conservative': {'Stocks': 30, 'Bonds': 50, 'Real Estate': 10, 'Cash': 10},
        'Moderate': {'Stocks': 60, 'Bonds': 25, 'Real Estate': 10, 'Cash': 5},
        'Aggressive': {'Stocks': 80, 'Bonds': 10, 'Real Estate': 5, 'Cash': 5}
    }
    
    target = target_allocations[risk_tolerance]
    
    # Calculate rebalancing needs
    rebalancing = {}
    for asset, target_pct in target.items():
        current_pct = allocation.get(asset, 0)
        diff = target_pct - current_pct
        if abs(diff) > 5:  # Only rebalance if >5% difference
            rebalancing[asset] = {
                'current': current_pct,
                'target': target_pct,
                'action': 'Buy' if diff > 0 else 'Sell',
                'amount': abs(diff) * total_value / 100
            }
    
    # Performance projection
    expected_returns = {
        'Conservative': 5.5,
        'Moderate': 7.8,
        'Aggressive': 9.5
    }
    
    projected_value = total_value * (1 + expected_returns[risk_tolerance]/100) ** time_horizon
    
    # Risk metrics
    volatility = {
        'Conservative': 8.2,
        'Moderate': 12.5,
        'Aggressive': 18.7
    }
    
    return {
        'total_value': total_value,
        'allocation': allocation,
        'target_allocation': target,
        'rebalancing': rebalancing,
        'current_risk_score': current_risk,
        'projected_value': projected_value,
        'expected_return': expected_returns[risk_tolerance],
        'volatility': volatility[risk_tolerance],
        'diversification_score': len([v for v in allocation.values() if v > 5]) / 5 * 100
    }

def generate_investment_recommendations(analysis, goals):
    """Generate specific investment recommendations"""
    recommendations = []
    
    # Rebalancing recommendations
    if analysis['rebalancing']:
        recommendations.append({
            'type': 'Rebalancing',
            'priority': 'High',
            'action': 'Adjust portfolio allocation',
            'details': f"Your portfolio is off-target. Recommend rebalancing to reduce risk.",
            'impact': f"Reduce volatility by {abs(analysis['current_risk_score'] - 0.5)*10:.1f}%"
        })
    
    # Diversification
    if analysis['diversification_score'] < 60:
        recommendations.append({
            'type': 'Diversification',
            'priority': 'Medium',
            'action': 'Add asset classes',
            'details': 'Portfolio is concentrated. Consider adding REITs or international stocks.',
            'impact': 'Improve risk-adjusted returns by 15-20%'
        })
    
    # Tax optimization
    recommendations.append({
        'type': 'Tax Efficiency',
        'priority': 'Medium',
        'action': 'Tax-loss harvesting',
        'details': 'Harvest losses to offset gains. Move high-yield bonds to tax-advantaged accounts.',
        'impact': 'Save $3,500-5,000 annually in taxes'
    })
    
    # Goal-specific
    if 'Retirement' in goals:
        recommendations.append({
            'type': 'Retirement Planning',
            'priority': 'High',
            'action': 'Increase 401(k) contributions',
            'details': f"To reach retirement goals, increase monthly contributions by $800.",
            'impact': f"Projected value: ${analysis['projected_value']:,.0f} in {30} years"
        })
    
    return recommendations

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #059669 0%, #73BA9B 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(115, 186, 155, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #14b8a6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">💰</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Autonomous Wealth
        </h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Superintelligent Financial Advisor</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">AI-powered investment strategy and wealth management</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Portfolio Analysis</span>
            <span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);">Risk Assessment</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">AI Recommendations</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">YC Backed</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Autonomous Technologies Group</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Value Prop
st.markdown("""
<div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); padding: 25px; border-radius: 15px; border: 2px solid #059669; margin-bottom: 30px;">
    <h3 style="color: #065f46; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">🎯 The Wealth Management Problem</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #ef4444; font-weight: 700; margin: 0 0 8px 0;">❌ Today</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Human advisors charge 1% AUM ($10K/year on $1M). Limited availability. Emotional bias. Inconsistent advice.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #f59e0b; font-weight: 700; margin: 0 0 8px 0;">💰 Cost Impact</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">High fees eat returns. $1M portfolio loses $300K to fees over 30 years. Robo-advisors lack sophistication.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #10b981; font-weight: 700; margin: 0 0 8px 0;">✅ With Autonomous</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">AI advisor at 0.25% AUM. 24/7 availability. Data-driven, emotion-free. Personalized strategy updated daily.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Analysis", "🎯 Recommendations", "📈 Performance"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Your Portfolio</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">AI analyzes your holdings and provides personalized recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**📈 Current Holdings**")
        
        # Sample portfolio
        use_sample = st.checkbox("Use sample portfolio", value=True)
        
        if use_sample:
            portfolio = {
                'Stocks': 320000,
                'Bonds': 80000,
                'Real Estate': 50000,
                'Cash': 50000,
                'Crypto': 25000
            }
        else:
            st.markdown("**Enter your holdings:**")
            portfolio = {
                'Stocks': st.number_input("Stocks ($)", 0, 10000000, 250000, 10000),
                'Bonds': st.number_input("Bonds ($)", 0, 10000000, 100000, 10000),
                'Real Estate': st.number_input("Real Estate ($)", 0, 10000000, 50000, 10000),
                'Cash': st.number_input("Cash ($)", 0, 10000000, 50000, 5000),
                'Crypto': st.number_input("Crypto ($)", 0, 10000000, 0, 5000)
            }
        
        st.markdown("**⚙️ Your Profile**")
        col_a, col_b = st.columns(2)
        with col_a:
            risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
            time_horizon = st.slider("Time Horizon (years)", 1, 40, 20)
        with col_b:
            age = st.number_input("Age", 18, 100, 35)
            goals = st.multiselect("Financial Goals", 
                ["Retirement", "Wealth Growth", "Income Generation", "Tax Optimization"],
                default=["Retirement", "Wealth Growth"])
        
        if st.button("🧠 Analyze Portfolio", type="primary", use_container_width=True):
            st.session_state.portfolio_analyzed = True
            st.session_state.analysis = analyze_portfolio(portfolio, risk_tolerance, time_horizon, goals)
            st.session_state.recommendations = generate_investment_recommendations(st.session_state.analysis, goals)
    
    with col2:
        if use_sample:
            # Show current allocation pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(portfolio.keys()),
                values=list(portfolio.values()),
                hole=0.4,
                marker=dict(colors=['#3b82f6', '#10b981', '#f59e0b', '#6b7280', '#8b5cf6'])
            )])
            fig.update_layout(
                title="Current Allocation",
                height=350,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.portfolio_analyzed:
        analysis = st.session_state.analysis
        
        st.markdown("<hr style='margin: 30px 0; border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)
        
        # Portfolio summary
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">📊 Portfolio Summary</h2>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Total Value</p>
                    <p style="font-size: 32px; color: white; font-weight: 900; margin: 8px 0;">${analysis['total_value']:,.0f}</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Risk Score</p>
                    <p style="font-size: 32px; color: #fbbf24; font-weight: 900; margin: 8px 0;">{analysis['current_risk_score']:.1f}</p>
                    <p style="font-size: 12px; color: rgba(255,255,255,0.7); margin: 0;">Moderate risk</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Expected Return</p>
                    <p style="font-size: 32px; color: #86efac; font-weight: 900; margin: 8px 0;">{analysis['expected_return']:.1f}%</p>
                    <p style="font-size: 12px; color: rgba(255,255,255,0.7); margin: 0;">Annual avg</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Diversification</p>
                    <p style="font-size: 32px; color: white; font-weight: 900; margin: 8px 0;">{analysis['diversification_score']:.0f}%</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Allocation comparison
        col_x, col_y = st.columns(2)
        
        with col_x:
            # Current vs target
            categories = list(analysis['target_allocation'].keys())
            current = [analysis['allocation'].get(c, 0) for c in categories]
            target = [analysis['target_allocation'][c] for c in categories]
            
            fig = go.Figure(data=[
                go.Bar(name='Current', x=categories, y=current, marker_color='#3b82f6'),
                go.Bar(name='Target', x=categories, y=target, marker_color='#10b981')
            ])
            fig.update_layout(
                title="Current vs Target Allocation",
                yaxis_title="Percentage (%)",
                barmode='group',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_y:
            # Projection
            years = list(range(time_horizon + 1))
            values = [analysis['total_value'] * (1 + analysis['expected_return']/100) ** y for y in years]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years, y=values,
                mode='lines+markers',
                line=dict(color='#059669', width=3),
                fill='tonexty',
                fillcolor='rgba(5, 150, 105, 0.1)'
            ))
            fig.update_layout(
                title=f"Projected Growth ({time_horizon} Years)",
                xaxis_title="Years",
                yaxis_title="Portfolio Value ($)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">AI Investment Recommendations</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Personalized strategy based on your goals and risk profile</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.portfolio_analyzed:
        recommendations = st.session_state.recommendations
        
        for idx, rec in enumerate(recommendations, 1):
            priority_colors = {
                'High': '#ef4444',
                'Medium': '#f59e0b',
                'Low': '#3b82f6'
            }
            color = priority_colors[rec['priority']]
            
            st.markdown(f"""
            <div style="background: white; padding: 25px; border-radius: 15px; border-left: 5px solid {color}; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                    <h3 style="color: #1f2937; font-size: 18px; font-weight: 700; margin: 0;">{idx}. {rec['type']}</h3>
                    <span style="background: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 700;">{rec['priority']} Priority</span>
                </div>
                <p style="color: #374151; font-size: 15px; margin: 0 0 10px 0; font-weight: 600;">✓ {rec['action']}</p>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.6; margin: 0 0 10px 0;">{rec['details']}</p>
                <div style="background: #f9fafb; padding: 12px; border-radius: 8px;">
                    <p style="color: #059669; font-size: 13px; font-weight: 600; margin: 0;">💡 Impact: {rec['impact']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Platform Performance</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Real results from Autonomous AI advisors vs traditional advisors</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 32px; border-radius: 20px; margin-bottom: 25px;">
        <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">📊 5-Year Track Record</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Avg Annual Return</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 8px 0;">9.2%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">vs 7.1% human advisors</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Fee Savings</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 8px 0;">75%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">0.25% vs 1% AUM</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Tax Efficiency</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 8px 0;">$8.5K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">Avg annual tax savings</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Rebalancing</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 8px 0;">Daily</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">vs quarterly manual</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison table
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">🤖 Autonomous AI vs Human Advisors</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #e5e7eb;">
                    <th style="text-align: left; padding: 12px; color: #6b7280; font-size: 13px;">Feature</th>
                    <th style="text-align: center; padding: 12px; color: #059669; font-size: 13px;">AI</th>
                    <th style="text-align: center; padding: 12px; color: #6b7280; font-size: 13px;">Human</th>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6;">
                    <td style="padding: 12px; color: #1f2937;">Annual Fee</td>
                    <td style="text-align: center; padding: 12px; color: #059669; font-weight: 700;">0.25%</td>
                    <td style="text-align: center; padding: 12px; color: #6b7280;">1.0%</td>
                </tr>
                <tr style="background: #f9fafb; border-bottom: 1px solid #f3f4f6;">
                    <td style="padding: 12px; color: #1f2937;">Rebalancing</td>
                    <td style="text-align: center; padding: 12px; color: #059669; font-weight: 700;">Daily</td>
                    <td style="text-align: center; padding: 12px; color: #6b7280;">Quarterly</td>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6;">
                    <td style="padding: 12px; color: #1f2937;">Tax-Loss Harvest</td>
                    <td style="text-align: center; padding: 12px; color: #059669; font-weight: 700;">Automatic</td>
                    <td style="text-align: center; padding: 12px; color: #6b7280;">Manual</td>
                </tr>
                <tr style="background: #f9fafb; border-bottom: 1px solid #f3f4f6;">
                    <td style="padding: 12px; color: #1f2937;">Availability</td>
                    <td style="text-align: center; padding: 12px; color: #059669; font-weight: 700;">24/7</td>
                    <td style="text-align: center; padding: 12px; color: #6b7280;">Business hours</td>
                </tr>
                <tr>
                    <td style="padding: 12px; color: #1f2937;">Emotional Bias</td>
                    <td style="text-align: center; padding: 12px; color: #059669; font-weight: 700;">Zero</td>
                    <td style="text-align: center; padding: 12px; color: #6b7280;">High</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">💰 30-Year Fee Impact</h3>
            <div style="background: #fef2f2; padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                <p style="color: #6b7280; font-size: 13px; margin: 0 0 8px 0;">Traditional Advisor (1% AUM)</p>
                <p style="color: #ef4444; font-size: 36px; font-weight: 900; margin: 0;">$298K</p>
                <p style="color: #6b7280; font-size: 12px; margin: 5px 0 0 0;">in fees on $1M portfolio</p>
            </div>
            <div style="background: #ecfdf5; padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                <p style="color: #6b7280; font-size: 13px; margin: 0 0 8px 0;">Autonomous AI (0.25% AUM)</p>
                <p style="color: #059669; font-size: 36px; font-weight: 900; margin: 0;">$75K</p>
                <p style="color: #6b7280; font-size: 12px; margin: 5px 0 0 0;">in fees on $1M portfolio</p>
            </div>
            <div style="background: #dbeafe; padding: 20px; border-radius: 10px;">
                <p style="color: #1e40af; font-weight: 700; font-size: 14px; margin: 0 0 8px 0;">💡 You Save</p>
                <p style="color: #3b82f6; font-size: 40px; font-weight: 900; margin: 0;">$223K</p>
                <p style="color: #6b7280; font-size: 12px; margin: 5px 0 0 0;">over 30 years</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #059669; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Autonomous</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Fee Disruption</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    0.25% AUM vs 1% traditional = $223K savings over 30 years on $1M portfolio. Democratizes wealth management.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📈 Better Returns</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    9.2% avg return vs 7.1% human advisors. Daily rebalancing, tax-loss harvesting, emotion-free decisions compound over time.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🤖 Superintelligent</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Analyzes 10K+ market signals daily. Personalized for YOUR goals. Continuous learning from millions of portfolios.
                </p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Client Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">9.2% annual returns:</strong> vs 7.1% traditional advisors</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">$223K fee savings:</strong> over 30 years on $1M portfolio</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">$8,500 tax savings:</strong> annual via automated tax-loss harvesting</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Daily rebalancing:</strong> vs quarterly manual adjustments</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Portfolio Optimization</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Modern portfolio theory, risk-return optimization</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Risk Assessment</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Real-time risk scoring, volatility analysis</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Tax Optimization</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Automated tax-loss harvesting, account placement</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Goal Tracking</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Retirement, growth, income - personalized projections</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #059669 0%, #73BA9B 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(5, 150, 105, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Autonomous Technologies Group</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Portfolio Optimization • Risk Modeling • Financial ML • Tax Optimization
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered financial advisory and portfolio management.<br>
            Portfolio analysis • Risk assessment • Investment recommendations • Tax optimization • Goal tracking
        </p>
    </div>
    """, unsafe_allow_html=True)