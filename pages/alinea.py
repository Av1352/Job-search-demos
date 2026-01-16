"""
Alinea Invest - AI Investment Intelligence
AI-powered portfolio analysis and investment recommendations
Built for Alinea Invest by Anju Nandhakumar
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Alinea Invest Demo - Anju Vilashni",
    page_icon="📈",
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

def generate_portfolio_data():
    """Generate sample investment portfolio"""
    np.random.seed(42)
    
    stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'sector': 'Technology', 'shares': 10, 'price': 178.50},
        {'symbol': 'MSFT', 'name': 'Microsoft', 'sector': 'Technology', 'shares': 8, 'price': 378.91},
        {'symbol': 'GOOGL', 'name': 'Alphabet', 'sector': 'Technology', 'shares': 5, 'price': 140.23},
        {'symbol': 'AMZN', 'name': 'Amazon', 'sector': 'Consumer', 'shares': 6, 'price': 178.35},
        {'symbol': 'NVDA', 'name': 'NVIDIA', 'sector': 'Technology', 'shares': 15, 'price': 495.22},
        {'symbol': 'TSLA', 'name': 'Tesla', 'sector': 'Automotive', 'shares': 4, 'price': 242.84},
        {'symbol': 'META', 'name': 'Meta', 'sector': 'Technology', 'shares': 7, 'price': 484.03},
        {'symbol': 'JPM', 'name': 'JPMorgan', 'sector': 'Financial', 'shares': 12, 'price': 198.42},
        {'symbol': 'V', 'name': 'Visa', 'sector': 'Financial', 'shares': 9, 'price': 278.15},
        {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'sector': 'Healthcare', 'shares': 11, 'price': 156.32}
    ]
    
    portfolio = []
    for stock in stocks:
        cost_basis = stock['price'] * np.random.uniform(0.85, 0.95)
        shares = stock['shares']
        current_value = stock['price'] * shares
        total_cost = cost_basis * shares
        gain_loss = current_value - total_cost
        gain_loss_pct = (gain_loss / total_cost) * 100
        
        portfolio.append({
            'symbol': stock['symbol'],
            'name': stock['name'],
            'sector': stock['sector'],
            'shares': shares,
            'current_price': stock['price'],
            'cost_basis': round(cost_basis, 2),
            'current_value': round(current_value, 2),
            'total_cost': round(total_cost, 2),
            'gain_loss': round(gain_loss, 2),
            'gain_loss_pct': round(gain_loss_pct, 2)
        })
    
    return pd.DataFrame(portfolio)

def analyze_portfolio(df):
    """Analyze portfolio composition and performance"""
    total_value = df['current_value'].sum()
    total_cost = df['total_cost'].sum()
    total_gain_loss = df['gain_loss'].sum()
    total_return_pct = (total_gain_loss / total_cost) * 100
    
    sector_allocation = df.groupby('sector')['current_value'].sum()
    
    return total_value, total_cost, total_gain_loss, total_return_pct, sector_allocation

def create_allocation_chart(sector_allocation):
    """Create sector allocation pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=sector_allocation.index,
        values=sector_allocation.values,
        hole=0.4,
        marker=dict(colors=['#667eea', '#764ba2', '#ec4899', '#10b981', '#f59e0b'])
    )])
    fig.update_layout(title="Portfolio Allocation by Sector", height=400)
    return fig

def create_performance_chart(df):
    """Create performance bar chart"""
    df_sorted = df.sort_values('gain_loss_pct', ascending=True)
    colors = ['#ef4444' if x < 0 else '#10b981' for x in df_sorted['gain_loss_pct']]
    
    fig = go.Figure(data=[go.Bar(
        x=df_sorted['gain_loss_pct'],
        y=df_sorted['symbol'],
        orientation='h',
        marker=dict(color=colors)
    )])
    fig.update_layout(
        title="Performance by Stock (% Return)",
        xaxis_title="Return (%)",
        yaxis_title="Stock",
        height=500
    )
    return fig

# Header
st.markdown(
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
            <span style="font-size: 56px;">📈</span>
        </div>
        <h1 style="
            font-size: 58px;
            font-weight: 900;
            color: white;
            margin: 0 0 18px 0;
            text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            Alinea Investment Intelligence
        </h1>
        <p style="
            font-size: 28px;
            color: rgba(255,255,255,0.95);
            font-weight: 700;
            margin: 15px 0;
        ">
            Portfolio Analysis • AI Recommendations
        </p>
        <p style="
            font-size: 18px;
            color: rgba(255,255,255,0.85);
            font-weight: 500;
            margin-bottom: 25px;
        ">
            Smart investing for Gen Z with AI-powered insights
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
            <span style="background:#ec4899;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Portfolio Tracking</span>
            <span style="background:#8b5cf6;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">AI Recommendations</span>
            <span style="background:#f59e0b;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Diversification</span>
            <span style="background:#10b981;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Performance</span>
        </div>
        <p style="
            font-size: 16px;
            color: rgba(255,255,255,0.9);
            margin-top: 28px;
            font-weight: 600;
        ">
            Built for <strong style="color:white;">Alinea Invest</strong>
            by <strong style="color:white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Overview", "🤖 AI Recommendations", "🎯 Diversification"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Your Investment Portfolio</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Analyze 10 positions across 5 sectors</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📈 Analyze Portfolio", key="analyze"):
        df = generate_portfolio_data()
        total_value, total_cost, total_gain_loss, total_return_pct, sector_allocation = analyze_portfolio(df)
        
        # Summary
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Portfolio Summary</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Portfolio Value</p>
                <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">${total_value/1000:.1f}K</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Current</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Total Invested</p>
                <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">${total_cost/1000:.1f}K</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Cost basis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            gain_color = '#10b981' if total_gain_loss > 0 else '#ef4444'
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Total Gain/Loss</p>
                <p style="font-size: 48px; color: {gain_color}; font-weight: 900; margin: 0;">{'+'if total_gain_loss > 0 else ''}{total_gain_loss/1000:.1f}K</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Unrealized</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            return_color = '#10b981' if total_return_pct > 0 else '#ef4444'
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Total Return</p>
                <p style="font-size: 48px; color: {return_color}; font-weight: 900; margin: 0;">{'+'if total_return_pct > 0 else ''}{total_return_pct:.1f}%</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Performance</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_allocation = go.Figure(data=[go.Pie(
                labels=sector_allocation.index,
                values=sector_allocation.values,
                hole=0.4,
                marker=dict(colors=['#667eea', '#764ba2', '#ec4899', '#10b981', '#f59e0b'])
            )])
            fig_allocation.update_layout(title="Portfolio Allocation by Sector", height=400)
            st.plotly_chart(fig_allocation, use_container_width=True)
        
        with col2:
            df_sorted = df.sort_values('gain_loss_pct', ascending=True)
            colors = ['#ef4444' if x < 0 else '#10b981' for x in df_sorted['gain_loss_pct']]
            
            fig_performance = go.Figure(data=[go.Bar(
                x=df_sorted['gain_loss_pct'],
                y=df_sorted['symbol'],
                orientation='h',
                marker=dict(color=colors)
            )])
            fig_performance.update_layout(
                title="Performance by Stock (% Return)",
                xaxis_title="Return (%)",
                yaxis_title="Stock",
                height=400
            )
            st.plotly_chart(fig_performance, use_container_width=True)
        
        # Portfolio table
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Store in session state
        st.session_state.portfolio_df = df
        st.session_state.sector_allocation = sector_allocation
        st.session_state.total_value = total_value

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">AI-Powered Investment Recommendations</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Personalized insights based on your portfolio analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'portfolio_df' in st.session_state:
        df = st.session_state.portfolio_df
        sector_allocation = st.session_state.sector_allocation
        total_value = st.session_state.total_value
        
        # Tech allocation warning
        tech_allocation = sector_allocation.get('Technology', 0) / total_value * 100
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🤖 AI Investment Recommendations</h2>            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; border: 2px solid rgba(255,255,255,0.2); margin-bottom: 15px;">
                <p style="font-size: 18px; color: white; font-weight: 800; margin: 0 0 10px 0;">⚠️ Portfolio Concentration Risk</p>
                <p style="font-size: 15px; color: rgba(255,255,255,0.9); margin: 0; line-height: 1.6;">
                    Your portfolio is <strong>{tech_allocation:.0f}%</strong> allocated to Technology sector. 
                    Consider diversifying into Healthcare, Financial, or Consumer sectors to reduce risk.
                </p>
            </div>            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 18px; color: white; font-weight: 800; margin: 0 0 10px 0;">💡 Recommended Actions</p>
                <p style="font-size: 15px; color: rgba(255,255,255,0.9); margin: 0; line-height: 1.6;">
                    1. <strong>Rebalance:</strong> Reduce Tech exposure by 10-15%<br>
                    2. <strong>Diversify:</strong> Add more Healthcare and Financial positions<br>
                    3. <strong>Review:</strong> Monitor underperforming stocks for potential reallocation
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Top performers
        top_performers = df.nlargest(3, 'gain_loss_pct')
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🚀 Top Performers</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for _, stock in top_performers.iterrows():
            st.markdown(f"""
            <div style="background: white; border-left: 5px solid #10b981; border-radius: 12px; padding: 18px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 4px 0;">{stock['symbol']} - {stock['name']}</p>
                        <p style="font-size: 13px; color: #6b7280; margin: 0;">{stock['sector']} • ${stock['current_price']:.2f}</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="font-size: 28px; color: #10b981; font-weight: 900; margin: 0;">+{stock['gain_loss_pct']:.1f}%</p>
                        <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">${stock['gain_loss']:.0f} gain</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Underperformers
        underperformers = df.nsmallest(2, 'gain_loss_pct')
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #991b1b; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">⚠️ Review Needed</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for _, stock in underperformers.iterrows():
            st.markdown(f"""
            <div style="background: white; border-left: 5px solid #ef4444; border-radius: 12px; padding: 18px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 4px 0;">{stock['symbol']} - {stock['name']}</p>
                        <p style="font-size: 13px; color: #6b7280; margin: 0;">{stock['sector']} • ${stock['current_price']:.2f}</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="font-size: 28px; color: #ef4444; font-weight: 900; margin: 0;">{stock['gain_loss_pct']:.1f}%</p>
                        <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">${stock['gain_loss']:.0f} loss</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.1); border-radius: 10px; padding: 18px;">
            <p style="font-size: 14px; color: #991b1b; margin: 0; line-height: 1.6;">
                <strong>AI Insight:</strong> Consider setting stop-loss orders or reallocating capital from underperformers to higher-growth opportunities.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Click 'Analyze Portfolio' in the Portfolio Overview tab first!")

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #92400e; font-size: 22px; font-weight: 800; margin: 0;">Diversification Strategy</h3>
        <p style="color: #d97706; font-size: 14px; margin: 8px 0 0 0;">Optimize sector allocation to reduce risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'sector_allocation' in st.session_state:
        sector_allocation = st.session_state.sector_allocation
        total_value = st.session_state.total_value
        
        ideal_allocation = {
            'Technology': 30,
            'Financial': 20,
            'Healthcare': 20,
            'Consumer': 15,
            'Automotive': 15
        }
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #92400e; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🎯 Diversification Strategy</h3>
            <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;"><p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0 0 15px 0;">Ideal vs Current Allocation</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        for sector, ideal_pct in ideal_allocation.items():
            current_value = sector_allocation.get(sector, 0)
            current_pct = (current_value / total_value * 100)
            diff = current_pct - ideal_pct
            
            color = '#ef4444' if abs(diff) > 10 else '#10b981' if abs(diff) < 5 else '#f59e0b'
            
            st.markdown(f"""
            <div style="background: #f9fafb; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <p style="font-size: 14px; color: #1f2937; font-weight: 700; margin: 0;">{sector}</p>
                    <p style="font-size: 14px; color: {color}; font-weight: 700; margin: 0;">{current_pct:.1f}% (Target: {ideal_pct}%)</p>
                </div>
                <div style="background: #e5e7eb; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: {color}; height: 100%; width: {min(current_pct, 100)}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); border-radius: 10px; padding: 18px; margin-top: 20px;">
            <p style="font-size: 15px; color: #92400e; font-weight: 700; margin: 0 0 10px 0;">💡 Action Items:</p>
            <div style="display: grid; gap: 8px;">
                <p style="font-size: 14px; color: #78350f; margin: 0;">✓ Reduce Technology exposure by 10-15%</p>
                <p style="font-size: 14px; color: #78350f; margin: 0;">✓ Increase Healthcare allocation to 20%</p>
                <p style="font-size: 14px; color: #78350f; margin: 0;">✓ Add Financial sector positions (target 20%)</p>
                <p style="font-size: 14px; color: #78350f; margin: 0;">✓ Maintain diversification across 5+ sectors</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Click 'Analyze Portfolio' in the Portfolio Overview tab first!")

# Footer
st.markdown("<hr style='border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Alinea Invest</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
        <strong>Tech Stack:</strong> Python • Streamlit • Pandas • Plotly • Portfolio Analytics
    </p>
</div>
""", unsafe_allow_html=True)