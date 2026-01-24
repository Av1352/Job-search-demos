"""
Use Dots - Global Payout Intelligence
AI-powered contractor payment optimization across borders
Built for Use Dots by Anju Nandhakumar
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from utils.sidebar import render_sidebar
render_sidebar()

# Page config
st.set_page_config(
    page_title="Use Dots Demo - Anju Vilashni",
    page_icon="💳",
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

def generate_payout_data():
    """Generate sample contractor payout data"""
    np.random.seed(42)
    countries = ['USA', 'UK', 'Canada', 'India', 'Brazil', 'Germany']
    
    data = []
    for i in range(50):
        country = np.random.choice(countries)
        amount = np.random.uniform(500, 15000)
        
        processing_days = {
            'USA': np.random.uniform(1, 3),
            'UK': np.random.uniform(1, 2),
            'Canada': np.random.uniform(2, 4),
            'India': np.random.uniform(3, 7),
            'Brazil': np.random.uniform(4, 8),
            'Germany': np.random.uniform(1, 3)
        }
        
        days = processing_days[country]
        fee_pct = 1.5 if country in ['USA', 'UK', 'Germany'] else 2.5
        fee = amount * fee_pct / 100
        
        data.append({
            'contractor_id': f'CTR{1000+i}',
            'country': country,
            'amount': round(amount, 2),
            'fee': round(fee, 2),
            'net_payout': round(amount - fee, 2),
            'processing_days': round(days, 1),
            'status': np.random.choice(['Pending', 'Processing', 'Completed'], p=[0.2, 0.3, 0.5])
        })
    
    return pd.DataFrame(data)

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
            <span style="font-size: 56px;">💳</span>
        </div>
        <h1 style="
            font-size: 58px;
            font-weight: 900;
            color: white;
            margin: 0 0 18px 0;
            text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            Use Dots Payout Intelligence
        </h1>
        <p style="
            font-size: 28px;
            color: rgba(255,255,255,0.95);
            font-weight: 700;
            margin: 15px 0;
        ">
            Global Payouts • Smart Routing
        </p>
        <p style="
            font-size: 18px;
            color: rgba(255,255,255,0.85);
            font-weight: 500;
            margin-bottom: 25px;
        ">
            AI-powered contractor payment optimization across 6 countries
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
            <span style="background:#ec4899;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Multi-Country</span>
            <span style="background:#8b5cf6;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Smart Routing</span>
            <span style="background:#f59e0b;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Cost Optimization</span>
            <span style="background:#10b981;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Real-Time</span>
        </div>
        <p style="
            font-size: 16px;
            color: rgba(255,255,255,0.9);
            margin-top: 28px;
            font-weight: 600;
        ">
            Built for <strong style="color:white;">Use Dots</strong>
            by <strong style="color:white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["📊 Payout Analytics", "🎯 Payout Optimizer"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Global Payment Performance</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Track volume, fees, and processing times across 6 countries</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Refresh Dashboard", key="refresh", use_container_width=True, type="primary"):
        df = generate_payout_data()
        
        total_volume = df['amount'].sum()
        total_fees = df['fee'].sum()
        avg_processing = df['processing_days'].mean()
        
        by_country = df.groupby('country').agg({
            'amount': 'sum',
            'processing_days': 'mean',
            'contractor_id': 'count'
        }).sort_values('amount', ascending=False)
        by_country.columns = ['volume', 'avg_days', 'count']
        
        # Summary
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🌍 Global Payout Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Total Volume</p>
                <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">${total_volume/1000:.0f}K</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Processed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(251, 191, 36, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(251, 191, 36, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Total Fees</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0;">${total_fees:,.0f}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Collected</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Avg Processing</p>
                <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">{avg_processing:.1f}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Days</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(16, 185, 129, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Contractors</p>
                <p style="font-size: 48px; color: #10b981; font-weight: 900; margin: 0;">50</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Active</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Country breakdown
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🌎 Performance by Country</h3>
        </div>
        """, unsafe_allow_html=True)
        
        colors = ['#667eea', '#10b981', '#ec4899', '#f59e0b', '#3b82f6', '#8b5cf6']
        for idx, (country, row) in enumerate(by_country.iterrows()):
            st.markdown(f"""
            <div style="background: white; border-left: 5px solid {colors[idx]}; border-radius: 12px; padding: 18px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 4px 0;">{country}</p>
                        <p style="font-size: 13px; color: #6b7280; margin: 0;">{int(row['count'])} contractors • {row['avg_days']:.1f} day avg processing</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="font-size: 28px; color: {colors[idx]}; font-weight: 900; margin: 0;">${row['volume']:,.0f}</p>
                        <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">Total volume</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_volume = go.Figure(data=[
                go.Bar(x=by_country.index, y=by_country['volume'],
                       marker=dict(color=colors[:len(by_country)]))
            ])
            fig_volume.update_layout(title="Payout Volume by Country", xaxis_title="Country", yaxis_title="Volume ($)", height=400)
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col2:
            fig_processing = go.Figure(data=[
                go.Bar(x=by_country.index, y=by_country['avg_days'], marker=dict(color='#3b82f6'))
            ])
            fig_processing.update_layout(title="Average Processing Time by Country", xaxis_title="Country", yaxis_title="Days", height=400)
            st.plotly_chart(fig_processing, use_container_width=True)
        
        # Recent payouts
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Optimize Your Contractor Payment</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Get the best payment method for your contractor's location</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amount = st.number_input("Payout Amount ($)", min_value=1, value=5000)
    
    with col2:
        country = st.selectbox("Contractor Country", ['USA', 'UK', 'Canada', 'India', 'Brazil', 'Germany'])
    
    with col3:
        urgency = st.radio("Urgency Level", ['Urgent', 'Normal', 'Standard'])
    
    if st.button("🚀 Get Optimal Method", use_container_width=True, type="primary"):
        # Determine method
        methods = {
            'instant': {'fee': 3.5, 'time': '< 1 hour', 'availability': ['USA', 'UK', 'Germany']},
            'fast': {'fee': 2.0, 'time': '1-2 days', 'availability': ['USA', 'UK', 'Canada', 'Germany']},
            'standard': {'fee': 1.5, 'time': '3-5 days', 'availability': ['USA', 'UK', 'Canada', 'Germany', 'India', 'Brazil']}
        }
        
        if urgency == 'Urgent' and country in methods['instant']['availability']:
            method = 'instant'
        elif urgency == 'Normal' and country in methods['fast']['availability']:
            method = 'fast'
        else:
            method = 'standard'
        
        selected = methods[method]
        fee = amount * selected['fee'] / 100
        net = amount - fee
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">💸 Optimal Payout Method</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Amount</p>
                <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">${amount:,.0f}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Payout</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Country</p>
                <p style="font-size: 32px; color: #667eea; font-weight: 900; margin: 0;">{country}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Destination</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: rgba(251, 191, 36, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(251, 191, 36, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Fee</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0;">${fee:.2f}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">{selected['fee']}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(16, 185, 129, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Time</p>
                <p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">{selected['time'].split()[0]}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">{selected['time']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; margin-top: 25px;">
            <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">✅ Recommended: {method.upper()} Transfer</h3>
            <div style="background: white; border-radius: 12px; padding: 22px; margin-bottom: 18px;">
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px;">
                    <div>
                        <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Net Payout</p>
                        <p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">${net:,.2f}</p>
                    </div>
                    <div>
                        <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Urgency Level</p>
                        <p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">{urgency}</p>
                    </div>
                </div>
            </div>
            <div style="background: rgba(16, 185, 129, 0.1); border-radius: 10px; padding: 18px;">
                <p style="font-size: 15px; color: #065f46; font-weight: 700; margin: 0 0 10px 0;">Why This Method?</p>
                <div style="display: grid; gap: 8px;">
                    <p style="font-size: 14px; color: #059669; margin: 0;">✓ Reliable for {country} transfers</p>
                    <p style="font-size: 14px; color: #059669; margin: 0;">✓ Compliant with local regulations</p>
                    <p style="font-size: 14px; color: #059669; margin: 0;">✓ Full tracking and notifications</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<hr style='border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Use Dots</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
        <strong>Tech Stack:</strong> Python • Streamlit • Routing Logic • Global Payment APIs
    </p>
</div>
""", unsafe_allow_html=True)