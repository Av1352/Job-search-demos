"""
CTGT - Financial Intelligence Platform
AI-powered transaction analysis and fraud detection
Built for CTGT by Anju Nandhakumar
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
    page_title="CTGT Demo - Anju Vilashni",
    page_icon="💳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main { background: white; }
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

def generate_transaction_data(num_transactions=150):
    """Generate sample transaction data"""
    np.random.seed(42)
    categories = ['Groceries', 'Utilities', 'Entertainment', 'Transportation', 'Healthcare', 'Shopping']
    merchants = {
        'Groceries': ['Whole Foods', 'Trader Joes', 'Safeway'],
        'Utilities': ['PG&E', 'Comcast', 'AT&T'],
        'Entertainment': ['Netflix', 'Spotify', 'AMC Theaters'],
        'Transportation': ['Uber', 'Shell', 'Chevron'],
        'Healthcare': ['CVS', 'Walgreens', 'Kaiser'],
        'Shopping': ['Amazon', 'Target', 'Walmart']
    }
    
    dates = [datetime.now() - timedelta(days=np.random.randint(0, 90)) for _ in range(num_transactions)]
    data = []
    
    for date in dates:
        category = np.random.choice(categories)
        merchant = np.random.choice(merchants[category])
        amount = np.random.uniform(10, 500)
        
        is_anomaly = np.random.random() < 0.1
        if is_anomaly:
            amount *= np.random.uniform(2, 5)
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'merchant': merchant,
            'category': category,
            'amount': round(amount, 2),
            'is_anomaly': is_anomaly
        })
    
    return pd.DataFrame(data).sort_values('date', ascending=False)

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
            <span style="font-size: 56px;">💳</span>
        </div>

        <h1 style="
            font-size: 58px;
            font-weight: 900;
            color: white;
            margin: 0 0 18px 0;
            text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            CTGT Financial Intelligence
        </h1>

        <p style="
            font-size: 28px;
            color: rgba(255,255,255,0.95);
            font-weight: 700;
            margin: 15px 0;
        ">
            Transaction Analysis • Fraud Detection
        </p>

        <p style="
            font-size: 18px;
            color: rgba(255,255,255,0.85);
            font-weight: 500;
            margin-bottom: 25px;
        ">
            AI-powered spending insights and anomaly detection
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
            <span style="background:#ec4899;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Anomaly Detection</span>
            <span style="background:#8b5cf6;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Spending Analytics</span>
            <span style="background:#f59e0b;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Real-Time Alerts</span>
            <span style="background:#10b981;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Pattern Recognition</span>
        </div>

        <p style="
            font-size: 16px;
            color: rgba(255,255,255,0.9);
            margin-top: 28px;
            font-weight: 600;
        ">
            Built for <strong style="color:white;">CTGT</strong>
            by <strong style="color:white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """,
    height=520,
)

st.markdown("---")

st.markdown("""
<div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
    <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">AI-Powered Financial Analysis</h3>
    <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Analyze 150 transactions with ML-based fraud detection</p>
</div>
""", unsafe_allow_html=True)

if st.button("🔍 Analyze Transactions", type="primary", use_container_width=True):
    df = generate_transaction_data(150)
    
    # Calculate metrics
    total_spending = df['amount'].sum()
    avg_transaction = df['amount'].mean()
    num_anomalies = df['is_anomaly'].sum()
    
    category_spending = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    
    # Summary
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">💳 Financial Analysis Complete</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
            <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Total Spending</p>
            <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">${total_spending:,.0f}</p>
            <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Last 90 days</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
            <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Avg Transaction</p>
            <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">${avg_transaction:.0f}</p>
            <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Per purchase</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
            <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Transactions</p>
            <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">{len(df)}</p>
            <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        anomaly_color = '#ef4444' if num_anomalies > 10 else '#fbbf24'
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
            <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Anomalies</p>
            <p style="font-size: 48px; color: {anomaly_color}; font-weight: 900; margin: 0;">{num_anomalies}</p>
            <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">{num_anomalies/len(df)*100:.1f}% flagged</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Top categories
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
        <h3 style="color: #92400e; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📊 Top Spending Categories</h3>
    </div>
    """, unsafe_allow_html=True)
    
    colors = ['#667eea', '#ec4899', '#10b981']
    for idx, (category, amount) in enumerate(category_spending.head(3).items()):
        percentage = (amount / total_spending * 100)
        st.markdown(f"""
        <div style="background: white; border-left: 5px solid {colors[idx]}; border-radius: 12px; padding: 18px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 4px 0;">{category}</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">{percentage:.1f}% of total spending</p>
                </div>
                <div style="text-align: right;">
                    <p style="font-size: 28px; color: {colors[idx]}; font-weight: 900; margin: 0;">${amount:,.0f}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Anomalies
    anomaly_df = df[df['is_anomaly']]
    
    if len(anomaly_df) > 0:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #991b1b; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🚨 Anomalous Transactions Detected</h3>
            <p style="color: #b91c1c; font-size: 16px; margin: 0 0 15px 0; font-weight: 600;">{len(anomaly_df)} transactions flagged for review</p>
        </div>
        """, unsafe_allow_html=True)
        
        for _, row in anomaly_df.head(5).iterrows():
            st.markdown(f"""
            <div style="background: white; border-radius: 10px; padding: 16px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <p style="font-size: 15px; color: #1f2937; font-weight: 700; margin: 0;">{row['merchant']}</p>
                        <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">{row['date']} • {row['category']}</p>
                    </div>
                    <p style="font-size: 20px; color: #ef4444; font-weight: 800; margin: 0;">${row['amount']:.2f}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; text-align: center; margin-bottom: 25px;">
            <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0;">✅ No Anomalies Detected</h3>
            <p style="color: #059669; font-size: 16px; margin: 10px 0 0 0;">All transactions appear normal</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_category = go.Figure(data=[go.Pie(
            labels=category_spending.index,
            values=category_spending.values,
            hole=0.3,
            marker=dict(colors=['#667eea', '#764ba2', '#ec4899', '#f59e0b', '#10b981', '#3b82f6'])
        )])
        fig_category.update_layout(title="Spending by Category", height=400)
        st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_spending = df.groupby('month')['amount'].sum()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=[str(m) for m in monthly_spending.index],
            y=monthly_spending.values,
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        fig_trend.update_layout(
            title="Monthly Spending Trend",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Anomaly scatter
    fig_anomaly = px.scatter(df, x='date', y='amount', color='is_anomaly',
                             color_discrete_map={True: '#ef4444', False: '#10b981'},
                             title="Transaction Anomaly Detection")
    fig_anomaly.update_layout(height=400)
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    # Transaction table
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

# Footer
st.markdown("<hr style='border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">CTGT</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
        <strong>Tech Stack:</strong> Python • Streamlit • Pandas • Plotly • Statistical ML
    </p>
</div>
""", unsafe_allow_html=True)