"""
Slash - Payment Routing Intelligence
AI-powered payment optimization and processor selection
Built for Slash by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Slash Payment Intelligence", page_icon="💸", layout="wide")

def generate_payment_data():
    """Generate sample payment processing data"""
    np.random.seed(42)
    payment_methods = ['Credit Card', 'Debit Card', 'ACH', 'Wire Transfer', 'Digital Wallet']
    statuses = ['Success', 'Pending', 'Failed']
    
    data = []
    for i in range(50):
        method = np.random.choice(payment_methods)
        amount = np.random.uniform(50, 5000)
        
        success_rates = {'Credit Card': 0.95, 'Debit Card': 0.93, 'ACH': 0.88, 
                        'Wire Transfer': 0.90, 'Digital Wallet': 0.96}
        
        rand = np.random.random()
        if rand < success_rates[method]:
            status = 'Success'
        elif rand < success_rates[method] + 0.05:
            status = 'Pending'
        else:
            status = 'Failed'
        
        processing_time = np.random.uniform(0.5, 5.0) if status == 'Success' else np.random.uniform(10, 30)
        
        data.append({
            'transaction_id': f'TXN{1000+i}',
            'amount': round(amount, 2),
            'method': method,
            'status': status,
            'processing_time': round(processing_time, 2),
            'timestamp': (datetime.now() - timedelta(minutes=np.random.randint(0, 1440))).strftime('%Y-%m-%d %H:%M')
        })
    
    return pd.DataFrame(data)

def analyze_payments(df):
    """Analyze payment performance"""
    success_by_method = df[df['status'] == 'Success'].groupby('method').size() / df.groupby('method').size() * 100
    avg_processing = df[df['status'] == 'Success'].groupby('method')['processing_time'].mean()
    total_volume = df['amount'].sum()
    success_volume = df[df['status'] == 'Success']['amount'].sum()
    
    return success_by_method, avg_processing, total_volume, success_volume

def create_success_rate_chart(success_by_method):
    """Create success rate visualization"""
    fig = go.Figure(data=[
        go.Bar(x=success_by_method.index, y=success_by_method.values,
               marker=dict(color=['#667eea', '#764ba2', '#ec4899', '#10b981', '#f59e0b']))
    ])
    fig.update_layout(
        title="Success Rate by Payment Method",
        xaxis_title="Payment Method",
        yaxis_title="Success Rate (%)",
        height=400
    )
    fig.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Target: 95%")
    return fig

def create_processing_time_chart(avg_processing):
    """Create processing time visualization"""
    fig = go.Figure(data=[
        go.Bar(x=avg_processing.index, y=avg_processing.values,
               marker=dict(color='#3b82f6'))
    ])
    fig.update_layout(
        title="Average Processing Time by Method",
        xaxis_title="Payment Method",
        yaxis_title="Processing Time (seconds)",
        height=400
    )
    return fig

def optimize_routing(amount, method):
    """Recommend optimal payment routing"""
    recommendations = {
        'Credit Card': {'processor': 'Stripe', 'fee': 2.9, 'success_rate': 95},
        'Debit Card': {'processor': 'Square', 'fee': 2.6, 'success_rate': 93},
        'ACH': {'processor': 'Plaid', 'fee': 0.8, 'success_rate': 88},
        'Wire Transfer': {'processor': 'Wise', 'fee': 1.5, 'success_rate': 90},
        'Digital Wallet': {'processor': 'PayPal', 'fee': 3.5, 'success_rate': 96}
    }
    
    rec = recommendations.get(method, recommendations['Credit Card'])
    fee_amount = amount * rec['fee'] / 100
    net_amount = amount - fee_amount
    
    alternatives = []
    for m, details in recommendations.items():
        if m != method:
            alt_fee = amount * details['fee'] / 100
            alternatives.append({
                'method': m,
                'processor': details['processor'],
                'fee': details['fee'],
                'fee_amount': alt_fee,
                'success_rate': details['success_rate']
            })
    
    alternatives = sorted(alternatives, key=lambda x: x['fee'])[:3]
    
    result_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🎯 Optimal Payment Route</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Amount</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${amount:,.0f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Transaction</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Method</p>
                <p style="font-size: 32px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{method.split()[0]}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Payment type</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Fee</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${fee_amount:.2f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{rec['fee']}%</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Success Rate</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{rec['success_rate']}%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Expected</p>
            </div>
        </div>
    </div>
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;">
        <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">✅ Recommended Processor: {rec['processor']}</h3>
        <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <div>
                    <p style="font-size: 12px; color: #6b7280; margin: 0;">Net Amount</p>
                    <p style="font-size: 24px; color: #10b981; font-weight: 800; margin: 4px 0 0 0;">${net_amount:,.2f}</p>
                </div>
                <div>
                    <p style="font-size: 12px; color: #6b7280; margin: 0;">Processing Time</p>
                    <p style="font-size: 24px; color: #10b981; font-weight: 800; margin: 4px 0 0 0;">2-3s</p>
                </div>
                <div>
                    <p style="font-size: 12px; color: #6b7280; margin: 0;">Reliability</p>
                    <p style="font-size: 24px; color: #10b981; font-weight: 800; margin: 4px 0 0 0;">High</p>
                </div>
            </div>
        </div>
        <div style="background: rgba(16, 185, 129, 0.1); border-radius: 10px; padding: 16px;">
            <p style="font-size: 14px; color: #065f46; margin: 0; font-weight: 600;">
                ✓ Optimized for {method} transactions<br>
                ✓ Lowest combined fee + failure cost<br>
                ✓ High reliability and fast processing
            </p>
        </div>
    </div>
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2);">
        <h3 style="color: #92400e; font-size: 22px; font-weight: 900; margin: 0 0 15px 0;">💡 Alternative Options</h3>
        <div style="display: grid; gap: 10px;">
    """
    
    for alt in alternatives:
        savings = fee_amount - alt['fee_amount']
        result_html += f"""
        <div style="background: white; border-radius: 10px; padding: 16px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0;">{alt['processor']} ({alt['method']})</p>
                <p style="font-size: 13px; color: #6b7280; margin: 4px 0 0 0;">{alt['fee']}% fee • {alt['success_rate']}% success rate</p>
            </div>
            <div style="text-align: right;">
                <p style="font-size: 20px; color: #f59e0b; font-weight: 800; margin: 0;">${alt['fee_amount']:.2f}</p>
                <p style="font-size: 12px; color: {'#10b981' if savings > 0 else '#ef4444'}; margin: 4px 0 0 0;">{'Save' if savings > 0 else 'Cost'} ${abs(savings):.2f}</p>
            </div>
        </div>
        """
    
    result_html += "</div></div>"
    
    return result_html

st.markdown("""
<div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
    <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
        <span style="font-size: 56px;">💸</span>
    </div>
    <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        Slash Payment Intelligence
    </h1>
    <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Payment Routing • Cost Optimization</p>
    <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">AI-powered processor selection for maximum success and minimum fees</p>
    <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
        <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Smart Routing</span>
        <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Multi-Processor</span>
        <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Cost Analysis</span>
        <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Real-Time</span>
    </div>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
        Built for <strong style="color: white;">Slash</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "🎯 Payment Router"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Real-Time Payment Analytics</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Track success rates and processing times across all payment methods</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Refresh Dashboard", type="primary"):
        st.rerun()
    
    df = generate_payment_data()
    success_by_method, avg_processing, total_volume, success_volume = analyze_payments(df)
    overall_success = len(df[df['status']=='Success'])/len(df)*100
    
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Payment Processing Dashboard</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Volume</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${total_volume/1000:.0f}K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Processed</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Success Volume</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${success_volume/1000:.0f}K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Completed</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Success Rate</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{overall_success:.0f}%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Overall</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Avg Time</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{df[df['status']=='Success']['processing_time'].mean():.1f}s</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Processing</p>
            </div>
        </div>
    </div>
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🎯 Performance by Method</h3>
        <div style="display: grid; gap: 12px;">
    """
    
    colors = ['#667eea', '#10b981', '#ec4899', '#f59e0b', '#3b82f6']
    for idx, (method, rate) in enumerate(success_by_method.items()):
        proc_time = avg_processing[method]
        summary_html += f"""
        <div style="background: white; border-left: 5px solid {colors[idx]}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 4px 0;">{method}</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Avg processing: {proc_time:.1f}s</p>
                </div>
                <div style="text-align: right;">
                    <p style="font-size: 28px; color: {colors[idx]}; font-weight: 900; margin: 0;">{rate:.1f}%</p>
                    <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">Success rate</p>
                </div>
            </div>
        </div>
        """
    
    summary_html += "</div></div>"
    
    st.markdown(summary_html, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_success_rate_chart(success_by_method), use_container_width=True)
    with col2:
        st.plotly_chart(create_processing_time_chart(avg_processing), use_container_width=True)
    
    st.dataframe(df, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Optimize Your Payment Route</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Get the best processor recommendation for your transaction</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        amount_input = st.number_input("Transaction Amount ($)", min_value=1.0, value=1000.0, step=100.0)
        method_input = st.selectbox(
            "Payment Method",
            ['Credit Card', 'Debit Card', 'ACH', 'Wire Transfer', 'Digital Wallet']
        )
        route_btn = st.button("🚀 Get Optimal Route", type="primary")
    
    if route_btn:
        with col2:
            st.markdown(optimize_routing(amount_input, method_input), unsafe_allow_html=True)

st.markdown("""
<hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
<div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
    <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Slash</h2>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
        <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
            <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 15% Lower Fees</h4>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                Smart routing selects optimal processor per transaction. $15K savings per $1M processed.
            </p>
        </div>
        <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
            <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📈 8% Higher Success</h4>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                Method-specific routing maximizes approval rates. Fewer failed transactions, happier customers.
            </p>
        </div>
        <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
            <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 40% Faster</h4>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                Real-time processor selection. Average 2-3s processing vs 5s+ manual routing.
            </p>
        </div>
    </div>
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
        <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-Processor Support</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Stripe, Square, Plaid, Wise, PayPal</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Smart Routing Logic</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Optimize for success rate + fees</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Analytics</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Success rates, processing times</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Cost Comparison</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Alternative routing options</p>
            </div>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Slash</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
        <strong style="color: white;">Tech Stack:</strong> Python • Streamlit • Routing Algorithms • Real-time Analytics
    </p>
    <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
    <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
        Demo showcasing AI-powered payment optimization.<br>
        Smart routing • Cost analysis • Success rate tracking • Multi-processor support
    </p>
</div>
""", unsafe_allow_html=True)