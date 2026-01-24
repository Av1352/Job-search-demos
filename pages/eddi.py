"""
Eddi - Financial Wellness Platform
AI-powered spending insights and budget optimization
Built for Eddi by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from utils.sidebar import render_sidebar
render_sidebar()

# Page config
st.set_page_config(
    page_title="Eddi Demo - Anju Vilashni",
    page_icon="💰",
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

def generate_spending_data():
    """Generate sample spending data"""
    np.random.seed(42)
    categories = ['Food & Dining', 'Transportation', 'Shopping', 'Entertainment', 
                  'Bills & Utilities', 'Healthcare', 'Personal Care', 'Travel']
    
    data = []
    for i in range(100):
        category = np.random.choice(categories)
        spending_ranges = {
            'Food & Dining': (10, 80),
            'Transportation': (5, 50),
            'Shopping': (20, 200),
            'Entertainment': (15, 100),
            'Bills & Utilities': (50, 300),
            'Healthcare': (30, 500),
            'Personal Care': (10, 100),
            'Travel': (100, 1000)
        }
        
        amount = np.random.uniform(*spending_ranges[category])
        date = datetime.now() - timedelta(days=np.random.randint(0, 30))
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'category': category,
            'amount': round(amount, 2),
            'merchant': f"Merchant {i+1}"
        })
    
    return pd.DataFrame(data).sort_values('date', ascending=False)

# Header - NO empty lines!
st.markdown("""
<div style="text-align: center; padding: 20px 30px 70px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
    <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; margin: 0 auto 25px auto; border: 5px solid white; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5);">
        <span style="font-size: 56px;">💰</span>
    </div>
    <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        Eddi Financial Wellness
    </h1>
    <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Spending Insights • Budget Optimization</p>
    <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">AI-powered financial wellness and personalized recommendations</p>
    <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; max-width: 850px; margin: 28px auto 0 auto;">
        <span style="background:#ec4899;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Spending Analysis</span>
        <span style="background:#8b5cf6;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Budget Recommendations</span>
        <span style="background:#f59e0b;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">AI Insights</span>
        <span style="background:#10b981;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Savings Goals</span>
    </div>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 28px; font-weight: 600;">
        Built for <strong style="color:white;">Eddi</strong> by <strong style="color:white;">Anju Nandhakumar</strong>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Spending Overview", "💡 AI Insights", "🎯 Budget Recommendations"])

with tab1:
    st.markdown("""
<div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
    <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Your Financial Snapshot</h3>
    <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Analyze 100 transactions across 8 spending categories</p>
</div>
""", unsafe_allow_html=True)
    
    if st.button("🔍 Analyze My Spending", use_container_width=True, type="primary"):
        df = generate_spending_data()
        
        total_spending = df['amount'].sum()
        category_spending = df.groupby('category')['amount'].sum().sort_values(ascending=False)
        daily_avg = total_spending / 30
        potential_savings = total_spending * 0.15
        
        # Summary
        st.markdown(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
    <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">💰 Financial Wellness Dashboard</h2>
</div>
""", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
<div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
    <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Total Spending</p>
    <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">${total_spending:,.0f}</p>
    <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Last 30 days</p>
</div>
""", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
<div style="background: rgba(251, 191, 36, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(251, 191, 36, 0.3);">
    <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Daily Average</p>
    <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0;">${daily_avg:.0f}</p>
    <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Per day</p>
</div>
""", unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
<div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
    <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Categories</p>
    <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">{len(category_spending)}</p>
    <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Tracked</p>
</div>
""", unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
<div style="background: rgba(16, 185, 129, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(16, 185, 129, 0.3);">
    <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Save Potential</p>
    <p style="font-size: 48px; color: #10b981; font-weight: 900; margin: 0;">${potential_savings:.0f}</p>
    <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">15% reduction</p>
</div>
""", unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            colors = ['#667eea', '#764ba2', '#ec4899', '#10b981', '#f59e0b', '#3b82f6', '#8b5cf6', '#ef4444']
            fig_category = go.Figure(data=[go.Bar(
                x=category_spending.index,
                y=category_spending.values,
                marker=dict(color=colors[:len(category_spending)])
            )])
            fig_category.update_layout(title="Spending by Category (Last 30 Days)", xaxis={'tickangle': -45}, height=400)
            st.plotly_chart(fig_category, use_container_width=True)
        
        with col2:
            daily_spending = df.groupby('date')['amount'].sum().sort_index()
            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(
                x=daily_spending.index, y=daily_spending.values,
                mode='lines+markers', line=dict(color='#667eea', width=3),
                fill='tonexty', fillcolor='rgba(102, 126, 234, 0.2)'
            ))
            fig_daily.update_layout(title="Daily Spending Trend", height=400)
            st.plotly_chart(fig_daily, use_container_width=True)
        
        # Store in session state
        st.session_state.df = df
        st.session_state.category_spending = category_spending
        st.session_state.total_spending = total_spending

with tab2:
    if 'total_spending' in st.session_state:
        category_spending = st.session_state.category_spending
        total_spending = st.session_state.total_spending
        
        top_category = category_spending.index[0]
        top_amount = category_spending.values[0]
        top_percentage = (top_amount / total_spending * 100)
        
        st.markdown(f"""
<div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
    <h3 style="color: #92400e; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🎯 AI-Powered Insights</h3>
    <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
        <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0 0 10px 0;">💰 Top Spending Category</p>
        <p style="font-size: 14px; color: #6b7280; margin: 0; line-height: 1.6;">
            You spent <strong style="color: #f59e0b;">${top_amount:.0f}</strong> on <strong>{top_category}</strong> 
            ({top_percentage:.0f}% of total spending). Consider setting a monthly budget of 
            <strong>${total_spending * 0.15:.0f}</strong> for this category.
        </p>
    </div>
    <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
        <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0 0 10px 0;">📊 Spending Pattern</p>
        <p style="font-size: 14px; color: #6b7280; margin: 0; line-height: 1.6;">
            Your average daily spending is <strong style="color: #f59e0b;">${total_spending/30:.0f}</strong>. 
            This projects to <strong>${total_spending/30*365:.0f}/year</strong>. 
            Small adjustments can lead to significant annual savings.
        </p>
    </div>
    <div style="background: white; border-radius: 12px; padding: 20px;">
        <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0 0 10px 0;">💡 Savings Opportunity</p>
        <p style="font-size: 14px; color: #6b7280; margin: 0; line-height: 1.6;">
            Reducing spending by just <strong>10%</strong> would save you 
            <strong style="color: #10b981;">${total_spending * 0.1:.0f}/month</strong> 
            or <strong>${total_spending * 0.1 * 12:.0f}/year</strong>. 
            Focus on your top 3 spending categories for maximum impact.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Click 'Analyze My Spending' in the Spending Overview tab first!")

with tab3:
    if 'category_spending' in st.session_state:
        category_spending = st.session_state.category_spending
        total_spending = st.session_state.total_spending
        
        recommended_budgets = {
            'Food & Dining': total_spending * 0.15,
            'Transportation': total_spending * 0.10,
            'Shopping': total_spending * 0.10,
            'Entertainment': total_spending * 0.08,
            'Bills & Utilities': total_spending * 0.25,
            'Healthcare': total_spending * 0.12,
            'Personal Care': total_spending * 0.05,
            'Travel': total_spending * 0.15
        }
        
        st.markdown("""
<div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
    <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">💡 Personalized Budget Recommendations</h3>
</div>
""", unsafe_allow_html=True)
        
        colors = ['#667eea', '#10b981', '#f59e0b', '#ec4899', '#3b82f6', '#8b5cf6', '#ef4444', '#764ba2']
        
        for idx, (category, actual) in enumerate(category_spending.items()):
            recommended = recommended_budgets.get(category, 0)
            diff = actual - recommended
            over_budget = diff > 0
            percentage = (diff / recommended * 100) if recommended > 0 else 0
            
            st.markdown(f"""
<div style="background: white; border-left: 5px solid {colors[idx % len(colors)]}; border-radius: 12px; padding: 18px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
        <div>
            <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 6px 0;">{category}</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Recommended: ${recommended:.0f}/month</p>
        </div>
        <div style="text-align: right;">
            <p style="font-size: 24px; color: {'#ef4444' if over_budget else '#10b981'}; font-weight: 900; margin: 0;">${actual:.0f}</p>
            <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">Actual spent</p>
        </div>
    </div>
    <div style="background: {'#fee2e2' if over_budget else '#d1fae5'}; border-radius: 8px; padding: 12px;">
        <p style="font-size: 14px; color: {'#991b1b' if over_budget else '#065f46'}; font-weight: 700; margin: 0;">
            {'⚠️ ' if over_budget else '✅ '}{abs(percentage):.0f}% {'over' if over_budget else 'under'} budget (${abs(diff):.0f})
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Click 'Analyze My Spending' in the Spending Overview tab first!")

# Footer
st.markdown("<hr style='border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Eddi</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
        <strong>Tech Stack:</strong> Python • Streamlit • Pandas • Plotly • ML Insights
    </p>
</div>
""", unsafe_allow_html=True)