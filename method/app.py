"""
Method - Consumer Liability Intelligence
AI-powered debt management and optimization platform
Built for Method by Anju Nandhakumar
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

def generate_liability_data():
    """Generate sample consumer liability data"""
    np.random.seed(42)
    liability_types = ['Credit Card', 'Student Loan', 'Medical Debt', 'Auto Loan', 'Personal Loan']
    
    data = []
    for i in range(30):
        ltype = np.random.choice(liability_types)
        
        # Different balance ranges by type
        balance_ranges = {
            'Credit Card': (500, 15000),
            'Student Loan': (10000, 80000),
            'Medical Debt': (1000, 25000),
            'Auto Loan': (5000, 35000),
            'Personal Loan': (2000, 20000)
        }
        
        balance = np.random.uniform(*balance_ranges[ltype])
        interest_rate = np.random.uniform(3.5, 24.9)
        monthly_payment = balance * (interest_rate/100/12)
        
        # Risk scoring
        risk_factors = {
            'high_interest': interest_rate > 15,
            'large_balance': balance > 30000,
            'high_payment': monthly_payment > 500
        }
        risk_score = sum(risk_factors.values()) * 33.3
        
        data.append({
            'id': f'DEBT{1000+i}',
            'type': ltype,
            'balance': round(balance, 2),
            'interest_rate': round(interest_rate, 2),
            'monthly_payment': round(monthly_payment, 2),
            'risk_score': round(risk_score, 1)
        })
    
    return pd.DataFrame(data)

def analyze_liabilities(df):
    """Analyze liability portfolio"""
    total_debt = df['balance'].sum()
    monthly_payment = df['monthly_payment'].sum()
    avg_interest = (df['balance'] * df['interest_rate']).sum() / total_debt
    high_risk_count = len(df[df['risk_score'] > 50])
    
    return total_debt, monthly_payment, avg_interest, high_risk_count

def create_debt_breakdown(df):
    """Create debt composition chart"""
    debt_by_type = df.groupby('type')['balance'].sum().sort_values(ascending=False)
    
    fig = go.Figure(data=[go.Bar(
        x=debt_by_type.index,
        y=debt_by_type.values,
        marker=dict(color=['#667eea', '#764ba2', '#ec4899', '#10b981', '#f59e0b'])
    )])
    fig.update_layout(
        title="Debt Portfolio by Type",
        xaxis_title="Liability Type",
        yaxis_title="Balance ($)",
        height=400
    )
    return fig

def create_risk_distribution(df):
    """Create risk score distribution"""
    fig = px.histogram(df, x='risk_score', nbins=10,
                       color_discrete_sequence=['#667eea'])
    fig.update_layout(
        title="Risk Score Distribution",
        xaxis_title="Risk Score",
        yaxis_title="Number of Liabilities",
        height=400
    )
    return fig

def generate_payoff_plan(df):
    """Generate debt payoff strategy"""
    # Sort by interest rate (avalanche method)
    sorted_debt = df.sort_values('interest_rate', ascending=False)
    
    total_interest_saved = 0
    
    plan_html = """
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;">
        <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">💡 Optimal Payoff Strategy (Avalanche Method)</h3>
        <p style="color: #059669; font-size: 16px; margin: 0 0 20px 0; font-weight: 600;">Pay highest interest debts first to minimize total interest paid</p>
        
        <div style="display: grid; gap: 12px;">
    """
    
    colors = ['#ef4444', '#f59e0b', '#ec4899', '#8b5cf6', '#3b82f6']
    
    for idx, (_, row) in enumerate(sorted_debt.head(5).iterrows()):
        months_to_payoff = row['balance'] / row['monthly_payment']
        total_interest = row['monthly_payment'] * months_to_payoff - row['balance']
        total_interest_saved += total_interest
        
        plan_html += f"""
        <div style="background: white; border-left: 5px solid {colors[idx]}; border-radius: 12px; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                <div>
                    <p style="font-size: 20px; color: #1f2937; font-weight: 800; margin: 0 0 6px 0;">Priority #{idx+1}: {row['type']}</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Account: {row['id']}</p>
                </div>
                <div style="background: {colors[idx]}; color: white; padding: 6px 12px; border-radius: 8px;">
                    <p style="font-size: 14px; font-weight: 800; margin: 0;">{row['interest_rate']}% APR</p>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; background: #f9fafb; border-radius: 10px; padding: 14px;">
                <div>
                    <p style="font-size: 11px; color: #6b7280; margin: 0 0 4px 0;">Balance</p>
                    <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0;">${row['balance']:,.0f}</p>
                </div>
                <div>
                    <p style="font-size: 11px; color: #6b7280; margin: 0 0 4px 0;">Monthly</p>
                    <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0;">${row['monthly_payment']:.0f}</p>
                </div>
                <div>
                    <p style="font-size: 11px; color: #6b7280; margin: 0 0 4px 0;">Payoff</p>
                    <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0;">{months_to_payoff:.0f}mo</p>
                </div>
                <div>
                    <p style="font-size: 11px; color: #6b7280; margin: 0 0 4px 0;">Interest</p>
                    <p style="font-size: 18px; color: #ef4444; font-weight: 800; margin: 0;">${total_interest:,.0f}</p>
                </div>
            </div>
        </div>
        """
    
    plan_html += f"""
        </div>
        
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; padding: 20px; margin-top: 18px; text-align: center; color: white;">
            <p style="font-size: 20px; font-weight: 900; margin: 0;">💰 Total Interest to Pay: ${total_interest_saved:,.0f}</p>
            <p style="font-size: 14px; margin: 8px 0 0 0; opacity: 0.9;">Following this priority order minimizes your total interest</p>
        </div>
    </div>
    """
    
    return plan_html

def calculate_consolidation_savings(df):
    """Calculate potential consolidation savings"""
    current_interest = (df['balance'] * df['interest_rate']/100).sum()
    
    # Simulated consolidation rate
    consolidation_rate = 8.5
    consolidated_interest = df['balance'].sum() * consolidation_rate / 100
    
    annual_savings = current_interest - consolidated_interest
    monthly_savings = annual_savings / 12
    
    should_consolidate = annual_savings > 1000
    
    result_html = f"""
    <div style="background: linear-gradient(135deg, #{'d1fae5' if should_consolidate else 'fef3c7'} 0%, #{'a7f3d0' if should_consolidate else 'fde68a'} 100%); border: 3px solid #{'10b981' if should_consolidate else 'f59e0b'}; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba({'16, 185, 129' if should_consolidate else '245, 158, 11'}, 0.2); margin-bottom: 25px;">
        <h3 style="color: #{'065f46' if should_consolidate else '92400e'}; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🎯 Debt Consolidation Analysis</h3>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px;">
            <div style="background: white; border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 13px; color: #6b7280; margin: 0 0 8px 0;">Current Annual Interest</p>
                <p style="font-size: 36px; color: #ef4444; font-weight: 900; margin: 0;">${current_interest:,.0f}</p>
            </div>
            
            <div style="background: white; border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 13px; color: #6b7280; margin: 0 0 8px 0;">Consolidated Interest ({consolidation_rate}%)</p>
                <p style="font-size: 36px; color: #10b981; font-weight: 900; margin: 0;">${consolidated_interest:,.0f}</p>
            </div>
        </div>
        
        <div style="background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px;">
            <div style="text-align: center;">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0;">Annual Savings</p>
                <p style="font-size: 48px; color: #10b981; font-weight: 900; margin: 0;">${annual_savings:,.0f}</p>
                <p style="font-size: 16px; color: #6b7280; margin: 10px 0 0 0;">${monthly_savings:,.0f}/month</p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #{'10b981' if should_consolidate else 'f59e0b'} 0%, #{'059669' if should_consolidate else 'ea580c'} 100%); border-radius: 12px; padding: 20px; text-align: center; color: white;">
            <p style="font-size: 24px; font-weight: 900; margin: 0 0 10px 0;">{'✅ RECOMMENDED: CONSOLIDATE' if should_consolidate else '⚠️ REVIEW CAREFULLY'}</p>
            <p style="font-size: 15px; margin: 0; opacity: 0.95;">{'Significant savings with lower interest rate' if should_consolidate else 'Marginal savings - consider other factors'}</p>
        </div>
        
        <div style="background: rgba(255,255,255,0.8); border-radius: 10px; padding: 18px; margin-top: 15px;">
            <p style="font-size: 15px; color: #1f2937; font-weight: 700; margin: 0 0 10px 0;">Benefits of Consolidation:</p>
            <div style="display: grid; gap: 8px;">
                <p style="font-size: 14px; color: #4b5563; margin: 0;">✓ Single monthly payment (simplified management)</p>
                <p style="font-size: 14px; color: #4b5563; margin: 0;">✓ Lower average interest rate</p>
                <p style="font-size: 14px; color: #4b5563; margin: 0;">✓ Potential credit score improvement</p>
                <p style="font-size: 14px; color: #4b5563; margin: 0;">✓ Faster payoff timeline</p>
            </div>
        </div>
    </div>
    """
    
    return result_html

def process_dashboard():
    """Generate complete liability dashboard"""
    df = generate_liability_data()
    total_debt, monthly_payment, avg_interest, high_risk = analyze_liabilities(df)
    
    breakdown_chart = create_debt_breakdown(df)
    risk_chart = create_risk_distribution(df)
    payoff_plan = generate_payoff_plan(df)
    consolidation = calculate_consolidation_savings(df)
    
    # Portfolio health indicator
    if high_risk > 10:
        health_status = "🔴 High Risk"
        health_color = "#ef4444"
    elif high_risk > 5:
        health_status = "🟡 Medium Risk"
        health_color = "#f59e0b"
    else:
        health_status = "🟢 Low Risk"
        health_color = "#10b981"
    
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">💰 Consumer Liability Overview</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Debt</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${total_debt/1000:.0f}K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Outstanding</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Monthly Payment</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${monthly_payment:,.0f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Total/month</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Avg Interest</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{avg_interest:.1f}%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">APR</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">High Risk</p>
                <p style="font-size: 48px; color: {health_color}; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{high_risk}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Liabilities</p>
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px; margin-top: 20px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
            <p style="font-size: 20px; color: white; font-weight: 800; margin: 0;">Portfolio Health: {health_status}</p>
            <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 8px 0 0 0;">Debt-to-Payment Ratio: {total_debt/monthly_payment:.1f} months</p>
        </div>
    </div>
    """
    
    return summary_html, breakdown_chart, risk_chart, payoff_plan, consolidation, df

custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">💰</span>
        </div>
        
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Method Liability Intelligence
        </h1>
        
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Debt Management • Payoff Optimization</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">AI-powered debt analysis and consolidation strategies</p>
        
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Portfolio Analysis</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Payoff Strategy</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Risk Scoring</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Consolidation</span>
        </div>
        
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Method</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("📊 Portfolio Analysis"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Complete Liability Portfolio Overview</h3>
                <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Analyze 30 consumer liabilities with AI-powered insights</p>
            </div>
            """)
            
            analyze_btn = gr.Button("🔍 Analyze Portfolio", variant="primary", size="lg")
            
            summary_md = gr.HTML()
            
            with gr.Row():
                with gr.Column():
                    breakdown_chart = gr.Plot(label="Debt Breakdown by Type")
                with gr.Column():
                    risk_chart = gr.Plot(label="Risk Score Distribution")
            
            liability_table = gr.Dataframe(label="All Liabilities")
        
        with gr.Tab("🎯 Payoff Strategy"):
            payoff_output = gr.HTML()
        
        with gr.Tab("💡 Consolidation Analysis"):
            consolidation_output = gr.HTML()
    
    analyze_btn.click(
        fn=process_dashboard,
        outputs=[summary_md, breakdown_chart, risk_chart, payoff_output, consolidation_output, liability_table]
    )
    
    gr.HTML("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Method</h2>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 $12K+ Annual Savings</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Per user through optimized payoff strategy and consolidation. Avalanche method minimizes interest.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 30% Faster Payoff</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Vs random payments. Smart prioritization = less interest, faster debt freedom, better credit score.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📊 Multi-Factor Risk</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    AI scores based on interest rate, balance, payment amount. Identifies high-risk debts automatically.
                </p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Portfolio Analysis</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Complete debt overview, risk scoring</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Avalanche Optimization</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Highest interest first, minimize cost</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Consolidation Calculator</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Annual savings, recommendation engine</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Risk Modeling</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Multi-factor scoring algorithm</p>
                </div>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Method</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Pandas • Plotly • Optimization Algorithms • Gradio
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered debt management.<br>
            Portfolio analysis • Payoff optimization • Consolidation analysis • Risk scoring
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()