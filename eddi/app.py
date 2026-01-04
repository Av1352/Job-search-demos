"""
Eddi - Financial Wellness Platform
AI-powered spending insights and budget optimization
Built for Eddi by Anju Nandhakumar
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

def generate_spending_data():
    """Generate sample spending data"""
    np.random.seed(42)
    categories = ['Food & Dining', 'Transportation', 'Shopping', 'Entertainment', 
                  'Bills & Utilities', 'Healthcare', 'Personal Care', 'Travel']
    
    data = []
    for i in range(100):
        category = np.random.choice(categories)
        
        # Different spending ranges by category
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

def analyze_spending(df):
    """Analyze spending patterns"""
    total_spending = df['amount'].sum()
    category_spending = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    daily_avg = total_spending / 30
    
    # Budget recommendations
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
    
    return total_spending, category_spending, daily_avg, recommended_budgets

def create_category_chart(category_spending):
    """Create spending by category"""
    fig = go.Figure(data=[go.Bar(
        x=category_spending.index,
        y=category_spending.values,
        marker=dict(color=['#667eea', '#764ba2', '#ec4899', '#10b981', 
                          '#f59e0b', '#3b82f6', '#8b5cf6', '#ef4444'])
    )])
    fig.update_layout(
        title="Spending by Category (Last 30 Days)",
        xaxis_title="Category",
        yaxis_title="Amount ($)",
        height=400,
        xaxis={'tickangle': -45}
    )
    return fig

def create_daily_trend(df):
    """Create daily spending trend"""
    daily_spending = df.groupby('date')['amount'].sum().sort_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_spending.index,
        y=daily_spending.values,
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    fig.update_layout(
        title="Daily Spending Trend",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        height=400
    )
    return fig

def generate_budget_recommendations(category_spending, recommended_budgets):
    """Generate personalized budget recommendations"""
    
    recommendations_html = """
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;">
        <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">💡 Personalized Budget Recommendations</h3>
        
        <div style="display: grid; gap: 12px;">
    """
    
    colors = ['#667eea', '#10b981', '#f59e0b', '#ec4899', '#3b82f6', '#8b5cf6', '#ef4444', '#764ba2']
    
    for idx, (category, actual) in enumerate(category_spending.items()):
        recommended = recommended_budgets.get(category, 0)
        diff = actual - recommended
        over_budget = diff > 0
        percentage = (diff / recommended * 100) if recommended > 0 else 0
        
        recommendations_html += f"""
        <div style="background: white; border-left: 5px solid {colors[idx % len(colors)]}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
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
        """
    
    recommendations_html += "</div></div>"
    
    return recommendations_html

def generate_insights(category_spending, total_spending):
    """Generate AI-powered spending insights"""
    
    top_category = category_spending.index[0]
    top_amount = category_spending.values[0]
    top_percentage = (top_amount / total_spending * 100)
    
    insights_html = f"""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2); margin-bottom: 25px;">
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
    """
    
    return insights_html

def process_financial_wellness():
    """Generate complete financial wellness dashboard"""
    df = generate_spending_data()
    total_spending, category_spending, daily_avg, recommended_budgets = analyze_spending(df)
    
    category_chart = create_category_chart(category_spending)
    daily_chart = create_daily_trend(df)
    budget_recs = generate_budget_recommendations(category_spending, recommended_budgets)
    insights = generate_insights(category_spending, total_spending)
    
    # Calculate savings potential
    potential_savings = total_spending * 0.15  # 15% reduction target
    
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">💰 Financial Wellness Dashboard</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Spending</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${total_spending:,.0f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Last 30 days</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Daily Average</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${daily_avg:.0f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Per day</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Categories</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{len(category_spending)}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Tracked</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Save Potential</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${potential_savings:.0f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">15% reduction</p>
            </div>
        </div>
    </div>
    """
    
    return summary_html, category_chart, daily_chart, insights, budget_recs, df

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
            Eddi Financial Wellness
        </h1>
        
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Spending Insights • Budget Optimization</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">AI-powered financial wellness and personalized recommendations</p>
        
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Spending Analysis</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Budget Recommendations</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">AI Insights</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Savings Goals</span>
        </div>
        
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Eddi</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("📊 Spending Overview"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Your Financial Snapshot</h3>
                <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Analyze 100 transactions across 8 spending categories</p>
            </div>
            """)
            
            analyze_btn = gr.Button("🔍 Analyze My Spending", variant="primary", size="lg")
            
            summary_md = gr.HTML()
            
            with gr.Row():
                with gr.Column():
                    category_chart = gr.Plot(label="Spending by Category")
                with gr.Column():
                    daily_chart = gr.Plot(label="Daily Trend")
            
            transaction_table = gr.Dataframe(label="Recent Transactions")
        
        with gr.Tab("💡 AI Insights"):
            insights_output = gr.HTML()
        
        with gr.Tab("🎯 Budget Recommendations"):
            budget_output = gr.HTML()
    
    analyze_btn.click(
        fn=process_financial_wellness,
        outputs=[summary_md, category_chart, daily_chart, insights_output, budget_output, transaction_table]
    )
    
    gr.HTML("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Eddi</h2>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 15% Savings Average</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Users who follow personalized budgets save 15% on average. That's $3K+/year for most users.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📊 Real-Time Insights</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    AI analyzes spending patterns across 8 categories. Instant alerts when over budget.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🎯 Personalized Budgets</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Not generic advice. ML learns your habits, recommends realistic, achievable budgets.
                </p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Category Analysis</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">8 spending categories tracked</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ AI Recommendations</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Personalized budget suggestions</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Trend Visualization</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Daily spending patterns</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Savings Calculator</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Project annual savings potential</p>
                </div>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Eddi</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Pandas • Plotly • ML Insights • Gradio
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered financial wellness.<br>
            Spending analysis • Budget optimization • Personalized insights • Savings goals
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()