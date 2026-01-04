"""
Alinea Invest - AI Investment Intelligence
AI-powered portfolio analysis and investment recommendations
Built for Alinea Invest by Anju Nandhakumar
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

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
    fig.update_layout(
        title="Portfolio Allocation by Sector",
        height=400
    )
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

def generate_ai_recommendations(df, sector_allocation, total_value):
    """Generate AI-powered investment recommendations"""
    
    # Find overweight sectors
    tech_allocation = sector_allocation.get('Technology', 0) / total_value * 100
    
    # Top performers
    top_performers = df.nlargest(3, 'gain_loss_pct')
    
    # Underperformers
    underperformers = df.nsmallest(2, 'gain_loss_pct')
    
    recommendations_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
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
    
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;">
        <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🚀 Top Performers</h3>
        
        <div style="display: grid; gap: 12px;">
    """
    
    for _, stock in top_performers.iterrows():
        recommendations_html += f"""
        <div style="background: white; border-left: 5px solid #10b981; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
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
        """
    
    recommendations_html += "</div></div>"
    
    recommendations_html += """
    <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(239, 68, 68, 0.2); margin-bottom: 25px;">
        <h3 style="color: #991b1b; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">⚠️ Review Needed</h3>
        
        <div style="display: grid; gap: 12px;">
    """
    
    for _, stock in underperformers.iterrows():
        recommendations_html += f"""
        <div style="background: white; border-left: 5px solid #ef4444; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
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
        """
    
    recommendations_html += """
        </div>
        
        <div style="background: rgba(239, 68, 68, 0.1); border-radius: 10px; padding: 18px; margin-top: 15px;">
            <p style="font-size: 14px; color: #991b1b; margin: 0; line-height: 1.6;">
                <strong>AI Insight:</strong> Consider setting stop-loss orders or reallocating capital from underperformers to higher-growth opportunities.
            </p>
        </div>
    </div>
    """
    
    return recommendations_html

def generate_diversification_suggestions(sector_allocation, total_value):
    """Generate sector diversification suggestions"""
    
    ideal_allocation = {
        'Technology': 30,
        'Financial': 20,
        'Healthcare': 20,
        'Consumer': 15,
        'Automotive': 15
    }
    
    suggestions_html = """
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2);">
        <h3 style="color: #92400e; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🎯 Diversification Strategy</h3>
        
        <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
            <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0 0 15px 0;">Ideal vs Current Allocation</p>
            
            <div style="display: grid; gap: 10px;">
    """
    
    for sector, ideal_pct in ideal_allocation.items():
        current_value = sector_allocation.get(sector, 0)
        current_pct = (current_value / total_value * 100)
        diff = current_pct - ideal_pct
        
        color = '#ef4444' if abs(diff) > 10 else '#10b981' if abs(diff) < 5 else '#f59e0b'
        
        suggestions_html += f"""
        <div style="background: #f9fafb; border-radius: 8px; padding: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <p style="font-size: 14px; color: #1f2937; font-weight: 700; margin: 0;">{sector}</p>
                <p style="font-size: 14px; color: {color}; font-weight: 700; margin: 0;">{current_pct:.1f}% (Target: {ideal_pct}%)</p>
            </div>
            <div style="background: #e5e7eb; border-radius: 4px; height: 8px; overflow: hidden;">
                <div style="background: {color}; height: 100%; width: {min(current_pct, 100)}%;"></div>
            </div>
        </div>
        """
    
    suggestions_html += """
            </div>
        </div>
        
        <div style="background: rgba(245, 158, 11, 0.1); border-radius: 10px; padding: 18px;">
            <p style="font-size: 15px; color: #92400e; font-weight: 700; margin: 0 0 10px 0;">💡 Action Items:</p>
            <div style="display: grid; gap: 8px;">
                <p style="font-size: 14px; color: #78350f; margin: 0;">✓ Reduce Technology exposure by 10-15%</p>
                <p style="font-size: 14px; color: #78350f; margin: 0;">✓ Increase Healthcare allocation to 20%</p>
                <p style="font-size: 14px; color: #78350f; margin: 0;">✓ Add Financial sector positions (target 20%)</p>
                <p style="font-size: 14px; color: #78350f; margin: 0;">✓ Maintain diversification across 5+ sectors</p>
            </div>
        </div>
    </div>
    """
    
    return suggestions_html

def process_portfolio():
    """Generate complete portfolio analysis"""
    df = generate_portfolio_data()
    total_value, total_cost, total_gain_loss, total_return_pct, sector_allocation = analyze_portfolio(df)
    
    allocation_chart = create_allocation_chart(sector_allocation)
    performance_chart = create_performance_chart(df)
    recommendations = generate_ai_recommendations(df, sector_allocation, total_value)
    diversification = generate_diversification_suggestions(sector_allocation, total_value)
    
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Portfolio Summary</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Portfolio Value</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${total_value/1000:.1f}K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Current</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Invested</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${total_cost/1000:.1f}K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Cost basis</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Gain/Loss</p>
                <p style="font-size: 48px; color: {'#86efac' if total_gain_loss > 0 else '#fca5a5'}; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{'+'if total_gain_loss > 0 else ''}{total_gain_loss/1000:.1f}K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Unrealized</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Return</p>
                <p style="font-size: 48px; color: {'#86efac' if total_return_pct > 0 else '#fca5a5'}; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{'+'if total_return_pct > 0 else ''}{total_return_pct:.1f}%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Performance</p>
            </div>
        </div>
    </div>
    """
    
    return summary_html, allocation_chart, performance_chart, recommendations, diversification, df

custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">📈</span>
        </div>
        
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Alinea Investment Intelligence
        </h1>
        
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Portfolio Analysis • AI Recommendations</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Smart investing for Gen Z with AI-powered insights</p>
        
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Portfolio Tracking</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">AI Recommendations</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Diversification</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Performance</span>
        </div>
        
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Alinea Invest</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("📊 Portfolio Overview"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Your Investment Portfolio</h3>
                <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Analyze 10 positions across 5 sectors</p>
            </div>
            """)
            
            analyze_btn = gr.Button("📈 Analyze Portfolio", variant="primary", size="lg")
            
            summary_md = gr.HTML()
            
            with gr.Row():
                with gr.Column():
                    allocation_chart = gr.Plot(label="Sector Allocation")
                with gr.Column():
                    performance_chart = gr.Plot(label="Stock Performance")
            
            portfolio_table = gr.Dataframe(label="Holdings")
        
        with gr.Tab("🤖 AI Recommendations"):
            recommendations_output = gr.HTML()
        
        with gr.Tab("🎯 Diversification"):
            diversification_output = gr.HTML()
    
    analyze_btn.click(
        fn=process_portfolio,
        outputs=[summary_md, allocation_chart, performance_chart, recommendations_output, diversification_output, portfolio_table]
    )
    
    gr.HTML("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Alinea Invest</h2>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🎯 Personalized Insights</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    AI analyzes your portfolio, identifies concentration risks, recommends rebalancing strategies.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📊 Real-Time Tracking</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Track 10+ positions, sector allocation, performance by stock. See gains/losses instantly.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🚀 Gen Z Focus</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Simple, visual, mobile-first. Learn investing while building wealth. No jargon.
                </p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Portfolio Analysis</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Value, cost basis, returns tracking</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ AI Recommendations</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Personalized rebalancing advice</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Sector Diversification</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Risk analysis by allocation</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Performance Visualization</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Charts, trends, comparisons</p>
                </div>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Alinea Invest</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Pandas • Plotly • Portfolio Analytics • Gradio
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered investment intelligence.<br>
            Portfolio tracking • Performance analysis • AI recommendations • Diversification strategies
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()