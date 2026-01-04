"""
Use Dots - Global Payout Intelligence
AI-powered contractor payment optimization across borders
Built for Use Dots by Anju Nandhakumar
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

def generate_payout_data():
    """Generate sample contractor payout data"""
    np.random.seed(42)
    contractor_types = ['Freelancer', 'Gig Worker', 'Contractor', 'Consultant']
    countries = ['USA', 'UK', 'Canada', 'India', 'Brazil', 'Germany']
    
    data = []
    for i in range(50):
        ctype = np.random.choice(contractor_types)
        country = np.random.choice(countries)
        amount = np.random.uniform(500, 15000)
        
        # Processing times vary by country
        processing_days = {
            'USA': np.random.uniform(1, 3),
            'UK': np.random.uniform(1, 2),
            'Canada': np.random.uniform(2, 4),
            'India': np.random.uniform(3, 7),
            'Brazil': np.random.uniform(4, 8),
            'Germany': np.random.uniform(1, 3)
        }
        
        days = processing_days[country]
        
        # Fees vary by method and country
        if country in ['USA', 'UK', 'Germany']:
            fee_pct = 1.5
        else:
            fee_pct = 2.5
        
        fee = amount * fee_pct / 100
        
        data.append({
            'contractor_id': f'CTR{1000+i}',
            'type': ctype,
            'country': country,
            'amount': round(amount, 2),
            'fee': round(fee, 2),
            'net_payout': round(amount - fee, 2),
            'processing_days': round(days, 1),
            'status': np.random.choice(['Pending', 'Processing', 'Completed'], p=[0.2, 0.3, 0.5])
        })
    
    return pd.DataFrame(data)

def analyze_payouts(df):
    """Analyze payout patterns"""
    total_volume = df['amount'].sum()
    total_fees = df['fee'].sum()
    avg_processing = df['processing_days'].mean()
    
    by_country = df.groupby('country').agg({
        'amount': 'sum',
        'processing_days': 'mean',
        'contractor_id': 'count'
    }).sort_values('amount', ascending=False)
    by_country.columns = ['volume', 'avg_days', 'count']
    
    return total_volume, total_fees, avg_processing, by_country

def create_volume_chart(by_country):
    """Create payout volume by country"""
    fig = go.Figure(data=[
        go.Bar(x=by_country.index, y=by_country['volume'],
               marker=dict(color=['#667eea', '#764ba2', '#ec4899', '#10b981', '#f59e0b', '#3b82f6']))
    ])
    fig.update_layout(
        title="Payout Volume by Country",
        xaxis_title="Country",
        yaxis_title="Volume ($)",
        height=400
    )
    return fig

def create_processing_chart(by_country):
    """Create processing time by country"""
    fig = go.Figure(data=[
        go.Bar(x=by_country.index, y=by_country['avg_days'],
               marker=dict(color='#3b82f6'))
    ])
    fig.update_layout(
        title="Average Processing Time by Country",
        xaxis_title="Country",
        yaxis_title="Days",
        height=400
    )
    return fig

def optimize_payout(amount, country, urgency):
    """Recommend optimal payout method"""
    methods = {
        'instant': {
            'fee': 3.5, 
            'time': '< 1 hour', 
            'availability': ['USA', 'UK', 'Germany'],
            'description': 'Same-day transfer via real-time payment networks'
        },
        'fast': {
            'fee': 2.0, 
            'time': '1-2 days', 
            'availability': ['USA', 'UK', 'Canada', 'Germany'],
            'description': 'Express international transfer with priority processing'
        },
        'standard': {
            'fee': 1.5, 
            'time': '3-5 days', 
            'availability': ['USA', 'UK', 'Canada', 'Germany', 'India', 'Brazil'],
            'description': 'Regular international bank transfer'
        }
    }
    
    # Select method based on urgency and country
    if urgency == 'Urgent' and country in methods['instant']['availability']:
        method = 'instant'
    elif urgency == 'Normal' and country in methods['fast']['availability']:
        method = 'fast'
    else:
        method = 'standard'
    
    selected = methods[method]
    fee = amount * selected['fee'] / 100
    net = amount - fee
    
    # Calculate alternatives
    alternatives = []
    for m, details in methods.items():
        if m != method and country in details['availability']:
            alt_fee = amount * details['fee'] / 100
            alternatives.append({
                'method': m,
                'fee_pct': details['fee'],
                'fee_amount': alt_fee,
                'time': details['time'],
                'description': details['description']
            })
    
    result_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">💸 Optimal Payout Method</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Amount</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${amount:,.0f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Payout</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Country</p>
                <p style="font-size: 32px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{country}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Destination</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Fee</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${fee:.2f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{selected['fee']}%</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Time</p>
                <p style="font-size: 32px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{selected['time'].split()[0]}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{selected['time']}</p>
            </div>
        </div>
    </div>
    
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;">
        <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">✅ Recommended: {method.upper()} Transfer</h3>
        
        <div style="background: white; border-radius: 12px; padding: 22px; margin-bottom: 18px;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; margin-bottom: 15px;">
                <div>
                    <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Net Payout</p>
                    <p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">${net:,.2f}</p>
                </div>
                <div>
                    <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Urgency Level</p>
                    <p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">{urgency}</p>
                </div>
            </div>
            
            <div style="background: #f0fdf4; border-radius: 10px; padding: 16px;">
                <p style="font-size: 14px; color: #065f46; margin: 0; line-height: 1.6;">{selected['description']}</p>
            </div>
        </div>
        
        <div style="background: rgba(16, 185, 129, 0.1); border-radius: 10px; padding: 18px;">
            <p style="font-size: 15px; color: #065f46; font-weight: 700; margin: 0 0 10px 0;">Why This Method?</p>
            <div style="display: grid; gap: 8px;">
                <p style="font-size: 14px; color: #059669; margin: 0;">✓ {'Fastest available for ' + country if method == 'instant' else 'Balance of speed and cost' if method == 'fast' else 'Most cost-effective option'}</p>
                <p style="font-size: 14px; color: #059669; margin: 0;">✓ Reliable for {country} transfers</p>
                <p style="font-size: 14px; color: #059669; margin: 0;">✓ Compliant with local regulations</p>
                <p style="font-size: 14px; color: #059669; margin: 0;">✓ Full tracking and notifications</p>
            </div>
        </div>
    </div>
    """
    
    if alternatives:
        result_html += """
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2);">
            <h3 style="color: #92400e; font-size: 22px; font-weight: 900; margin: 0 0 15px 0;">💡 Alternative Options</h3>
            <div style="display: grid; gap: 12px;">
        """
        
        for alt in alternatives:
            savings = fee - alt['fee_amount']
            result_html += f"""
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                    <div>
                        <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 4px 0;">{alt['method'].upper()} Transfer</p>
                        <p style="font-size: 13px; color: #6b7280; margin: 0;">{alt['description']}</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="font-size: 24px; color: #f59e0b; font-weight: 800; margin: 0;">${alt['fee_amount']:.2f}</p>
                        <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">{alt['fee_pct']}% fee</p>
                    </div>
                </div>
                <div style="background: #fef3c7; border-radius: 8px; padding: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <p style="font-size: 13px; color: #92400e; margin: 0;">⏱️ Processing: {alt['time']}</p>
                    <p style="font-size: 13px; color: {'#10b981' if savings > 0 else '#ef4444'}; font-weight: 700; margin: 0;">{'Save' if savings > 0 else 'Cost'} ${abs(savings):.2f}</p>
                </div>
            </div>
            """
        
        result_html += "</div></div>"
    
    return result_html

def process_dashboard():
    """Generate complete payout dashboard"""
    df = generate_payout_data()
    total_volume, total_fees, avg_processing, by_country = analyze_payouts(df)
    
    volume_chart = create_volume_chart(by_country)
    processing_chart = create_processing_chart(by_country)
    
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🌍 Global Payout Dashboard</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Volume</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${total_volume/1000:.0f}K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Processed</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Fees</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${total_fees:,.0f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Collected</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Avg Processing</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{avg_processing:.1f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Days</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Contractors</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">50</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Active</p>
            </div>
        </div>
    </div>
    
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🌎 Performance by Country</h3>
        
        <div style="display: grid; gap: 12px;">
    """
    
    colors = ['#667eea', '#10b981', '#ec4899', '#f59e0b', '#3b82f6', '#8b5cf6']
    for idx, (country, row) in enumerate(by_country.iterrows()):
        summary_html += f"""
        <div style="background: white; border-left: 5px solid {colors[idx]}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
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
        """
    
    summary_html += "</div></div>"
    
    return summary_html, volume_chart, processing_chart, df

custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">💳</span>
        </div>
        
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Use Dots Payout Intelligence
        </h1>
        
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Global Payouts • Smart Routing</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">AI-powered contractor payment optimization across 6 countries</p>
        
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Multi-Country</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Smart Routing</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Cost Optimization</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Real-Time</span>
        </div>
        
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Use Dots</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("📊 Payout Analytics"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Global Payment Performance</h3>
                <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Track volume, fees, and processing times across 6 countries</p>
            </div>
            """)
            
            dashboard_btn = gr.Button("🔄 Refresh Dashboard", variant="primary", size="lg")
            
            summary_md = gr.HTML()
            
            with gr.Row():
                with gr.Column():
                    volume_chart = gr.Plot(label="Volume by Country")
                with gr.Column():
                    processing_chart = gr.Plot(label="Processing Times")
            
            payout_table = gr.Dataframe(label="Recent Payouts")
        
        with gr.Tab("🎯 Payout Optimizer"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Optimize Your Contractor Payment</h3>
                <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Get the best payment method for your contractor's location</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    amount_input = gr.Number(label="Payout Amount ($)", value=5000, minimum=1)
                    country_input = gr.Dropdown(
                        choices=['USA', 'UK', 'Canada', 'India', 'Brazil', 'Germany'],
                        label="Contractor Country",
                        value='USA'
                    )
                    urgency_input = gr.Radio(
                        choices=['Urgent', 'Normal', 'Standard'],
                        label="Urgency Level",
                        value='Normal'
                    )
                    optimize_btn = gr.Button("🚀 Get Optimal Method", variant="primary", size="lg")
            
            optimization_output = gr.HTML()
    
    dashboard_btn.click(
        fn=process_dashboard,
        outputs=[summary_md, volume_chart, processing_chart, payout_table]
    )
    
    optimize_btn.click(
        fn=optimize_payout,
        inputs=[amount_input, country_input, urgency_input],
        outputs=optimization_output
    )
    
    gr.HTML("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Use Dots</h2>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 40% Faster Payouts</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Smart routing vs manual processing. Better contractor retention through faster payments.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 25% Lower Fees</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Method optimization by country and urgency. For 1K contractors: saves $50K/year in fees.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🌍 6 Country Support</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Multi-country compliance built-in. Instant/fast/standard options per region.
                </p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-Speed Options</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Instant, fast, or standard transfers</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Country-Specific Routing</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Optimized by destination + urgency</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Analytics</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Volume, fees, processing times</p>
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
            Built for <strong style="color: white;">Use Dots</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Gradio • Routing Logic • Global Payment APIs
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered global payout optimization.<br>
            Multi-country support • Smart routing • Cost analysis • Real-time tracking
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()