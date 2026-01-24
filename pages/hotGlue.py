"""
HotGlue - SaaS Integration Platform
AI-powered API connector and data synchronization
Built for HotGlue by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from utils.sidebar import render_sidebar
render_sidebar()

st.set_page_config(page_title="HotGlue Integration Platform", layout="wide")

def generate_integration_data():
    """Generate sample integration data"""
    np.random.seed(42)
    
    integrations = [
        {'name': 'Salesforce', 'category': 'CRM', 'records': 15420, 'api': 'REST'},
        {'name': 'HubSpot', 'category': 'CRM', 'records': 8932, 'api': 'REST'},
        {'name': 'Stripe', 'category': 'Payment', 'records': 22145, 'api': 'REST'},
        {'name': 'QuickBooks', 'category': 'Accounting', 'records': 5678, 'api': 'REST'},
        {'name': 'Slack', 'category': 'Communication', 'records': 3421, 'api': 'Webhook'},
        {'name': 'Google Sheets', 'category': 'Productivity', 'records': 12890, 'api': 'REST'},
        {'name': 'Shopify', 'category': 'E-commerce', 'records': 18234, 'api': 'GraphQL'},
        {'name': 'Zendesk', 'category': 'Support', 'records': 7654, 'api': 'REST'},
        {'name': 'Jira', 'category': 'Project Mgmt', 'records': 4321, 'api': 'REST'},
        {'name': 'Mailchimp', 'category': 'Marketing', 'records': 9876, 'api': 'REST'}
    ]
    
    sync_data = []
    for integration in integrations:
        for i in range(5):
            sync_time = datetime.now() - timedelta(hours=np.random.randint(1, 48))
            records_synced = int(integration['records'] * np.random.uniform(0.05, 0.15))
            is_success = np.random.random() < 0.95
            duration = np.random.uniform(5, 45)
            
            sync_data.append({
                'integration': integration['name'],
                'category': integration['category'],
                'api_type': integration['api'],
                'timestamp': sync_time.strftime('%Y-%m-%d %H:%M'),
                'records_synced': records_synced,
                'duration': round(duration, 1),
                'status': 'Success' if is_success else 'Failed',
                'throughput': round(records_synced / (duration / 60), 0)
            })
    
    return pd.DataFrame(sync_data)

def analyze_integrations(df):
    """Analyze integration performance"""
    total_syncs = len(df)
    success_count = len(df[df['status'] == 'Success'])
    success_rate = (success_count / total_syncs) * 100
    
    total_records = df['records_synced'].sum()
    avg_duration = df['duration'].mean()
    avg_throughput = df['throughput'].mean()
    
    by_integration = df.groupby('integration').agg({
        'records_synced': 'sum',
        'duration': 'mean',
        'throughput': 'mean',
        'status': lambda x: (x == 'Success').sum() / len(x) * 100
    })
    by_integration.columns = ['total_records', 'avg_duration', 'avg_throughput', 'success_rate']
    
    by_category = df.groupby('category')['records_synced'].sum()
    
    return total_syncs, success_rate, total_records, avg_duration, avg_throughput, by_integration, by_category

def create_integration_chart(by_integration):
    """Create integration performance chart"""
    fig = go.Figure(data=[
        go.Bar(x=by_integration.index, y=by_integration['total_records'],
               marker=dict(color=['#667eea', '#764ba2', '#ec4899', '#10b981', '#f59e0b', 
                                 '#3b82f6', '#8b5cf6', '#ef4444', '#14b8a6', '#f97316']))
    ])
    fig.update_layout(
        title="Total Records Synced by Integration",
        xaxis_title="Integration",
        yaxis_title="Records",
        height=400,
        xaxis={'tickangle': -45}
    )
    return fig

def create_category_chart(by_category):
    """Create category distribution pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=by_category.index,
        values=by_category.values,
        hole=0.4,
        marker=dict(colors=['#667eea', '#764ba2', '#ec4899', '#10b981', '#f59e0b', 
                           '#3b82f6', '#8b5cf6', '#ef4444', '#14b8a6', '#f97316'])
    )])
    fig.update_layout(
        title="Records by Integration Category",
        height=400
    )
    return fig

def create_integration_flow(source_app, target_app, data_type):
    """Create integration flow visualization"""
    
    field_mappings = {
        'Customer Data': {
            'fields': ['name', 'email', 'phone', 'company', 'address'],
            'transformations': ['Email validation', 'Phone formatting', 'Address standardization']
        },
        'Invoices': {
            'fields': ['invoice_id', 'amount', 'date', 'customer_id', 'items'],
            'transformations': ['Currency conversion', 'Date formatting', 'Tax calculation']
        },
        'Support Tickets': {
            'fields': ['ticket_id', 'subject', 'status', 'priority', 'assignee'],
            'transformations': ['Status mapping', 'Priority normalization', 'Auto-assignment']
        },
        'Products': {
            'fields': ['sku', 'name', 'price', 'inventory', 'category'],
            'transformations': ['Price rounding', 'Inventory sync', 'Category mapping']
        },
        'Tasks': {
            'fields': ['task_id', 'title', 'status', 'due_date', 'assignee'],
            'transformations': ['Status conversion', 'Date formatting', 'User mapping']
        }
    }
    
    config = field_mappings.get(data_type, field_mappings['Customer Data'])
    estimated_records = np.random.randint(500, 5000)
    estimated_duration = round(estimated_records / 150, 1)
    
    flow_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🔄 Integration Flow</h2>
        <div style="display: grid; grid-template-columns: 1fr auto 1fr; gap: 20px; align-items: center;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 40px; margin: 0 0 10px 0;">📤</p>
                <p style="font-size: 24px; color: white; font-weight: 900; margin: 0 0 8px 0;">{source_app}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">Source</p>
            </div>
            <div style="color: white; font-size: 32px;">→</div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 40px; margin: 0 0 10px 0;">📥</p>
                <p style="font-size: 24px; color: white; font-weight: 900; margin: 0 0 8px 0;">{target_app}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">Target</p>
            </div>
        </div>
        <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 20px; margin-top: 20px; border: 2px solid rgba(255,255,255,0.2);">
            <p style="font-size: 16px; color: white; font-weight: 700; margin: 0;">Data Type: {data_type}</p>
        </div>
    </div>
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;">
        <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📋 Field Mappings</h3>
        <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
            <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0 0 15px 0;">Synced Fields ({len(config['fields'])})</p>
            <div style="display: grid; gap: 8px;">"""
    
    for field in config['fields']:
        flow_html += f"""
        <div style="background: #f0fdf4; border-left: 4px solid #10b981; border-radius: 8px; padding: 12px;">
            <p style="font-size: 14px; color: #065f46; font-weight: 600; margin: 0;">{field}</p>
        </div>"""
    
    flow_html += """
            </div>
        </div>
        <div style="background: white; border-radius: 12px; padding: 20px;">
            <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0 0 15px 0;">Transformations Applied</p>
            <div style="display: grid; gap: 8px;">"""
    
    for transform in config['transformations']:
        flow_html += f"""
        <div style="background: #f0f9ff; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 12px;">
            <p style="font-size: 14px; color: #1e40af; font-weight: 600; margin: 0;">✓ {transform}</p>
        </div>"""
    
    flow_html += f"""
            </div>
        </div>
    </div>
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2);">
        <h3 style="color: #92400e; font-size: 22px; font-weight: 900; margin: 0 0 15px 0;">⚡ Estimated Performance</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div style="background: white; border-radius: 10px; padding: 18px; text-align: center;">
                <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Records/Sync</p>
                <p style="font-size: 28px; color: #f59e0b; font-weight: 800; margin: 0;">{estimated_records:,}</p>
            </div>
            <div style="background: white; border-radius: 10px; padding: 18px; text-align: center;">
                <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Duration</p>
                <p style="font-size: 28px; color: #f59e0b; font-weight: 800; margin: 0;">{estimated_duration}min</p>
            </div>
            <div style="background: white; border-radius: 10px; padding: 18px; text-align: center;">
                <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Throughput</p>
                <p style="font-size: 28px; color: #f59e0b; font-weight: 800; margin: 0;">150/min</p>
            </div>
        </div>
        <div style="background: rgba(245, 158, 11, 0.1); border-radius: 10px; padding: 18px; margin-top: 15px;">
            <p style="font-size: 14px; color: #92400e; margin: 0; line-height: 1.6;">
                <strong>💡 Pro Tip:</strong> Enable real-time sync for critical data or schedule batch syncs every 15 minutes for optimal performance.
            </p>
        </div>
    </div>"""
    
    return flow_html

def process_dashboard():
    """Generate complete integration dashboard"""
    df = generate_integration_data()
    total_syncs, success_rate, total_records, avg_duration, avg_throughput, by_integration, by_category = analyze_integrations(df)
    
    integration_chart = create_integration_chart(by_integration)
    category_chart = create_category_chart(by_category)
    
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🔌 Integration Dashboard</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Syncs</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{total_syncs}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Last 48h</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Records Synced</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{total_records/1000:.1f}K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Total</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Success Rate</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{success_rate:.0f}%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Reliability</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Throughput</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{avg_throughput:.0f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Records/min</p>
            </div>
        </div>
    </div>
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🔗 Integration Performance</h3>
        <div style="display: grid; gap: 12px;">"""
    
    colors = ['#667eea', '#10b981', '#ec4899', '#f59e0b', '#3b82f6', '#8b5cf6', '#ef4444', '#14b8a6', '#f97316', '#764ba2']
    for idx, (integration, row) in enumerate(by_integration.iterrows()):
        summary_html += f"""
        <div style="background: white; border-left: 5px solid {colors[idx % len(colors)]}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 4px 0;">{integration}</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">{row['avg_duration']:.1f}s avg • {row['avg_throughput']:.0f} rec/min</p>
                </div>
                <div style="text-align: right;">
                    <p style="font-size: 28px; color: {colors[idx % len(colors)]}; font-weight: 900; margin: 0;">{row['total_records']:,.0f}</p>
                    <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">{row['success_rate']:.0f}% success</p>
                </div>
            </div>
        </div>"""
    
    summary_html += "</div></div>"
    
    return summary_html, integration_chart, category_chart, df

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🔌</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            HotGlue Integration Platform
        </h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">SaaS Connectors • Data Sync</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Connect any SaaS app with zero-code integrations</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">10+ Integrations</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Real-Time Sync</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Field Mapping</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Zero-Code</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">HotGlue</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📊 Integration Analytics", "🔌 Create Integration"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Data Sync Performance</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Track 50 syncs across 10 SaaS integrations</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Load Dashboard", type="primary", use_container_width=True):
        summary_html, integration_chart, category_chart, df = process_dashboard()
        
        st.markdown(summary_html, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(integration_chart, use_container_width=True)
        with col2:
            st.plotly_chart(category_chart, use_container_width=True)
        
        st.dataframe(df, use_container_width=True, height=400)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Build Your Integration Flow</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Connect apps, map fields, and sync data automatically</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_app = st.selectbox(
            "Source App",
            ['Salesforce', 'HubSpot', 'Stripe', 'QuickBooks', 'Slack', 
             'Google Sheets', 'Shopify', 'Zendesk', 'Jira', 'Mailchimp'],
            index=0
        )
    
    with col2:
        target_app = st.selectbox(
            "Target App",
            ['Salesforce', 'HubSpot', 'Stripe', 'QuickBooks', 'Slack', 
             'Google Sheets', 'Shopify', 'Zendesk', 'Jira', 'Mailchimp'],
            index=1
        )
    
    with col3:
        data_type = st.selectbox(
            "Data Type",
            ['Customer Data', 'Invoices', 'Support Tickets', 'Products', 'Tasks'],
            index=0
        )
    
    if st.button("⚡ Create Integration", type="primary", use_container_width=True):
        flow_html = create_integration_flow(source_app, target_app, data_type)
        st.markdown(flow_html, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for HotGlue</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🔌 10+ Pre-Built Connectors</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Salesforce, HubSpot, Stripe, QuickBooks, Shopify, Zendesk, and more. No custom API code needed.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 150 Records/Min</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    High-throughput data sync with automatic field mapping and transformations. 95% success rate.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🎯 Zero-Code Setup</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Visual flow builder, automatic field mapping, pre-built transformations. Live in minutes, not weeks.
                </p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-App Support</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">CRM, payment, e-commerce, support</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Sync</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Instant or scheduled batch updates</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Field Mapping</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Auto-detect and transform fields</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Performance Tracking</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Success rates, throughput, duration</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">HotGlue</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • REST APIs • Data Transformation • Plotly • Streamlit
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing SaaS integration platform.<br>
            10+ connectors • Real-time sync • Field mapping • Zero-code setup
        </p>
    </div>
    """, unsafe_allow_html=True)