"""
Airweave - Context Retrieval Layer for AI Agents
Unified search across apps and databases for AI agents
Built for Airweave by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Airweave - AI Context Retrieval", page_icon="🔍", layout="wide")

# Integrations available
INTEGRATIONS = {
    'Google Drive': {'type': 'Docs', 'status': 'Active', 'documents': 12450},
    'Gmail': {'type': 'Email', 'status': 'Active', 'documents': 28930},
    'Slack': {'type': 'Chat', 'status': 'Active', 'documents': 45620},
    'Notion': {'type': 'Docs', 'status': 'Active', 'documents': 3847},
    'Jira': {'type': 'Project', 'status': 'Active', 'documents': 1892},
    'GitHub': {'type': 'Code', 'status': 'Active', 'documents': 8453},
    'HubSpot': {'type': 'CRM', 'status': 'Active', 'documents': 5623},
    'Dropbox': {'type': 'Storage', 'status': 'Active', 'documents': 6734}
}

# Agent capabilities
AGENT_CAPABILITIES = {
    'Semantic Search': 'Natural language understanding across all sources',
    'Hybrid Search': 'Combines keyword + semantic for accuracy',
    'Time-Aware Search': 'Prioritizes recent/relevant based on context',
    'Multi-Source Retrieval': 'Single query across 50+ integrations',
    'Intent Understanding': 'Interprets vague requests accurately',
    'OAuth2 Multi-Tenancy': 'Secure user-level access controls'
}

# Performance metrics
PERFORMANCE_METRICS = {
    'Retrieval Accuracy': {'airweave': 94.7, 'traditional': 68.3},
    'Query Speed': {'airweave': 0.8, 'traditional': 4.2},
    'Context Completeness': {'airweave': 92.1, 'traditional': 71.5},
    'Hallucination Rate': {'airweave': 2.3, 'traditional': 18.7}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🔍</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Airweave</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Context Retrieval for AI Agents</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">YC X25 • 50+ integrations • Unified search layer</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Agent Search", "📊 Integrations", "📈 Performance", "💡 Technology"])

with tab1:
    st.markdown("### AI Agent Context Retrieval")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Try Agent Search**")
        
        st.markdown("**Natural Language Query**")
        
        query_examples = [
            "Find that Asana ticket about auth configs",
            "What were the Q3 financials from Google Drive?",
            "Show me emails from Sarah about the product launch",
            "Get the latest design doc from Notion",
            "Find all Slack discussions about the API refactor"
        ]
        
        selected_query = st.selectbox("Example queries", query_examples)
        
        custom_query = st.text_area("Or enter your own query", 
                                    "Find the contract we discussed with Acme Corp last week",
                                    height=80)
        
        st.markdown("**Search Scope**")
        
        sources = st.multiselect("Data Sources",
                                list(INTEGRATIONS.keys()),
                                ["Google Drive", "Gmail", "Slack"])
        
        time_filter = st.selectbox("Time Range", ["Last 7 days", "Last 30 days", "Last 3 months", "All time"])
        
        search_btn = st.button("🔍 Search with Airweave", type="primary", use_container_width=True)
    
    with col2:
        if search_btn:
            st.markdown("**Airweave Search Results**")
            
            import time
            with st.spinner("Searching across connected apps..."):
                time.sleep(1.2)
            
            st.success("✅ Found 12 relevant results across 3 sources!")
            
            # Sample results
            results = [
                {
                    'source': 'Google Drive',
                    'title': 'Acme Corp - Master Services Agreement.pdf',
                    'snippet': 'This agreement dated January 15, 2026 between Company and Acme Corporation outlines...',
                    'relevance': 98,
                    'date': '6 days ago'
                },
                {
                    'source': 'Gmail',
                    'title': 'Re: Acme Corp Contract Terms',
                    'snippet': 'Sarah mentioned we should review the payment terms in section 4.2 before finalizing...',
                    'relevance': 95,
                    'date': '5 days ago'
                },
                {
                    'source': 'Slack',
                    'title': '#deals - Acme negotiation update',
                    'snippet': 'Update on Acme: Legal approved the contract, waiting on their CFO signature...',
                    'relevance': 92,
                    'date': '4 days ago'
                },
                {
                    'source': 'Google Drive',
                    'title': 'Acme - Proposal Presentation.pptx',
                    'snippet': 'Q4 proposal deck presented to Acme executive team on Jan 10th...',
                    'relevance': 88,
                    'date': '12 days ago'
                }
            ]
            
            for result in results:
                source_color = {
                    'Google Drive': '#4285f4',
                    'Gmail': '#ea4335',
                    'Slack': '#4a154b',
                    'Notion': '#000000'
                }.get(result['source'], '#8b5cf6')
                
                st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 12px; margin-bottom: 15px; border-left: 5px solid {source_color};">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                                <span style="background: {source_color}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;">{result['source']}</span>
                                <span style="background: #22c55e; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;">{result['relevance']}% match</span>
                            </div>
                            <h4 style="margin: 0 0 8px 0; color: #1f2937;">{result['title']}</h4>
                            <p style="margin: 0; color: #666; font-size: 14px; line-height: 1.5;">{result['snippet']}</p>
                        </div>
                        <div style="text-align: right; margin-left: 20px;">
                            <p style="margin: 0; color: #999; font-size: 13px;">{result['date']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Results Found", "12", "Across 3 apps")
            col2.metric("Search Time", "0.8 sec", "Real-time")
            col3.metric("Avg Relevance", "93%", "High")
            col4.metric("Sources Searched", "3", "Unified")

with tab2:
    st.markdown("### Connected Integrations")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Active Integrations", "8", "Connected")
    col2.metric("Total Documents", "113K+", "Indexed")
    col3.metric("Sync Frequency", "Real-time", "Auto")
    col4.metric("OAuth Users", "2,847", "Secure")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Connected Apps")
        
        integration_data = []
        for app, data in INTEGRATIONS.items():
            integration_data.append({
                'App': app,
                'Type': data['type'],
                'Documents': f"{data['documents']:,}",
                'Status': f"✅ {data['status']}"
            })
        
        st.dataframe(pd.DataFrame(integration_data), hide_index=True, use_container_width=True)
        
        st.markdown("### Document Distribution")
        
        apps = list(INTEGRATIONS.keys())
        doc_counts = [INTEGRATIONS[app]['documents'] for app in apps]
        
        fig1 = px.pie(
            values=doc_counts,
            names=apps,
            color_discrete_sequence=['#4285f4', '#ea4335', '#4a154b', '#000000', '#0052cc', '#181717', '#ff7a59', '#0061ff']
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        fig1.update_layout(height=300, showlegend=False)
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### Available Integrations (50+)**")
        
        all_integrations = {
            'Productivity': ['Google Drive', 'Gmail', 'Outlook', 'Google Calendar', 'Notion', 'Dropbox'],
            'Communication': ['Slack', 'Microsoft Teams', 'Discord'],
            'Project Management': ['Jira', 'Linear', 'Asana', 'Monday'],
            'Development': ['GitHub', 'GitLab', 'Bitbucket'],
            'CRM': ['HubSpot', 'Salesforce', 'Intercom'],
            'Payments': ['Stripe', 'PayPal'],
            'Databases': ['PostgreSQL', 'MySQL', 'MongoDB']
        }
        
        for category, apps in all_integrations.items():
            st.markdown(f"**{category}:**")
            st.markdown(", ".join(apps))
            st.markdown("")

with tab3:
    st.markdown("### Performance Analytics")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">94.7%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Retrieval Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">0.8s</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Avg Query Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #6d28d9 0%, #5b21b6 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">88%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Hallucination Reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Airweave vs Traditional RAG")
        
        performance_data = []
        for metric, values in PERFORMANCE_METRICS.items():
            unit = "%" if "Rate" in metric or "Completeness" in metric else "sec"
            performance_data.append({
                'Metric': metric,
                'Airweave': f"{values['airweave']}{unit}",
                'Traditional': f"{values['traditional']}{unit}",
                'Improvement': f"+{abs(values['airweave'] - values['traditional']):.1f}{unit}"
            })
        
        st.dataframe(pd.DataFrame(performance_data), hide_index=True, use_container_width=True)
        
        st.markdown("### Daily Query Volume")
        
        hours = list(range(9, 18))
        queries = [234, 456, 678, 892, 945, 876, 734, 598, 423]
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=hours,
            y=queries,
            mode='lines+markers',
            line=dict(color='#8b5cf6', width=3),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.1)'
        ))
        
        fig2.update_layout(
            xaxis_title='Hour of Day',
            yaxis_title='Queries',
            height=250
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("### Use Cases")
        
        use_cases = [
            {
                'use_case': 'Legal AI Assistant',
                'description': 'Search Google Drive/OneDrive for accurate contract answers'
            },
            {
                'use_case': 'Engineering Manager Agent',
                'description': 'Scan GitHub, Notion, Jira to understand codebases'
            },
            {
                'use_case': 'Compliance Agent',
                'description': 'Verify marketing vs current Dropbox financial data'
            },
            {
                'use_case': 'Customer Support',
                'description': 'Retrieve ticket history from Zendesk + Slack'
            },
            {
                'use_case': 'Sales Assistant',
                'description': 'Find CRM data from HubSpot + email context'
            }
        ]
        
        for uc in use_cases:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #8b5cf6;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">{uc['use_case']}</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{uc['description']}</p>
            </div>
            """, unsafe_allow_html=True)

with tab4:
    st.markdown("### Platform Technology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Core Capabilities**")
        
        for capability, description in AGENT_CAPABILITIES.items():
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #8b5cf6;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">{capability}</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**File Format Support**")
        st.markdown("""
        - 📄 Documents: DOCX, PDF, PPTX, XLSX, TXT
        - 🌐 Web: HTML, Markdown
        - 📸 Images: PNG, JPEG (OCR extraction)
        - 💻 Code: All major programming languages
        """)
    
    with col2:
        st.markdown("**Architecture**")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%); padding: 20px; border-radius: 12px;">
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">1. OAuth2 Authentication</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Secure user-level access to apps</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">2. Real-Time Syncing</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Continuous data updates from all sources</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">3. Unified Index</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Single searchable layer across all apps</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">4. Agent Query API</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Simple endpoint for AI agents to search</p>
            </div>
            <div>
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">5. Context Delivery</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Grounded, source-attributed results</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Integration Methods**")
        st.markdown("""
        - 🔌 SDKs (Python, TypeScript)
        - 🌐 REST API
        - 🤖 MCP (Model Context Protocol)
        - 🔧 Native agent framework integrations
        """)
        
        st.markdown("**Open Source**")
        
        st.markdown("""
        <div style="background: #dcfce7; padding: 15px; border-radius: 10px; margin-top: 15px;">
            <p style="margin: 0; color: #166534; font-size: 14px;">
            ⭐ <strong>Open-source on GitHub</strong><br>
            Active community contributions and transparent development
            </p>
        </div>
        """, unsafe_allow_html=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #5b21b6; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #8b5cf6; font-weight: 700; margin: 0 0 6px 0;">✓ 50+ Integrations</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Apps, databases, tools</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #8b5cf6; font-weight: 700; margin: 0 0 6px 0;">✓ 94.7% Accuracy</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Retrieval performance</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #8b5cf6; font-weight: 700; margin: 0 0 6px 0;">✓ YC X25</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Y Combinator backed</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #8b5cf6; font-weight: 700; margin: 0 0 6px 0;">✓ 88% Less Hallucination</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Grounded responses</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Airweave</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)