"""
Candid Intelligence - AI Agents for Energy Infrastructure
Multi-agent systems for construction pre-construction workflows
Built for Candid Intelligence by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Candid Intelligence", page_icon="⚡", layout="wide")

# AI agent tasks
AI_AGENT_TASKS = {
    'Document Reconciliation Agent': {
        'tasks': ['Analyze engineering docs', 'Find inconsistencies', 'Flag conflicts'],
        'volume': 12450,
        'time_saved': '420 hrs/month'
    },
    'Information Chase Agent': {
        'tasks': ['Track missing data', 'Auto-follow-up emails', 'Coordinate responses'],
        'volume': 8920,
        'time_saved': '340 hrs/month'
    },
    'ERP Navigation Agent': {
        'tasks': ['Operate legacy software', 'Schedule coordination', 'Data entry automation'],
        'volume': 15680,
        'time_saved': '580 hrs/month'
    },
    'Contractor Coordination Agent': {
        'tasks': ['Multi-party communication', 'Timeline tracking', 'Dependency management'],
        'volume': 6730,
        'time_saved': '280 hrs/month'
    }
}

# Pre-construction phases
PRECONSTRUCTION_PHASES = {
    'Bidding': {'traditional': 6, 'with_ai': 2, 'bottleneck': 'Manual document review'},
    'Engineering': {'traditional': 8, 'with_ai': 3, 'bottleneck': 'Document inconsistencies'},
    'Procurement': {'traditional': 6, 'with_ai': 2, 'bottleneck': 'Vendor coordination'},
    'Approvals': {'traditional': 4, 'with_ai': 1, 'bottleneck': 'Missing information'}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">⚡</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Candid Intelligence</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">AI Agents for Energy Infrastructure</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Multi-agent systems • Pre-construction automation • $5.5M seed</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["⚡ AI Agents", "📊 Pre-Construction", "📈 Performance", "💡 Technology"])

with tab1:
    st.markdown("### Multi-Agent Pre-Construction Automation")
    
    st.markdown("""
    <div style="background: #fef3c7; padding: 20px; border-radius: 12px; border-left: 5px solid #f59e0b;">
        <h4 style="margin: 0 0 12px 0; color: #92400e;">⚠️ The Problem: 2-Year Pre-Construction Timeline</h4>
        <p style="margin: 0; color: #78350f; font-size: 15px; line-height: 1.6;">
        Energy infrastructure projects take 4 years total. The first 2 years are consumed entirely by pre-construction:
        <br><br>
        • <strong>Bidding (6 months):</strong> Manual review of thousands of documents<br>
        • <strong>Engineering (8 months):</strong> Inconsistencies across drawings, specs, emails<br>
        • <strong>Procurement (6 months):</strong> Vendor coordination, missing information<br>
        • <strong>Approvals (4 months):</strong> Chasing data, regulatory delays
        <br><br>
        <strong>Result:</strong> 2 years before first construction starts. Billions in delays.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #dcfce7; padding: 20px; border-radius: 12px; border-left: 5px solid #22c55e; margin-top: 15px;">
        <h4 style="margin: 0 0 12px 0; color: #166534;">✓ Solution: Multi-Agent AI System</h4>
        <p style="margin: 0 0 10px 0; color: #15803d; font-size: 15px;">
        Candid deploys specialized AI agents that work 24/7 across the pre-construction workflow:
        </p>
        <ul style="margin: 0; padding-left: 20px; color: #15803d; font-size: 14px;">
            <li><strong>Document Agent:</strong> Reconciles inconsistencies across thousands of engineering docs</li>
            <li><strong>Information Agent:</strong> Chases missing data automatically (no more email tag)</li>
            <li><strong>ERP Agent:</strong> Operates legacy software through the UI (no API needed)</li>
            <li><strong>Coordination Agent:</strong> Manages dozens of contractors, tracks dependencies</li>
        </ul>
        <p style="margin: 10px 0 0 0; color: #15803d; font-size: 15px;">
        <strong>Goal:</strong> Collapse pre-construction from 24 months → 8 months
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### AI Agent Activity Dashboard")
    
    agent_activity = []
    for agent, data in AI_AGENT_TASKS.items():
        agent_activity.append({
            'Agent': agent,
            'Primary Tasks': ', '.join(data['tasks'][:2]),
            'Monthly Volume': f"{data['volume']:,}",
            'Time Saved': data['time_saved']
        })
    
    st.dataframe(pd.DataFrame(agent_activity), hide_index=True, use_container_width=True)

with tab2:
    st.markdown("### Pre-Construction Timeline Transformation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Traditional Process (24 months)")
        
        traditional_data = []
        for phase, data in PRECONSTRUCTION_PHASES.items():
            traditional_data.append({
                'Phase': phase,
                'Duration': f"{data['traditional']} months",
                'Bottleneck': data['bottleneck']
            })
        
        st.dataframe(pd.DataFrame(traditional_data), hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: #fee2e2; padding: 20px; border-radius: 12px; margin-top: 15px;">
            <h4 style="margin: 0 0 10px 0; color: #991b1b;">⏰ Total Timeline</h4>
            <p style="font-size: 32px; font-weight: 900; color: #991b1b; margin: 0;">24 months</p>
            <p style="margin: 8px 0 0 0; color: #7f1d1d; font-size: 14px;">Before first construction starts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### With Candid AI (8 months)")
        
        candid_data = []
        for phase, data in PRECONSTRUCTION_PHASES.items():
            candid_data.append({
                'Phase': phase,
                'Duration': f"{data['with_ai']} months",
                'AI Automation': f"Automates {data['bottleneck'].lower()}"
            })
        
        st.dataframe(pd.DataFrame(candid_data), hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: #dcfce7; padding: 20px; border-radius: 12px; margin-top: 15px;">
            <h4 style="margin: 0 0 10px 0; color: #166534;">⚡ Accelerated Timeline</h4>
            <p style="font-size: 32px; font-weight: 900; color: #166534; margin: 0;">8 months</p>
            <p style="margin: 8px 0 0 0; color: #15803d; font-size: 14px;">67% timeline reduction</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Platform Performance")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">67%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Timeline Reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d97706 0%, #b45309 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">1,620</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Hours Saved Monthly</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #b45309 0%, #92400e 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">$5.5M</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Seed Funding</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Time Savings by Phase")
        
        phases = list(PRECONSTRUCTION_PHASES.keys())
        traditional_times = [PRECONSTRUCTION_PHASES[p]['traditional'] for p in phases]
        ai_times = [PRECONSTRUCTION_PHASES[p]['with_ai'] for p in phases]
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Bar(
            x=phases,
            y=traditional_times,
            name='Traditional',
            marker_color='#94a3b8'
        ))
        
        fig1.add_trace(go.Bar(
            x=phases,
            y=ai_times,
            name='With Candid AI',
            marker_color='#f59e0b'
        ))
        
        fig1.update_layout(
            barmode='group',
            yaxis_title='Duration (months)',
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### Cost Impact")
        
        cost_data = {
            'Category': ['Labor savings', 'Timeline compression', 'Error prevention', 'Efficiency gains'],
            'Annual Value': ['$8.5M', '$12.3M', '$4.2M', '$6.8M']
        }
        
        st.dataframe(pd.DataFrame(cost_data), hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; margin-top: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #92400e;">💰 Total Annual Impact</h4>
            <p style="font-size: 32px; font-weight: 900; color: #92400e; margin: 0;">$31.8M</p>
            <p style="margin: 8px 0 0 0; color: #78350f; font-size: 14px;">Per large energy infrastructure project</p>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("### Platform Technology & Traction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Funding & Investors**")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%); padding: 20px; border-radius: 12px;">
            <h4 style="margin: 0 0 15px 0; color: #9a3412;">💰 $5.5M Seed Round</h4>
            <p style="margin: 8px 0; color: #92400e; font-size: 14px;">
            <strong>Raised in under 1 week</strong>
            </p>
            <p style="margin: 8px 0; color: #92400e; font-size: 14px;">
            <strong>Investors:</strong><br>
            • Backed by investors of OpenAI and xAI<br>
            • Quiet Capital (lead investor)
            </p>
            <p style="margin: 8px 0; color: #92400e; font-size: 14px;">
            <strong>World-Class Angels:</strong><br>
            • Yann LeCun (Meta AI Chief Scientist)<br>
            • Carnegie Mellon alumni network<br>
            • MIT E14 Fund
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Market Focus**")
        st.markdown("""
        - ⚡ Energy infrastructure projects
        - 🏗️ Power generation facilities
        - 🔌 Grid infrastructure
        - 💾 Data center construction
        - 🌐 AI compute infrastructure
        """)
    
    with col2:
        st.markdown("**Customer Traction**")
        
        st.markdown("""
        <div style="background: #dcfce7; padding: 20px; border-radius: 12px;">
            <h4 style="margin: 0 0 12px 0; color: #166534;">✓ POCs with Leading EPCs</h4>
            <p style="margin: 0; color: #15803d; font-size: 14px;">
            Proof-of-concepts deployed with major Engineering, Procurement, Construction firms
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Technology Stack**")
        
        tech_data = {
            'Component': ['Multi-Agent Orchestration', 'Document Intelligence', 'Legacy System Control', 'Workflow Automation'],
            'Technology': ['Real-time coordination', 'NLP + computer vision', 'UI automation', 'Process mining']
        }
        
        st.dataframe(pd.DataFrame(tech_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Why Energy Infrastructure?**")
        
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #f59e0b; margin-top: 15px;">
            <p style="margin: 0; color: #92400e; font-size: 14px; line-height: 1.6;">
            Energy projects resemble building cities: power generation, utilities, digital economy backbone (including AI compute infrastructure). Timelines are the choke point—4 years total with 2 years in pre-construction before first work begins.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #92400e; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #f59e0b; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-Agent Systems</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Real-time coordination</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #f59e0b; font-weight: 700; margin: 0 0 6px 0;">✓ $5.5M Funded</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">OpenAI/xAI investors</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #f59e0b; font-weight: 700; margin: 0 0 6px 0;">✓ 67% Timeline Reduction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">24 → 8 months</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #f59e0b; font-weight: 700; margin: 0 0 6px 0;">✓ World-Class Angels</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Yann LeCun, MIT E14</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Candid Intelligence</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)