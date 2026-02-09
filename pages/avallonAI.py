"""
Avallon AI - Insurance Claims Automation Platform
Full-stack AI agents for claims operations
Built for Avallon AI by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Avallon AI - Claims Automation", page_icon="📋", layout="wide")

# Claims data
CLAIM_TYPES = {
    'Workers Compensation': {'avg_processing': 45, 'ai_processing': 3.2, 'automation': 89.5, 'volume': 3847},
    'Auto': {'avg_processing': 38, 'ai_processing': 2.8, 'automation': 91.2, 'volume': 4523},
    'Property': {'avg_processing': 52, 'ai_processing': 4.1, 'automation': 85.3, 'volume': 2156}
}

# AI agent capabilities
AI_AGENTS = {
    'Voice AI Agent': 'Handles intake calls, extracts claim details, natural conversation',
    'Document Processing Agent': 'Analyzes PDFs, medical reports, photos, invoices',
    'Email Coordination Agent': 'Manages communication with claimants, vendors, employers',
    'Data Entry Agent': 'Automatically updates CMS systems with structured data',
    'Exposure Analysis Agent': 'Assesses claim severity, estimates reserves',
    'Status Tracking Agent': 'Monitors claim lifecycle, provides updates'
}

# Performance metrics
PERFORMANCE_METRICS = {
    'Processing Time': {'before': 45, 'after': 3.5, 'unit': 'min'},
    'Automation Rate': {'before': 15, 'after': 87, 'unit': '%'},
    'Cost per Claim': {'before': 45, 'after': 2.40, 'unit': '$'},
    'Customer Satisfaction': {'before': 3.4, 'after': 4.7, 'unit': '/5'}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">📋</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Avallon AI</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Insurance Claims Automation</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">YC X25 • $4.6M seed • 10x revenue growth</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Claims Processing", "📊 Platform Analytics", "🤖 AI Agents", "💡 Technology"])

with tab1:
    st.markdown("### Automated Claims Workflow")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**New Claim Intake**")
        
        claim_type = st.selectbox("Claim Type", list(CLAIM_TYPES.keys()))
        claimant_name = st.text_input("Claimant Name", "John Smith")
        policy_number = st.text_input("Policy Number", "WC-2025-48392")
        incident_date = st.date_input("Incident Date", datetime.now() - timedelta(days=3))
        
        st.markdown("**Claim Details**")
        
        description = st.text_area("Incident Description",
                                   "Employee slipped on wet floor at warehouse, injured lower back. Seeking medical treatment.",
                                   height=80)
        
        severity = st.select_slider("Severity Assessment", 
                                    options=["Minor", "Moderate", "Serious", "Critical"],
                                    value="Moderate")
        
        st.markdown("**Documentation**")
        
        has_photos = st.checkbox("Incident photos uploaded", value=True)
        has_medical = st.checkbox("Medical records attached", value=True)
        has_witness = st.checkbox("Witness statements", value=False)
        
        process_btn = st.button("📋 Process with AI Agents", type="primary", use_container_width=True)
    
    with col2:
        if process_btn:
            st.markdown("**AI Agent Processing Workflow**")
            
            import time
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            claim_data = CLAIM_TYPES[claim_type]
            
            stages = [
                ("🎤 Voice AI: Conducting intake call with claimant...", 0.15),
                ("📄 Document Agent: Extracting data from medical reports...", 0.3),
                ("🔍 Analysis Agent: Categorizing claim type and severity...", 0.5),
                ("💾 Data Agent: Updating CMS system automatically...", 0.7),
                ("👤 Routing Agent: Assigning to best-fit adjuster...", 0.85),
                ("✅ Coordination Agent: Notifying all parties...", 1.0)
            ]
            
            for stage, progress in stages:
                status_text.text(stage)
                progress_bar.progress(progress)
                time.sleep(0.5)
            
            st.success("✅ Claim processed and routed automatically!")
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%); padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Claim Processing Summary</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Claim ID</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">WC-2026-48392</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Status</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">✅ Assigned</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Processing Time</p>
                        <p style="font-size: 24px; color: white; font-weight: 900; margin: 0;">{claim_data['ai_processing']} min</p>
                        <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">vs {claim_data['avg_processing']} min manual</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Assigned To</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Adjuster M. Chen</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Actions Completed by AI**")
            
            actions = pd.DataFrame({
                'Agent': ['Voice AI', 'Document AI', 'Data Entry', 'Coordination', 'Exposure Analysis'],
                'Action': [
                    'Conducted intake call, extracted details',
                    'Analyzed injury photos + medical records',
                    'Updated CMS with structured claim data',
                    'Notified employer, medical provider, adjuster',
                    'Estimated reserves at $12,500 - $18,000'
                ],
                'Status': ['✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete']
            })
            
            st.dataframe(actions, hide_index=True, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Time Saved", "42 min", f"vs {claim_data['avg_processing']} min manual")
            col2.metric("Accuracy", "98.3%", "Validated")
            col3.metric("Automation", f"{claim_data['automation']}%", "Full-stack")
            col4.metric("Cost", "$2.40", "vs $45 manual")

with tab2:
    st.markdown("### Claims Operations Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Claims/Month", "8,450", "+10x growth")
    col2.metric("Automation Rate", "87.2%", "High")
    col3.metric("Avg Processing", "3.5 min", "-93%")
    col4.metric("Customer Satisfaction", "4.7/5", "+38%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Claims by Type")
        
        claims_data = []
        for claim_type, data in CLAIM_TYPES.items():
            claims_data.append({
                'Type': claim_type,
                'Monthly Volume': data['volume'],
                'Automation': f"{data['automation']}%",
                'Time Saved': f"{data['avg_processing'] - data['ai_processing']:.1f} min"
            })
        
        st.dataframe(pd.DataFrame(claims_data), hide_index=True, use_container_width=True)
        
        st.markdown("### Processing Time Comparison")
        
        claim_types_list = list(CLAIM_TYPES.keys())
        manual_times = [CLAIM_TYPES[ct]['avg_processing'] for ct in claim_types_list]
        ai_times = [CLAIM_TYPES[ct]['ai_processing'] for ct in claim_types_list]
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Bar(
            x=claim_types_list,
            y=manual_times,
            name='Manual',
            marker_color='#94a3b8'
        ))
        
        fig1.add_trace(go.Bar(
            x=claim_types_list,
            y=ai_times,
            name='Avallon AI',
            marker_color='#0ea5e9'
        ))
        
        fig1.update_layout(
            barmode='group',
            yaxis_title='Processing Time (min)',
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### Monthly Growth")
        
        months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        claims_processed = [340, 890, 2340, 4120, 5680, 6890, 7820, 8450]
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=months,
            y=claims_processed,
            mode='lines+markers',
            line=dict(color='#0ea5e9', width=3),
            fill='tozeroy',
            fillcolor='rgba(14, 165, 233, 0.1)'
        ))
        
        fig2.update_layout(
            yaxis_title='Claims Processed',
            height=300
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### Economic Impact")
        
        impact_data = {
            'Category': ['Labor cost savings', 'Error reduction', 'Faster settlements', 'Improved satisfaction'],
            'Annual Value': ['$12.5M', '$3.8M', '$5.2M', '$2.1M']
        }
        
        st.dataframe(pd.DataFrame(impact_data), hide_index=True, use_container_width=True)

with tab3:
    st.markdown("### Full-Stack AI Agent System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**AI Agent Capabilities**")
        
        for agent, description in AI_AGENTS.items():
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #0ea5e9;">
                <p style="margin: 0; font-weight: 700; color: #075985;">{agent}</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**Multi-Modal Processing**")
        st.markdown("""
        - ✅ Voice calls (inbound/outbound)
        - ✅ Documents (PDF, images, scans)
        - ✅ Email communication
        - ✅ Structured data (CMS entry)
        - ✅ Unstructured text analysis
        - ✅ Photo damage assessment
        """)
    
    with col2:
        st.markdown("**Workflow Automation**")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bae6fd 100%); padding: 20px; border-radius: 12px;">
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #075985;">1. Intake</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #0c4a6e;">Voice AI handles initial call, gathers details</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #075985;">2. Documentation</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #0c4a6e;">AI extracts data from photos, medical reports</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #075985;">3. Categorization</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #0c4a6e;">AI classifies claim type, estimates reserves</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #075985;">4. CMS Update</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #0c4a6e;">Structured data entry automatically</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #075985;">5. Coordination</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #0c4a6e;">Notify all parties (employer, vendor, adjuster)</p>
            </div>
            <div>
                <p style="margin: 0; font-weight: 700; color: #075985;">6. Tracking</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #0c4a6e;">Monitor lifecycle, provide status updates</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Integration**")
        st.markdown("""
        - 🏢 CMS platforms (all major)
        - 📞 IVR phone systems
        - 📊 Data warehouses
        - 📧 Email systems
        - 💼 TPA/MGA systems
        """)

with tab4:
    st.markdown("### Platform Technology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Performance vs Manual**")
        
        performance_data = []
        for metric, values in PERFORMANCE_METRICS.items():
            unit = values['unit']
            performance_data.append({
                'Metric': metric,
                'Manual': f"{values['before']}{unit}",
                'Avallon AI': f"{values['after']}{unit}",
                'Improvement': f"+{abs(values['after'] - values['before']):.1f}{unit if unit != '$' else ''}"
            })
        
        st.dataframe(pd.DataFrame(performance_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Company Traction**")
        
        traction_data = {
            'Milestone': ['Founded', 'YC Batch', 'Revenue Growth', 'Funding'],
            'Status': ['2025', 'X25', '10x in 3 months', '$4.6M seed']
        }
        
        st.dataframe(pd.DataFrame(traction_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Team Background**")
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px;">
            <h4 style="margin: 0 0 12px 0; color: #92400e;">Founding Team</h4>
            <ul style="margin: 0; padding-left: 20px; color: #78350f; font-size: 14px;">
                <li><strong>Cornelius Schramm (CEO):</strong> Ex-FINN (founding US engineer, $15M ARR)</li>
                <li><strong>Bryan Guin (COO):</strong> Ex-Agentive (YC S23), Cornell ML researcher, ex-EY</li>
                <li><strong>Moritz Bartusch:</strong> Ex-Taktile (ML for fintechs/insurers), MIT</li>
                <li><strong>Leander Peter:</strong> Ex-FINN (core infrastructure)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Market Opportunity**")
        
        st.markdown("""
        <div style="background: #dcfce7; padding: 20px; border-radius: 12px; margin-top: 15px;">
            <h4 style="margin: 0 0 10px 0; color: #166534;">📊 Insurance Industry Crisis</h4>
            <p style="margin: 0; color: #15803d; font-size: 14px;">
            • <strong>400,000 workers</strong> exiting by 2026 (BLS)<br>
            • TPAs hit hardest (adjuster capacity crunch)<br>
            • Rising claim volumes + operational complexity<br>
            • $100B+ automation opportunity
            </p>
        </div>
        """, unsafe_allow_html=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #dbeafe 0%, #bae6fd 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #075985; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0ea5e9; font-weight: 700; margin: 0 0 6px 0;">✓ 10x Revenue Growth</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">3 months at YC</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0ea5e9; font-weight: 700; margin: 0 0 6px 0;">✓ 87% Automation</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Full-stack claims</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0ea5e9; font-weight: 700; margin: 0 0 6px 0;">✓ $4.6M Funded</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Frontline Ventures</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0ea5e9; font-weight: 700; margin: 0 0 6px 0;">✓ 93% Time Reduction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">3.5 min vs 45 min</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Avallon AI</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)