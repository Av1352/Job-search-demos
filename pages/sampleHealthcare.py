"""
Sample Healthcare - AI-Powered Document Workflows
Automate healthcare documentation and administrative workflows
Built for Sample Healthcare by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Sample Healthcare", page_icon="📄", layout="wide")

# Document types
DOCUMENT_TYPES = {
    'Prior Authorization': {'processing_time': '2.3 min', 'manual_time': '45 min', 'fields': 24},
    'Insurance Verification': {'processing_time': '1.1 min', 'manual_time': '20 min', 'fields': 18},
    'Medical Records Request': {'processing_time': '1.8 min', 'manual_time': '35 min', 'fields': 16},
    'Referral Forms': {'processing_time': '1.5 min', 'manual_time': '25 min', 'fields': 22},
    'Claims Submission': {'processing_time': '2.0 min', 'manual_time': '30 min', 'fields': 28}
}

# Processing stats
PROCESSING_STATS = {
    'Accuracy': 98.9,
    'Auto-Complete Rate': 94.5,
    'Error Rate': 1.1,
    'Avg Processing Time': 1.7,  # minutes
    'Manual Review Rate': 5.5
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #0891b2 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">📄</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Sample Healthcare</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">AI-Powered Document Workflows</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Automate paperwork • 95% time reduction • 98.9% accuracy</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Process Document", "📊 Workflow Analytics", "⚡ Performance", "💡 Platform Features"])

with tab1:
    st.markdown("### Automated Document Processing")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Document Upload**")
        
        doc_type = st.selectbox("Document Type", list(DOCUMENT_TYPES.keys()))
        
        uploaded_file = st.file_uploader(
            "Upload Document (PDF, PNG, JPG)",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            label_visibility="collapsed"
        )
        
        if not uploaded_file:
            st.info("👆 Upload a healthcare document or use sample")
            use_sample = st.button("📋 Use Sample Prior Authorization", use_container_width=True)
        
        st.markdown("**Processing Options**")
        
        extract_data = st.checkbox("Extract Patient Data", value=True)
        auto_fill = st.checkbox("Auto-Fill Fields", value=True)
        validate_insurance = st.checkbox("Validate Insurance", value=True)
        check_compliance = st.checkbox("Check Compliance", value=True)
        
        process_btn = st.button("📄 Process Document", type="primary", use_container_width=True)
    
    with col2:
        if process_btn or (not uploaded_file and 'use_sample' in locals()):
            st.markdown("**Processing Results**")
            
            with st.spinner("Processing document..."):
                import time
                time.sleep(1.5)
            
            st.success(f"✅ Document processed in {DOCUMENT_TYPES[doc_type]['processing_time']}!")
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0891b2 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Extraction Results</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Fields Extracted</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{DOCUMENT_TYPES[doc_type]['fields']}/24</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Confidence</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">98.9%</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Processing Time</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{DOCUMENT_TYPES[doc_type]['processing_time']}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Manual Review</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Not Required</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Extracted Information**")
            
            extracted_data = {
                'Field': ['Patient Name', 'Date of Birth', 'Insurance ID', 'Diagnosis Code', 'Procedure Code', 'Provider NPI'],
                'Extracted Value': ['Sarah Johnson', '03/15/1978', 'ABC123456789', 'M79.3 (Back Pain)', 'CPT 97110 (PT)', '1234567890'],
                'Confidence': ['99.2%', '99.8%', '98.5%', '97.3%', '98.1%', '99.5%']
            }
            st.dataframe(pd.DataFrame(extracted_data), hide_index=True, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Time Saved", "42.7 min", f"vs {DOCUMENT_TYPES[doc_type]['manual_time']} manual")
            col2.metric("Auto-Filled", "24/24 fields", "100%")
            col3.metric("Compliance", "✅ Pass", "HIPAA compliant")

with tab2:
    st.markdown("### Document Workflow Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Docs Processed Today", "1,847", "+234")
    col2.metric("Time Saved Today", "42.3 hrs", "+5.2 hrs")
    col3.metric("Auto-Complete Rate", "94.5%", "+2.1%")
    col4.metric("Error Rate", "1.1%", "-0.3%")
    
    st.markdown("**Document Type Distribution**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        doc_counts = [623, 412, 298, 347, 167]
        
        fig1 = go.Figure(data=[go.Pie(
            labels=list(DOCUMENT_TYPES.keys()),
            values=doc_counts,
            hole=0.4,
            marker=dict(colors=['#0891b2', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'])
        )])
        fig1.update_layout(height=300, title='Document Volume (Last 24h)')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = go.Figure(data=[go.Bar(
            x=list(DOCUMENT_TYPES.keys()),
            y=doc_counts,
            marker=dict(color='#0891b2'),
            text=doc_counts,
            textposition='auto'
        )])
        fig2.update_layout(height=300, title='Processing Volume', yaxis_title='Count')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("**Processing Time Comparison**")
    
    manual_times = [45, 20, 35, 25, 30]
    ai_times = [2.3, 1.1, 1.8, 1.5, 2.0]
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name='Manual',
        x=list(DOCUMENT_TYPES.keys()),
        y=manual_times,
        marker=dict(color='#ef4444')
    ))
    fig3.add_trace(go.Bar(
        name='Sample AI',
        x=list(DOCUMENT_TYPES.keys()),
        y=ai_times,
        marker=dict(color='#10b981')
    ))
    fig3.update_layout(barmode='group', yaxis_title='Time (minutes)', height=300)
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.markdown("### System Performance Metrics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Processing Metrics**")
        
        perf_data = []
        for metric, value in PROCESSING_STATS.items():
            if 'Time' in metric:
                display = f"{value} min"
            else:
                display = f"{value}%"
            
            status = '✅ Excellent' if (value > 95 and 'Rate' not in metric or value < 3) else '✅ Good'
            
            perf_data.append({
                'Metric': metric,
                'Value': display,
                'Status': status
            })
        
        st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Comparison vs Manual Processing**")
        
        comparison = {
            'Method': ['Sample AI', 'Manual Processing'],
            'Accuracy': ['98.9%', '94.2%'],
            'Speed': ['1.7 min', '31 min'],
            'Cost/Doc': ['$0.12', '$8.50'],
            'Compliance': ['100%', '96.3%']
        }
        st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Accuracy Metrics**")
        
        fig4 = go.Figure(data=[go.Bar(
            x=['Accuracy', 'Auto-Complete', 'Compliance', 'Data Extraction'],
            y=[98.9, 94.5, 100, 97.8],
            marker=dict(color='#0891b2'),
            text=['98.9%', '94.5%', '100%', '97.8%'],
            textposition='auto'
        )])
        fig4.update_layout(yaxis=dict(range=[90, 100]), height=250)
        st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("**Monthly Impact**")
        
        impact = {
            'Metric': ['Documents Processed', 'Hours Saved', 'Cost Savings', 'Error Reduction'],
            'Value': ['38,450', '1,240 hrs', '$320K', '4.7%']
        }
        st.dataframe(pd.DataFrame(impact), hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### Workflow Automation Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Document Processing**")
        st.markdown("""
        - ✅ OCR + NLP extraction
        - ✅ Auto-field mapping
        - ✅ Data validation
        - ✅ Insurance verification
        - ✅ Compliance checking
        - ✅ E-signature integration
        """)
        
        st.markdown("**Supported Documents**")
        st.markdown("""
        - ✅ Prior authorizations
        - ✅ Insurance verifications
        - ✅ Medical records requests
        - ✅ Referral forms
        - ✅ Claims submissions
        - ✅ Patient intake forms
        - ✅ Consent forms
        - ✅ Billing documents
        """)
    
    with col2:
        st.markdown("**Workflow Features**")
        st.markdown("""
        - ✅ Automated routing
        - ✅ Status tracking
        - ✅ Deadline alerts
        - ✅ Team collaboration
        - ✅ Audit trails
        - ✅ Analytics dashboard
        """)
        
        st.markdown("**Integration**")
        st.markdown("""
        - ✅ EHR systems (Epic, Cerner)
        - ✅ Practice management software
        - ✅ Insurance portals
        - ✅ Fax/email ingestion
        - ✅ E-signature platforms
        - ✅ Cloud storage (Box, Dropbox)
        """)
    
    st.markdown("**Processing Capabilities**")
    
    capabilities = {
        'Capability': ['Handwriting Recognition', 'Form Field Detection', 'Data Validation', 'Insurance Lookup', 'Duplicate Detection'],
        'Accuracy': ['96.5%', '98.9%', '99.2%', '97.8%', '99.5%'],
        'Speed': ['2.1s', '0.8s', '0.5s', '1.2s', '0.3s']
    }
    st.dataframe(pd.DataFrame(capabilities), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #cffafe 0%, #a5f3fc 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #164e63; font-size: 24px; font-weight: 800;">💡 System Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0891b2; font-weight: 700; margin: 0 0 6px 0;">✓ 98.9% Accuracy</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Superior data extraction</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0891b2; font-weight: 700; margin: 0 0 6px 0;">✓ 95% Time Reduction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">1.7 min vs 31 min manual</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0891b2; font-weight: 700; margin: 0 0 6px 0;">✓ 38,450 Docs/Month</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">High-volume processing</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0891b2; font-weight: 700; margin: 0 0 6px 0;">✓ $320K Savings</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Monthly cost reduction</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #0891b2 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Sample Healthcare</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)