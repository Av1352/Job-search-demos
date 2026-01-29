"""
Trellis AI - Streamline Healthcare Paperwork
AI for automating medical forms and administrative workflows
Built for Trellis AI by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Trellis AI", page_icon="📋", layout="wide")

# Form types
FORM_TYPES = {
    'Patient Intake': {'fields': 32, 'completion_time': '3.2 min', 'manual_time': '15 min'},
    'Insurance Forms': {'fields': 24, 'completion_time': '2.1 min', 'manual_time': '12 min'},
    'Consent Forms': {'fields': 18, 'completion_time': '1.5 min', 'manual_time': '8 min'},
    'Medical History': {'fields': 45, 'completion_time': '4.5 min', 'manual_time': '25 min'},
    'Referral Paperwork': {'fields': 28, 'completion_time': '2.8 min', 'manual_time': '18 min'}
}

# Automation stats
AUTOMATION_STATS = {
    'Field Auto-Fill Rate': 96.8,
    'Accuracy': 99.1,
    'Patient Satisfaction': 4.9,
    'Staff Time Saved': 82.5,
    'Form Completion Rate': 97.3
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #8b5cf6 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">📋</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Trellis AI</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Streamline Healthcare Paperwork</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Intelligent forms • Auto-fill • 82.5% time savings • 4.9/5 patient satisfaction</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Smart Forms", "📊 Clinic Dashboard", "⚡ Performance", "💡 Features"])

with tab1:
    st.markdown("### Intelligent Form Completion")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Select Form Type**")
        
        form_type = st.selectbox("Form Type", list(FORM_TYPES.keys()))
        
        st.markdown("**Patient Information**")
        
        patient_name = st.text_input("Patient Name", "Emily Rodriguez")
        dob = st.date_input("Date of Birth", value=datetime(1985, 7, 22))
        
        st.markdown("**AI Features**")
        
        enable_autofill = st.checkbox("Enable Auto-Fill", value=True)
        enable_validation = st.checkbox("Real-Time Validation", value=True)
        prefill_ehr = st.checkbox("Pre-fill from EHR", value=True)
        
        st.markdown("**Voice Input**")
        use_voice = st.checkbox("Voice Dictation Mode", value=False)
        
        process_btn = st.button("📋 Process Form", type="primary", use_container_width=True)
    
    with col2:
        if process_btn:
            st.markdown("**Form Processing**")
            
            with st.spinner("Processing form..."):
                import time
                time.sleep(1.3)
            
            st.success(f"✅ Form completed in {FORM_TYPES[form_type]['completion_time']}!")
            
            fields_filled = FORM_TYPES[form_type]['fields']
            auto_filled = int(fields_filled * 0.968)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #8b5cf6 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Form Completion Summary</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Total Fields</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{fields_filled}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Auto-Filled</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{auto_filled}/{fields_filled}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Time Taken</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{FORM_TYPES[form_type]['completion_time']}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Time Saved</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{FORM_TYPES[form_type]['manual_time']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Auto-Filled Fields**")
            
            sample_fields = {
                'Field': ['Full Name', 'Date of Birth', 'Address', 'Phone', 'Email', 'Insurance ID', 'PCP Name', 'Allergies'],
                'Value': ['Emily Rodriguez', '07/22/1985', '123 Main St, Boston MA', '(617) 555-0123', 'emily.r@email.com', 'UHC987654321', 'Dr. Sarah Kim', 'Penicillin'],
                'Source': ['Manual', 'EHR', 'EHR', 'EHR', 'EHR', 'Insurance DB', 'EHR', 'EHR'],
                'Confidence': ['100%', '100%', '99.8%', '99.5%', '99.2%', '98.7%', '99.1%', '99.9%']
            }
            st.dataframe(pd.DataFrame(sample_fields), hide_index=True, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", "99.1%", "✓")
            col2.metric("Auto-Fill Rate", f"{auto_filled/fields_filled*100:.1f}%", "+3.2%")
            col3.metric("Patient Time", "4.5 min", f"-{FORM_TYPES[form_type]['manual_time']}")

with tab2:
    st.markdown("### Clinic Operations Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Forms Today", "847", "+123")
    col2.metric("Time Saved", "18.7 hrs", "+2.3 hrs")
    col3.metric("Completion Rate", "97.3%", "+1.2%")
    col4.metric("Patient NPS", "4.9/5", "+0.4")
    
    st.markdown("**Form Volume by Type**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        form_counts = [312, 234, 156, 98, 47]
        
        fig1 = go.Figure(data=[go.Bar(
            x=list(FORM_TYPES.keys()),
            y=form_counts,
            marker=dict(color='#8b5cf6'),
            text=form_counts,
            textposition='auto'
        )])
        fig1.update_layout(yaxis_title='Forms Completed', height=300)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = go.Figure(data=[go.Pie(
            labels=list(FORM_TYPES.keys()),
            values=form_counts,
            hole=0.4,
            marker=dict(colors=['#8b5cf6', '#a855f7', '#c084fc', '#d8b4fe', '#e9d5ff'])
        )])
        fig2.update_layout(height=300, title='Distribution')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("**Weekly Trends**")
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    forms_completed = [782, 823, 847, 856, 791]
    time_saved = [16.2, 17.1, 18.7, 19.2, 16.8]
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=days, y=forms_completed,
        name='Forms Completed',
        marker=dict(color='#8b5cf6')
    ))
    fig3.add_trace(go.Scatter(
        x=days, y=time_saved,
        name='Hours Saved',
        yaxis='y2',
        line=dict(color='#10b981', width=3)
    ))
    fig3.update_layout(
        yaxis=dict(title='Forms'),
        yaxis2=dict(title='Hours Saved', overlaying='y', side='right'),
        height=300
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.markdown("### Automation Performance")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Automation Metrics**")
        
        auto_data = []
        for metric, value in AUTOMATION_STATS.items():
            if 'Satisfaction' in metric:
                display = f"{value}/5"
            else:
                display = f"{value}%"
            
            auto_data.append({
                'Metric': metric,
                'Value': display,
                'Status': '✅ Excellent'
            })
        
        st.dataframe(pd.DataFrame(auto_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Time Savings Analysis**")
        
        savings = {
            'Form Type': list(FORM_TYPES.keys()),
            'Manual Time': ['15 min', '12 min', '8 min', '25 min', '18 min'],
            'AI Time': ['3.2 min', '2.1 min', '1.5 min', '4.5 min', '2.8 min'],
            'Savings': ['11.8 min', '9.9 min', '6.5 min', '20.5 min', '15.2 min']
        }
        st.dataframe(pd.DataFrame(savings), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Performance Scores**")
        
        fig4 = go.Figure(data=[go.Bar(
            x=list(AUTOMATION_STATS.keys()),
            y=list(AUTOMATION_STATS.values()),
            marker=dict(color='#8b5cf6'),
            text=[f"{v}%" if v < 10 else f"{v}%" if 'Satisfaction' not in k else f"{v}/5" for k, v in AUTOMATION_STATS.items()],
            textposition='auto'
        )])
        fig4.update_layout(yaxis=dict(range=[80, 100]), height=250)
        st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("**Monthly Impact**")
        
        monthly_impact = {
            'Metric': ['Forms Processed', 'Hours Saved', 'Cost Reduction', 'Patients Served'],
            'Value': ['18,450', '1,120 hrs', '$168K', '12,300']
        }
        st.dataframe(pd.DataFrame(monthly_impact), hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### Platform Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Intelligent Forms**")
        st.markdown("""
        - ✅ Auto-fill from EHR data
        - ✅ Smart field suggestions
        - ✅ Voice dictation support
        - ✅ Mobile-optimized
        - ✅ Multi-language support
        - ✅ Accessibility compliant
        """)
        
        st.markdown("**Data Sources**")
        st.markdown("""
        - ✅ EHR integration (Epic, Cerner)
        - ✅ Insurance databases
        - ✅ Patient portals
        - ✅ Previous forms
        - ✅ Pharmacy records
        - ✅ Lab results
        """)
    
    with col2:
        st.markdown("**Workflow Automation**")
        st.markdown("""
        - ✅ Conditional logic (skip irrelevant fields)
        - ✅ Real-time validation
        - ✅ Error prevention
        - ✅ Progress saving
        - ✅ E-signature integration
        - ✅ Automated routing
        """)
        
        st.markdown("**Analytics & Reporting**")
        st.markdown("""
        - ✅ Completion rate tracking
        - ✅ Time-to-complete metrics
        - ✅ Patient satisfaction surveys
        - ✅ Staff efficiency reports
        - ✅ Bottleneck identification
        - ✅ Trend analysis
        """)
    
    st.markdown("**Form Types Supported**")
    
    supported = {
        'Form Category': ['Patient Intake', 'Insurance', 'Consent', 'Medical History', 'Referral'],
        'Fields': [32, 24, 18, 45, 28],
        'Auto-Fill Rate': ['97.2%', '96.8%', '95.3%', '96.1%', '97.5%'],
        'Avg Time': ['3.2 min', '2.1 min', '1.5 min', '4.5 min', '2.8 min']
    }
    st.dataframe(pd.DataFrame(supported), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #5b21b6; font-size: 24px; font-weight: 800;">💡 System Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ 96.8% Auto-Fill</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Minimal patient effort</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ 82.5% Time Saved</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">3.2 min vs 15 min manual</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ 4.9/5 Satisfaction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">High patient approval</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ 18,450 Forms/Month</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">High-volume automation</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #8b5cf6 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Trellis AI</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashvi Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)