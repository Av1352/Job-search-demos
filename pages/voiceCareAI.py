"""
VoiceCare AI - Healthcare Back-Office Voice Automation
AI voice agent "Joy" for administrative conversations
Built for VoiceCare AI by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="VoiceCare AI - Voice Automation", page_icon="🎤", layout="wide")

# Call types and automation
CALL_TYPES = {
    'Benefits Verification': {'avg_duration': 28, 'joy_duration': 3.2, 'automation': 98.5, 'volume': 12450},
    'Prior Authorization': {'avg_duration': 45, 'joy_duration': 5.8, 'automation': 96.2, 'volume': 8920},
    'Claims Follow-up': {'avg_duration': 32, 'joy_duration': 4.1, 'automation': 97.8, 'volume': 15680},
    'Appointment Scheduling': {'avg_duration': 12, 'joy_duration': 1.8, 'automation': 99.1, 'volume': 18340},
    'Insurance Verification': {'avg_duration': 18, 'joy_duration': 2.5, 'automation': 98.9, 'volume': 10230}
}

# Joy capabilities
JOY_CAPABILITIES = {
    'Long Conversations': 'Handles 45+ min calls with extended hold times',
    'Zero-Skip Architecture': 'Never bypasses critical questions',
    'Hallucination-Free': 'Context-bound, verifiable responses only',
    'HIPAA Compliant': 'SOC 2 Type II certified for healthcare',
    'Multi-Modal': 'Voice + data integration with EHR systems',
    'RLHF Trained': 'Reinforcement learning from healthcare experts'
}

# Performance metrics
PERFORMANCE_METRICS = {
    'Call Completion': {'joy': 98.7, 'human': 92.3},
    'Accuracy Rate': {'joy': 99.2, 'human': 94.5},
    'Avg Handle Time': {'joy': 3.5, 'human': 28.0},
    'Cost per Call': {'joy': 0.85, 'human': 12.50},
    'Availability': {'joy': 100, 'human': 42}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🎤</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">VoiceCare AI</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Healthcare Back-Office Voice Automation</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Meet "Joy" • 98.7% automation • Zero-skip architecture</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎤 Joy AI Agent", "📞 Call Automation", "📊 Platform Analytics", "💡 Technology"])

with tab1:
    st.markdown("### Meet Joy - Your AI Voice Agent")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Simulate a Call with Joy**")
        
        call_type = st.selectbox("Call Type", list(CALL_TYPES.keys()))
        
        st.markdown("**Call Details**")
        
        patient_name = st.text_input("Patient Name", "Jennifer Martinez")
        patient_id = st.text_input("Patient ID", "PAT-28395")
        insurance = st.selectbox("Insurance Provider", 
                                ["UnitedHealthcare", "Aetna", "BCBS", "Cigna", "Humana"])
        
        if call_type == "Benefits Verification":
            procedure = st.text_input("Procedure", "MRI - Lower Back")
            cpt_code = st.text_input("CPT Code", "72148")
        elif call_type == "Prior Authorization":
            medication = st.text_input("Medication/Procedure", "Dupixent (Dupilumab)")
            diagnosis = st.text_input("Diagnosis", "Atopic Dermatitis")
        elif call_type == "Claims Follow-up":
            claim_id = st.text_input("Claim Number", "CLM-2025-48392")
            claim_amount = st.text_input("Claim Amount", "$3,845.00")
        
        start_call = st.button("🎤 Start Call with Joy", type="primary", use_container_width=True)
    
    with col2:
        if start_call:
            st.markdown("**Joy AI Agent - Live Call Simulation**")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            
            call_data = CALL_TYPES[call_type]
            
            stages = [
                ("🔊 Joy: Hello, this is Joy from VoiceCare AI calling on behalf of Memorial Hospital...", 0.1),
                ("🎧 Payer: Thank you for calling. Can I have your reference number?", 0.2),
                (f"🔊 Joy: Patient ID is {patient_id}, calling regarding {call_type.lower()}...", 0.3),
                ("⏳ Joy: Navigating hold time... playing hold music for payer...", 0.4),
                ("🎧 Payer: Thank you for holding. I have the account pulled up...", 0.5),
                ("🔊 Joy: Extracting information from EHR, verifying patient details...", 0.6),
                ("🎧 Payer: Authorization approved. Reference number is AUTH-2025-8392...", 0.8),
                ("🔊 Joy: Confirming details, updating system records...", 0.9),
                ("✅ Joy: Call completed successfully. Updating EHR with results...", 1.0)
            ]
            
            for stage, progress in stages:
                status_text.markdown(f"**{stage}**")
                progress_bar.progress(progress)
                time.sleep(0.6)
            
            st.success("✅ Call completed successfully by Joy!")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%); padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Call Summary</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Call Duration</p>
                        <p style="font-size: 24px; color: white; font-weight: 900; margin: 0;">{} min</p>
                        <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">vs {} min manual</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Automation Rate</p>
                        <p style="font-size: 24px; color: white; font-weight: 900; margin: 0;">{}%</p>
                        <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Fully autonomous</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Status</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">✅ Approved</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Time Saved</p>
                        <p style="font-size: 24px; color: white; font-weight: 900; margin: 0;">{} min</p>
                    </div>
                </div>
            </div>
            """.format(
                round(call_data['joy_duration'], 1),
                call_data['avg_duration'],
                round(call_data['automation'], 1),
                round(call_data['avg_duration'] - call_data['joy_duration'], 1)
            ), unsafe_allow_html=True)
            
            st.markdown("**Call Transcript Analysis**")
            
            transcript_data = {
                'Speaker': ['Joy', 'Payer', 'Joy', 'Payer', 'Joy'],
                'Action': [
                    'Introduced purpose, provided patient info',
                    'Requested reference number',
                    'Provided ID, waited on hold (14 min)',
                    'Confirmed authorization',
                    'Documented results in EHR'
                ],
                'Duration': ['0:45', '0:30', '14:20', '1:15', '0:35']
            }
            
            st.dataframe(pd.DataFrame(transcript_data), hide_index=True, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Questions Asked", "12", "Complete")
            col2.metric("Hold Time", "14.2 min", "Handled")
            col3.metric("Accuracy", "100%", "Verified")
            col4.metric("Cost", "$0.85", "vs $12.50")

with tab2:
    st.markdown("### Automated Call Operations")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Calls Today", "3,847", "+820")
    col2.metric("Automation Rate", "98.3%", "High")
    col3.metric("Avg Handle Time", "3.5 min", "-89%")
    col4.metric("Time Saved", "1,640 hrs", "Monthly")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Call Volume by Type")
        
        call_volume_data = []
        for call, data in CALL_TYPES.items():
            call_volume_data.append({
                'Call Type': call,
                'Daily Volume': data['volume'],
                'Automation': f"{data['automation']}%",
                'Time Saved': f"{round((data['avg_duration'] - data['joy_duration']) * data['volume'] / 60, 0)} hrs"
            })
        
        st.dataframe(pd.DataFrame(call_volume_data), hide_index=True, use_container_width=True)
        
        st.markdown("### Call Duration Comparison")
        
        call_types_list = list(CALL_TYPES.keys())
        human_times = [CALL_TYPES[ct]['avg_duration'] for ct in call_types_list]
        joy_times = [CALL_TYPES[ct]['joy_duration'] for ct in call_types_list]
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Bar(
            x=call_types_list,
            y=human_times,
            name='Human Agent',
            marker_color='#94a3b8'
        ))
        
        fig1.add_trace(go.Bar(
            x=call_types_list,
            y=joy_times,
            name='Joy AI',
            marker_color='#06b6d4'
        ))
        
        fig1.update_layout(
            barmode='group',
            yaxis_title='Duration (minutes)',
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### Daily Call Pattern")
        
        hours = list(range(8, 18))
        call_volume = [245, 380, 520, 680, 745, 820, 790, 680, 540, 420]
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=hours,
            y=call_volume,
            mode='lines+markers',
            line=dict(color='#3b82f6', width=3),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        fig2.update_layout(
            xaxis_title='Hour of Day',
            yaxis_title='Calls Handled by Joy',
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### Call Outcomes")
        
        outcomes = pd.DataFrame({
            'Outcome': ['Approved', 'Pending', 'Denied', 'Escalated'],
            'Count': [3247, 385, 142, 73],
            'Percentage': [84.4, 10.0, 3.7, 1.9]
        })
        
        fig3 = px.pie(outcomes, values='Count', names='Outcome',
                     color_discrete_sequence=['#22c55e', '#eab308', '#ef4444', '#94a3b8'])
        fig3.update_traces(textposition='inside', textinfo='percent+label')
        fig3.update_layout(height=250, showlegend=False)
        
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.markdown("### Platform Performance")
    
    # Impact metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">89%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Time Reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">93%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Cost Savings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">98.7%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Call Completion Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Joy vs Human Performance")
        
        performance_data = []
        for metric, values in PERFORMANCE_METRICS.items():
            performance_data.append({
                'Metric': metric,
                'Joy AI': f"{values['joy']}" + ('%' if metric != 'Cost per Call' and metric != 'Avg Handle Time' else ''),
                'Human': f"{values['human']}" + ('%' if metric != 'Cost per Call' and metric != 'Avg Handle Time' else '')
            })
        
        st.dataframe(pd.DataFrame(performance_data), hide_index=True, use_container_width=True)
        
        st.markdown("### Monthly Trends")
        
        months = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        calls_automated = [12400, 28900, 42500, 58300, 67800, 78450, 89320]
        automation_rate = [94.2, 95.8, 96.5, 97.2, 97.9, 98.3, 98.7]
        
        fig4 = go.Figure()
        
        fig4.add_trace(go.Scatter(
            x=months,
            y=calls_automated,
            name='Calls Automated',
            yaxis='y',
            line=dict(color='#06b6d4', width=3)
        ))
        
        fig4.add_trace(go.Scatter(
            x=months,
            y=automation_rate,
            name='Automation Rate (%)',
            yaxis='y2',
            line=dict(color='#8b5cf6', width=3)
        ))
        
        fig4.update_layout(
            yaxis=dict(title='Calls Automated'),
            yaxis2=dict(title='Automation Rate (%)', overlaying='y', side='right'),
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        st.markdown("### Economic Impact")
        
        impact_data = {
            'Category': ['Labor Cost Savings', 'Efficiency Gains', 'Error Reduction', 'Faster Processing'],
            'Annual Value': ['$18.5M', '$8.2M', '$3.7M', '$5.1M']
        }
        
        st.dataframe(pd.DataFrame(impact_data), hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; margin: 20px 0;">
            <h4 style="margin: 0 0 10px 0; color: #92400e;">💰 Total Annual Impact</h4>
            <p style="font-size: 32px; font-weight: 900; color: #92400e; margin: 0;">$35.5M</p>
            <p style="margin: 8px 0 0 0; color: #78350f; font-size: 14px;">For 100K calls/month volume</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Time Savings by Call Type")
        
        call_types_list = list(CALL_TYPES.keys())
        time_saved = [(CALL_TYPES[ct]['avg_duration'] - CALL_TYPES[ct]['joy_duration']) for ct in call_types_list]
        
        fig5 = go.Figure(data=[go.Bar(
            x=call_types_list,
            y=time_saved,
            marker_color='#3b82f6',
            text=[f"{t:.1f} min" for t in time_saved],
            textposition='auto'
        )])
        
        fig5.update_layout(
            yaxis_title='Time Saved (minutes)',
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig5, use_container_width=True)

with tab4:
    st.markdown("### Joy Technology Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Core Capabilities**")
        
        for capability, description in JOY_CAPABILITIES.items():
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #06b6d4;">
                <p style="margin: 0; font-weight: 700; color: #0e7490;">{capability}</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**Technical Architecture**")
        st.markdown("""
        - ✅ Multi-modal agentic AI architecture
        - ✅ Reinforcement learning (RLHF)
        - ✅ Proprietary healthcare conversation data
        - ✅ Zero-skip validation layer
        - ✅ Context-bound response system
        - ✅ HIPAA + SOC 2 Type II certified
        - ✅ EHR integration (Epic, Cerner)
        - ✅ Real-time transcription + NLU
        """)
    
    with col2:
        st.markdown("**Call Workflow**")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%); padding: 20px; border-radius: 12px;">
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #0c4a6e;">1. Initiation</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #0369a1;">Joy dials payer, introduces purpose</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #0c4a6e;">2. Information Exchange</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #0369a1;">Provides patient data from EHR automatically</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #0c4a6e;">3. Hold Navigation</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #0369a1;">Waits on hold (45+ min capability)</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #0c4a6e;">4. Complex Conversation</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #0369a1;">Handles nuanced back-and-forth</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #0c4a6e;">5. Outcome Capture</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #0369a1;">Documents results, updates systems</p>
            </div>
            <div>
                <p style="margin: 0; font-weight: 700; color: #0c4a6e;">6. Human Escalation</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #0369a1;">Routes complex cases to staff when needed</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Integration Points**")
        st.markdown("""
        - 🏥 EHR systems (Epic, Cerner, Athena)
        - 💳 Payer portals (automated login)
        - 📞 Phone systems (SIP/VoIP)
        - 📊 RCM platforms
        - 📋 Practice management systems
        - 🔔 Staff notification systems
        """)
        
        st.markdown("**Use Cases Automated**")
        
        use_cases = {
            'Use Case': ['Benefits verification', 'Prior authorizations', 'Claims follow-up', 'Appointment scheduling', 'Insurance verification'],
            'Volume': ['High', 'High', 'Very High', 'Very High', 'High']
        }
        
        st.dataframe(pd.DataFrame(use_cases), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #0c4a6e; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #06b6d4; font-weight: 700; margin: 0 0 6px 0;">✓ 98.7% Call Completion</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Autonomous automation</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #06b6d4; font-weight: 700; margin: 0 0 6px 0;">✓ 89% Time Reduction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">3.5 min vs 28 min</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #06b6d4; font-weight: 700; margin: 0 0 6px 0;">✓ Zero-Skip Architecture</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Hallucination-free</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #06b6d4; font-weight: 700; margin: 0 0 6px 0;">✓ $35.5M Annual Savings</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">For 100K calls/month</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for VoiceCare AI</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)