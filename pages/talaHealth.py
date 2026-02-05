"""
Tala Health - AI-Powered Virtual Care Platform
24/7 patient access with AI agents and licensed clinicians
Built for Tala Health by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Tala Health - AI Virtual Care", page_icon="🏥", layout="wide")

# Care journey stages
CARE_STAGES = {
    'Intake': {'time': '2 min', 'automation': 98.5, 'status': 'AI'},
    'Triage': {'time': '3 min', 'automation': 95.2, 'status': 'AI + Clinician'},
    'Diagnosis': {'time': '8 min', 'automation': 87.3, 'status': 'Clinician'},
    'Care Plan': {'time': '5 min', 'automation': 92.1, 'status': 'AI + Clinician'},
    'Coordination': {'time': '6 min', 'automation': 96.8, 'status': 'AI'}
}

# AI agent types
AI_AGENTS = {
    'Clinical Agent': {'tasks': ['Symptom analysis', 'SOAP note generation', 'Care planning'], 'accuracy': 94.2},
    'Triage Agent': {'tasks': ['Urgency assessment', 'Provider matching', 'Escalation'], 'accuracy': 96.5},
    'Coordination Agent': {'tasks': ['Scheduling', 'Insurance verification', 'Referrals'], 'accuracy': 98.1},
    'Documentation Agent': {'tasks': ['Chart completion', 'Coding', 'Billing'], 'accuracy': 97.3}
}

# Performance metrics
PERFORMANCE_METRICS = {
    'Time to Care': {'traditional': '7-14 days', 'tala': '4 hours', 'improvement': '95%'},
    'Patient Satisfaction': {'traditional': '3.2/5', 'tala': '4.8/5', 'improvement': '50%'},
    'Cost per Visit': {'traditional': '$185', 'tala': '$42', 'improvement': '77%'},
    'First-Contact Resolution': {'traditional': '45%', 'tala': '82%', 'improvement': '82%'}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🏥</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Tala Health</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">AI-Powered Virtual Care</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">24/7 access • 4-hour care resolution • AI + clinicians</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Care Navigator", "📋 Patient Journey", "📊 Platform Analytics", "💡 Multi-Agent System"])

with tab1:
    st.markdown("### 24/7 AI-Powered Care Navigation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Start Your Care Journey**")
        
        patient_name = st.text_input("Patient Name", "Michael Chen")
        age = st.number_input("Age", 18, 100, 34)
        
        st.markdown("**Tell Us Your Symptoms**")
        
        chief_complaint = st.text_area(
            "What brings you in today?",
            "I've had a persistent cough for 5 days with some chest tightness. No fever but feeling tired.",
            height=80
        )
        
        duration = st.selectbox("How long have you had these symptoms?", 
                               ["1-2 days", "3-5 days", "1-2 weeks", "Longer"])
        
        severity = st.select_slider("Pain/Discomfort Level", 
                                    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                    value=4)
        
        st.markdown("**Additional Information**")
        
        medical_history = st.multiselect("Any medical conditions?",
                                        ["None", "Asthma", "Diabetes", "High BP", "Heart disease"],
                                        ["Asthma"])
        
        allergies = st.text_input("Allergies", "Penicillin")
        
        insurance = st.selectbox("Insurance", ["UnitedHealthcare", "Aetna", "BCBS", "Cigna", "Self-pay"])
        
        start_btn = st.button("🤖 Start AI Assessment", type="primary", use_container_width=True)
    
    with col2:
        if start_btn:
            st.markdown("**AI Care Navigator - Real-Time Processing**")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            stages = [
                ("Analyzing symptoms with clinical AI...", 0.2),
                ("Running differential diagnosis...", 0.4),
                ("Checking insurance coverage...", 0.6),
                ("Generating SOAP note...", 0.8),
                ("Matching with clinician...", 1.0)
            ]
            
            for stage, progress in stages:
                status_text.text(stage)
                progress_bar.progress(progress)
                time.sleep(0.4)
            
            st.success("✅ AI assessment complete - Clinician review ready!")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">AI-Generated Assessment</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Primary Concern</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Respiratory Infection</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Urgency Level</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Routine (24-48hrs)</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Matched Provider</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Dr. Sarah Johnson, NP</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Next Available</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Today, 2:30 PM</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**AI-Generated SOAP Note**")
            
            st.markdown("""
            <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; border-left: 5px solid #3b82f6;">
                <p style="margin: 0 0 10px 0;"><strong>Subjective:</strong> 34yo M with 5-day history of productive cough with chest tightness. Denies fever. Reports fatigue. PMH significant for asthma. No recent exacerbations. Allergy: Penicillin.</p>
                <p style="margin: 10px 0;"><strong>Objective:</strong> Awaiting vitals from video visit. Preliminary screening negative for respiratory distress indicators.</p>
                <p style="margin: 10px 0;"><strong>Assessment:</strong> Likely acute bronchitis vs upper respiratory infection. Consider asthma exacerbation given PMH. Low probability of bacterial pneumonia based on symptom profile.</p>
                <p style="margin: 10px 0 0 0;"><strong>Plan:</strong> Video consultation with NP. Consider CXR if exam suggests lower respiratory involvement. Albuterol PRN if wheezing present. Avoid PCN-containing antibiotics if needed. Follow-up in 3-5 days or sooner if worsening.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Differential Diagnosis (AI-Generated)**")
            
            differentials = pd.DataFrame({
                'Diagnosis': ['Acute Bronchitis', 'Upper Respiratory Infection', 'Asthma Exacerbation', 'Pneumonia', 'COVID-19'],
                'Probability': [68.2, 52.1, 34.5, 12.3, 8.9],
                'Confidence': ['High', 'High', 'Medium', 'Low', 'Low']
            })
            
            st.dataframe(differentials, hide_index=True, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("AI Processing", "2.3 min", "vs 45 min manual")
            col2.metric("SOAP Accuracy", "94.2%", "Validated")
            col3.metric("Insurance", "Verified", "✓ Covered")
            col4.metric("Time to Care", "4 hours", "vs 7 days")

with tab2:
    st.markdown("### Complete Patient Journey")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Active Patients", "18,450", "+2,340")
    col2.metric("Avg Resolution", "4.2 hours", "-93%")
    col3.metric("Today's Encounters", "1,847", "Real-time")
    col4.metric("AI Automation", "95.3%", "High")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Care Journey Stages")
        
        journey_data = []
        for stage, data in CARE_STAGES.items():
            journey_data.append({
                'Stage': stage,
                'Avg Time': data['time'],
                'Automation': f"{data['automation']}%",
                'Handled By': data['status']
            })
        
        st.dataframe(pd.DataFrame(journey_data), hide_index=True, use_container_width=True)
        
        st.markdown("### Patient Journey Flow")
        
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #3b82f6;">
            <div style="margin-bottom: 15px;">
                <h4 style="margin: 0 0 8px 0; color: #3b82f6;">1. 24/7 AI Intake (2 min)</h4>
                <p style="margin: 0; color: #666; font-size: 14px;">Patient describes symptoms via chat/mobile → AI conducts dynamic questioning → Pulls authorized medical history</p>
            </div>
            <div style="margin-bottom: 15px;">
                <h4 style="margin: 0 0 8px 0; color: #3b82f6;">2. AI Triage (3 min)</h4>
                <p style="margin: 0; color: #666; font-size: 14px;">Symptom checker runs differential diagnosis → Urgency assessment → SOAP note generation</p>
            </div>
            <div style="margin-bottom: 15px;">
                <h4 style="margin: 0 0 8px 0; color: #8b5cf6;">3. Clinician Review (8 min)</h4>
                <p style="margin: 0; color: #666; font-size: 14px;">Licensed NP/physician reviews AI-generated chart → Video/secure chat with patient → Clinical decision</p>
            </div>
            <div style="margin-bottom: 15px;">
                <h4 style="margin: 0 0 8px 0; color: #3b82f6;">4. Care Coordination (6 min)</h4>
                <p style="margin: 0; color: #666; font-size: 14px;">AI handles lab ordering, imaging scheduling, insurance pre-auth, specialist referrals</p>
            </div>
            <div>
                <h4 style="margin: 0 0 8px 0; color: #3b82f6;">5. Follow-Up (Automated)</h4>
                <p style="margin: 0; color: #666; font-size: 14px;">AI flags abnormal results → Plain-language explanations → Escalates to specialists when needed</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Time to Resolution")
        
        comparison_data = pd.DataFrame({
            'Stage': ['Intake', 'Triage', 'Diagnosis', 'Coordination', 'Total'],
            'Traditional': [15, 45, 120, 180, 360],
            'Tala Health': [2, 3, 8, 6, 19]
        })
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Bar(
            x=comparison_data['Stage'],
            y=comparison_data['Traditional'],
            name='Traditional',
            marker_color='#94a3b8'
        ))
        
        fig1.add_trace(go.Bar(
            x=comparison_data['Stage'],
            y=comparison_data['Tala Health'],
            name='Tala Health',
            marker_color='#3b82f6'
        ))
        
        fig1.update_layout(
            barmode='group',
            yaxis_title='Time (minutes)',
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("### Daily Patient Volume")
        
        hours = list(range(0, 24))
        volume = [45, 28, 18, 12, 15, 32, 78, 156, 234, 287, 245, 198, 176, 189, 201, 223, 198, 167, 145, 123, 98, 87, 76, 58]
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=hours,
            y=volume,
            mode='lines+markers',
            line=dict(color='#8b5cf6', width=3),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.1)'
        ))
        
        fig2.update_layout(
            xaxis_title='Hour of Day',
            yaxis_title='Patient Encounters',
            height=250,
            showlegend=False
        )
        
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Platform Performance Analytics")
    
    # Key impact metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">95%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Faster Time to Care</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">77%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Cost Reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ec4899 0%, #f97316 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">82%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">First-Contact Resolution</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Performance vs Traditional Care")
        
        performance_data = []
        for metric, data in PERFORMANCE_METRICS.items():
            performance_data.append({
                'Metric': metric,
                'Traditional': data['traditional'],
                'Tala Health': data['tala'],
                'Improvement': data['improvement']
            })
        
        st.dataframe(pd.DataFrame(performance_data), hide_index=True, use_container_width=True)
        
        st.markdown("### Monthly Outcomes")
        
        months = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        encounters = [3400, 5600, 8900, 12400, 15800, 18450]
        resolution_rate = [75, 78, 79, 81, 81, 82]
        
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=months,
            y=encounters,
            name='Total Encounters',
            yaxis='y',
            line=dict(color='#3b82f6', width=3)
        ))
        
        fig3.add_trace(go.Scatter(
            x=months,
            y=resolution_rate,
            name='Resolution Rate (%)',
            yaxis='y2',
            line=dict(color='#8b5cf6', width=3)
        ))
        
        fig3.update_layout(
            yaxis=dict(title='Encounters'),
            yaxis2=dict(title='Resolution Rate (%)', overlaying='y', side='right'),
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("### AI Automation Rate by Stage")
        
        stages = list(CARE_STAGES.keys())
        automation_rates = [CARE_STAGES[s]['automation'] for s in stages]
        
        fig4 = go.Figure(data=[go.Bar(
            x=stages,
            y=automation_rates,
            marker_color='#3b82f6',
            text=[f"{rate}%" for rate in automation_rates],
            textposition='auto'
        )])
        
        fig4.update_layout(
            yaxis=dict(range=[85, 100], title='Automation Rate (%)'),
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("### Economic Impact")
        
        impact_data = {
            'Metric': ['Cost per Visit', 'Monthly Patients', 'Cost Savings', 'Provider Efficiency'],
            'Value': ['$42', '18,450', '$2.64M/month', '5.2x improvement']
        }
        
        st.dataframe(pd.DataFrame(impact_data), hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; margin-top: 15px;">
            <h4 style="margin: 0 0 10px 0; color: #92400e;">💰 Annual Impact</h4>
            <p style="font-size: 28px; font-weight: 900; color: #92400e; margin: 0;">$31.7M</p>
            <p style="margin: 8px 0 0 0; color: #78350f; font-size: 14px;">Total cost savings vs traditional care</p>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("### Multi-Agent AI Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**AI Agent Capabilities**")
        
        agent_data = []
        for agent, data in AI_AGENTS.items():
            agent_data.append({
                'Agent Type': agent,
                'Primary Tasks': ', '.join(data['tasks'][:2]),
                'Accuracy': f"{data['accuracy']}%"
            })
        
        st.dataframe(pd.DataFrame(agent_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Clinical AI Agent**")
        st.markdown("""
        - ✅ LLM-based symptom analysis
        - ✅ Dynamic questioning engine
        - ✅ Medical history integration
        - ✅ Differential diagnosis generation
        - ✅ SOAP note creation (structured)
        - ✅ Fine-tuned on clinical datasets
        - ✅ Real-time learning from outcomes
        """)
        
        st.markdown("**Coordination AI Agent**")
        st.markdown("""
        - ✅ Insurance eligibility verification
        - ✅ Prior authorization submission
        - ✅ Lab/imaging scheduling
        - ✅ Specialist referral management
        - ✅ EHR integration
        - ✅ Billing/coding automation
        """)
    
    with col2:
        st.markdown("**Technology Stack**")
        
        tech_stack = {
            'Component': ['Intake AI', 'Clinical AI', 'Coordination', 'Infrastructure', 'Compliance'],
            'Technology': ['LLM + NLP', 'Fine-tuned models', 'Multi-agent system', 'Cloud-native', 'HIPAA + FDA audit trails']
        }
        
        st.dataframe(pd.DataFrame(tech_stack), hide_index=True, use_container_width=True)
        
        st.markdown("**Multi-Agent Workflow**")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #e9d5ff 100%); padding: 20px; border-radius: 12px;">
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #1e40af;">1. Conversational Intake Agent</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #1e3a8a;">Captures symptoms via chat → Dynamic follow-up questions</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #7c3aed;">2. Clinical Intelligence Agent</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #5b21b6;">Runs differential diagnosis → Generates SOAP notes</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #1e40af;">3. Triage & Matching Agent</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #1e3a8a;">Urgency assessment → Clinician matching → Scheduling</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #7c3aed;">4. Care Coordination Agent</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #5b21b6;">Insurance verification → Lab orders → Specialist referrals</p>
            </div>
            <div>
                <p style="margin: 0; font-weight: 700; color: #1e40af;">5. Results & Follow-up Agent</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #1e3a8a;">Flags abnormal findings → Patient explanations → Escalation</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Integration Points**")
        st.markdown("""
        - 🏥 EHR systems (Epic, Cerner, Athena)
        - 💳 Insurance verification APIs
        - 🔬 Lab ordering (Quest, LabCorp)
        - 📷 Imaging scheduling
        - 📊 Claims processing systems
        - 📱 Patient mobile apps
        """)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #dbeafe 0%, #e9d5ff 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #1e40af; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #3b82f6; font-weight: 700; margin: 0 0 6px 0;">✓ 24/7 AI Access</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Conversational intake anytime</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #3b82f6; font-weight: 700; margin: 0 0 6px 0;">✓ 4-Hour Resolution</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">vs 7-14 days traditional</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #3b82f6; font-weight: 700; margin: 0 0 6px 0;">✓ 95.3% AI Automation</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Multi-agent coordination</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #3b82f6; font-weight: 700; margin: 0 0 6px 0;">✓ $31.7M Annual Savings</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">77% cost reduction</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Tala Health</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)