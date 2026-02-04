"""
Kaigo Health - Remote Care Platform
Real-time patient monitoring with intelligent alerts and care coordination
Built for Kaigo Health by Anju Vilashini Nandhakumar
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
st.set_page_config(page_title="Kaigo Health - Remote Care", page_icon="🏥", layout="wide")

# Generate synthetic patient data
def generate_patient_data():
    patients = [
        {"id": 1, "name": "Margaret Chen", "age": 78, "condition": "Heart Failure", "risk": "High"},
        {"id": 2, "name": "Robert Johnson", "age": 65, "condition": "COPD", "risk": "Medium"},
        {"id": 3, "name": "Patricia Williams", "age": 82, "condition": "Diabetes", "risk": "Low"},
        {"id": 4, "name": "James Miller", "age": 71, "condition": "Hypertension", "risk": "Medium"},
        {"id": 5, "name": "Linda Davis", "age": 69, "condition": "Post-Surgery", "risk": "Low"},
    ]
    return pd.DataFrame(patients)

def generate_vitals(patient_id, days=7):
    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
    
    # Base values with some variation
    if patient_id == 1:  # High risk patient
        hr_base = 85 + np.random.randn(len(dates)) * 8
        bp_sys = 145 + np.random.randn(len(dates)) * 12
        bp_dia = 88 + np.random.randn(len(dates)) * 8
        spo2 = 94 + np.random.randn(len(dates)) * 2
    else:
        hr_base = 72 + np.random.randn(len(dates)) * 5
        bp_sys = 125 + np.random.randn(len(dates)) * 8
        bp_dia = 78 + np.random.randn(len(dates)) * 5
        spo2 = 97 + np.random.randn(len(dates)) * 1.5
    
    return pd.DataFrame({
        'timestamp': dates,
        'heart_rate': hr_base.clip(60, 120),
        'bp_systolic': bp_sys.clip(100, 180),
        'bp_diastolic': bp_dia.clip(60, 110),
        'spo2': spo2.clip(90, 100),
        'temperature': (98.6 + np.random.randn(len(dates)) * 0.5).clip(97, 100)
    })

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🏥</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Kaigo Health</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Remote Care Platform</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Real-time monitoring • Care coordination • Intelligent alerts</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Patient Dashboard", "🚨 Alerts & Monitoring", "👥 Care Coordination", "📈 Analytics"])

with tab1:
    st.markdown("### Remote Patient Monitoring Dashboard")
    
    patients_df = generate_patient_data()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">145</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Active Patients</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">8</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Active Alerts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">94.2%</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Compliance Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">12.3</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Avg Days to Recovery</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Patient list with details
    st.markdown("### Current Patient Status")
    
    selected_patient = st.selectbox("Select Patient to Monitor", 
                                   patients_df['name'].tolist(), 
                                   key="patient_select")
    
    patient_info = patients_df[patients_df['name'] == selected_patient].iloc[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 25px; border-radius: 12px; border-left: 5px solid #667eea;">
            <h3 style="margin: 0 0 15px 0;">{patient_info['name']}</h3>
            <p style="margin: 8px 0;"><strong>Age:</strong> {patient_info['age']}</p>
            <p style="margin: 8px 0;"><strong>Condition:</strong> {patient_info['condition']}</p>
            <p style="margin: 8px 0;"><strong>Risk Level:</strong> <span style="color: {'#e74c3c' if patient_info['risk'] == 'High' else '#f39c12' if patient_info['risk'] == 'Medium' else '#27ae60'}; font-weight: 600;">{patient_info['risk']}</span></p>
            <p style="margin: 8px 0;"><strong>Last Check-in:</strong> 2 hours ago</p>
            <p style="margin: 8px 0;"><strong>Next Appointment:</strong> Feb 8, 2026</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Quick Actions")
        if st.button("📞 Schedule Call", use_container_width=True):
            st.success("Call scheduled with care team")
        if st.button("💊 Review Medications", use_container_width=True):
            st.info("Medication review opened")
        if st.button("📋 Update Care Plan", use_container_width=True):
            st.info("Care plan editor opened")
    
    with col2:
        vitals_data = generate_vitals(patient_info['id'])
        
        # Latest vitals
        latest = vitals_data.iloc[-1]
        
        st.markdown("### Current Vital Signs")
        
        vital_cols = st.columns(4)
        
        with vital_cols[0]:
            st.metric("Heart Rate", f"{int(latest['heart_rate'])} bpm", 
                     f"{int(latest['heart_rate'] - vitals_data.iloc[-24]['heart_rate'])} vs 24h ago")
        
        with vital_cols[1]:
            st.metric("Blood Pressure", f"{int(latest['bp_systolic'])}/{int(latest['bp_diastolic'])}", 
                     "Stable")
        
        with vital_cols[2]:
            st.metric("SpO2", f"{latest['spo2']:.1f}%", 
                     f"{latest['spo2'] - vitals_data.iloc[-24]['spo2']:.1f}% vs 24h ago")
        
        with vital_cols[3]:
            st.metric("Temperature", f"{latest['temperature']:.1f}°F", 
                     "Normal")
        
        # Vital trends
        st.markdown("### 7-Day Vital Trends")
        
        fig = go.Figure()
        
        # Resample to daily for cleaner view
        daily_vitals = vitals_data.set_index('timestamp').resample('6H').mean().reset_index()
        
        fig.add_trace(go.Scatter(
            x=daily_vitals['timestamp'], 
            y=daily_vitals['heart_rate'],
            name="Heart Rate",
            line=dict(color='#667eea', width=3),
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title="Heart Rate Trend (Last 7 Days)",
            xaxis_title="Date",
            yaxis_title="Heart Rate (bpm)",
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Intelligent Alert System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Active Alerts")
        
        alerts = [
            {
                "patient": "Margaret Chen",
                "alert": "Heart Rate Elevated",
                "severity": "High",
                "time": "15 min ago",
                "value": "92 bpm (baseline: 75 bpm)",
                "action": "Care team notified"
            },
            {
                "patient": "Robert Johnson",
                "alert": "Missed Medication",
                "severity": "Medium",
                "time": "1 hour ago",
                "value": "Blood pressure medication",
                "action": "Patient reminder sent"
            },
            {
                "patient": "Margaret Chen",
                "alert": "Low SpO2 Detection",
                "severity": "High",
                "time": "3 hours ago",
                "value": "93.2% (threshold: 95%)",
                "action": "Resolved - Patient used oxygen"
            },
            {
                "patient": "James Miller",
                "alert": "Appointment Due",
                "severity": "Low",
                "time": "5 hours ago",
                "value": "Follow-up scheduled tomorrow",
                "action": "Confirmation sent"
            }
        ]
        
        for alert in alerts:
            severity_color = {
                "High": "#e74c3c",
                "Medium": "#f39c12",
                "Low": "#3498db"
            }[alert['severity']]
            
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 12px; margin-bottom: 15px; border-left: 5px solid {severity_color};">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <h4 style="margin: 0 0 10px 0;">{alert['alert']}</h4>
                        <p style="margin: 5px 0; color: #666;"><strong>Patient:</strong> {alert['patient']}</p>
                        <p style="margin: 5px 0; color: #666;"><strong>Details:</strong> {alert['value']}</p>
                        <p style="margin: 5px 0; color: #666;"><strong>Action:</strong> {alert['action']}</p>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: {severity_color}; color: white; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">{alert['severity']}</span>
                        <p style="margin: 10px 0 0 0; color: #999; font-size: 13px;">{alert['time']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Alert Configuration")
        
        alert_col1, alert_col2 = st.columns(2)
        
        with alert_col1:
            st.number_input("Heart Rate Threshold (bpm)", 60, 120, 85)
            st.number_input("SpO2 Threshold (%)", 85, 100, 95)
        
        with alert_col2:
            st.number_input("Systolic BP Threshold", 100, 180, 140)
            st.number_input("Temperature Threshold (°F)", 97, 102, 99)
        
        if st.button("💾 Save Alert Settings", type="primary"):
            st.success("Alert thresholds updated successfully!")
    
    with col2:
        st.markdown("### Alert Statistics")
        
        alert_stats = pd.DataFrame({
            'Category': ['Vital Signs', 'Medication', 'Appointments', 'Other'],
            'Count': [12, 8, 5, 3]
        })
        
        fig = px.pie(alert_stats, values='Count', names='Category',
                     color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#43e97b'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=300, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; margin-top: 20px;">
            <h4 style="margin: 0 0 15px 0;">Response Time</h4>
            <p style="margin: 8px 0;"><strong>Average:</strong> 4.2 minutes</p>
            <p style="margin: 8px 0;"><strong>Critical:</strong> 1.8 minutes</p>
            <p style="margin: 8px 0;"><strong>Resolution:</strong> 18.5 minutes</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Care Team Coordination")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Care Team Members")
        
        team = [
            {"name": "Dr. Sarah Johnson", "role": "Primary Physician", "status": "Online"},
            {"name": "Emily Chen, RN", "role": "Care Coordinator", "status": "Online"},
            {"name": "Dr. Michael Lee", "role": "Cardiologist", "status": "Offline"},
            {"name": "Anna Williams, NP", "role": "Nurse Practitioner", "status": "Online"},
        ]
        
        for member in team:
            status_color = "#27ae60" if member['status'] == "Online" else "#95a5a6"
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #e0e0e0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0;">{member['name']}</h4>
                        <p style="margin: 5px 0 0 0; color: #666;">{member['role']}</p>
                    </div>
                    <span style="background: {status_color}; color: white; padding: 4px 10px; border-radius: 15px; font-size: 11px;">{member['status']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Recent Care Notes")
        
        notes = [
            {"author": "Dr. Sarah Johnson", "time": "2 hours ago", "note": "Patient reports improvement in shortness of breath. Continue current medication regimen."},
            {"author": "Emily Chen, RN", "time": "5 hours ago", "note": "Completed daily check-in. All vitals within normal range. Patient adherent to care plan."},
            {"author": "Anna Williams, NP", "time": "1 day ago", "note": "Medication review completed. Adjusted dosage per cardiologist recommendation."}
        ]
        
        for note in notes:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <strong>{note['author']}</strong>
                    <span style="color: #999; font-size: 13px;">{note['time']}</span>
                </div>
                <p style="margin: 0; color: #555;">{note['note']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Add Care Note")
        
        patient_select = st.selectbox("Patient", patients_df['name'].tolist(), key="note_patient")
        note_text = st.text_area("Note", placeholder="Enter care note...", height=100)
        
        if st.button("📝 Add Note", type="primary"):
            st.success("Care note added successfully!")
        
        st.markdown("---")
        
        st.markdown("### Task Management")
        
        tasks = [
            {"task": "Review medication for Margaret Chen", "due": "Today", "priority": "High"},
            {"task": "Schedule follow-up for Robert Johnson", "due": "Tomorrow", "priority": "Medium"},
            {"task": "Update care plan for Patricia Williams", "due": "Feb 5", "priority": "Low"}
        ]
        
        for task in tasks:
            priority_color = {
                "High": "#e74c3c",
                "Medium": "#f39c12",
                "Low": "#3498db"
            }[task['priority']]
            
            col_check, col_task = st.columns([0.1, 0.9])
            with col_check:
                st.checkbox("", key=f"task_{task['task']}")
            with col_task:
                st.markdown(f"""
                <div style="padding: 10px; border-left: 3px solid {priority_color}; margin-bottom: 10px;">
                    <p style="margin: 0; font-weight: 600;">{task['task']}</p>
                    <p style="margin: 5px 0 0 0; color: #666; font-size: 13px;">Due: {task['due']} • Priority: {task['priority']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### Communication")
        
        if st.button("📧 Send Team Update", use_container_width=True):
            st.success("Update sent to care team")
        if st.button("📱 Message Patient", use_container_width=True):
            st.success("Message sent to patient")
        if st.button("🔔 Create Alert", use_container_width=True):
            st.info("Alert created")

with tab4:
    st.markdown("### Remote Care Analytics")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">32%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Hospital Readmission Reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">$2,450</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Cost Savings Per Patient/Month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">4.8/5</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Patient Satisfaction</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Patient outcomes over time
        st.markdown("### Patient Outcomes Over Time")
        
        outcomes_data = pd.DataFrame({
            'Month': ['Oct', 'Nov', 'Dec', 'Jan', 'Feb'],
            'Readmissions': [28, 24, 21, 19, 18],
            'ER Visits': [45, 38, 32, 28, 25]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=outcomes_data['Month'], y=outcomes_data['Readmissions'],
                            name='Readmissions', marker_color='#667eea'))
        fig.add_trace(go.Bar(x=outcomes_data['Month'], y=outcomes_data['ER Visits'],
                            name='ER Visits', marker_color='#764ba2'))
        
        fig.update_layout(
            barmode='group',
            height=300,
            xaxis_title="Month",
            yaxis_title="Count",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monitoring compliance
        st.markdown("### Device Compliance Rate")
        
        compliance_data = pd.DataFrame({
            'Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
            'Compliance': [88, 91, 94, 94.2]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=compliance_data['Week'],
            y=compliance_data['Compliance'],
            mode='lines+markers',
            line=dict(color='#43e97b', width=4),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            height=250,
            xaxis_title="Week",
            yaxis_title="Compliance Rate (%)",
            yaxis_range=[80, 100]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Alert response metrics
        st.markdown("### Alert Response Metrics")
        
        response_data = pd.DataFrame({
            'Severity': ['Critical', 'High', 'Medium', 'Low'],
            'Avg Response (min)': [1.8, 4.2, 8.5, 15.3],
            'Count': [5, 12, 28, 18]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=response_data['Severity'],
            y=response_data['Avg Response (min)'],
            marker_color=['#e74c3c', '#f39c12', '#3498db', '#95a5a6'],
            text=response_data['Avg Response (min)'],
            texttemplate='%{text:.1f} min',
            textposition='outside'
        ))
        
        fig.update_layout(
            height=300,
            xaxis_title="Alert Severity",
            yaxis_title="Average Response Time (min)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ROI metrics
        st.markdown("### ROI Impact")
        
        st.markdown("""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 12px;">
            <h4 style="margin: 0 0 15px 0;">Monthly Savings Breakdown</h4>
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                    <span>Reduced ER visits:</span>
                    <strong>$185K</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                    <span>Prevented readmissions:</span>
                    <strong>$220K</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                    <span>Shorter hospital stays:</span>
                    <strong>$95K</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                    <span>Care efficiency:</span>
                    <strong>$155K</strong>
                </div>
                <hr style="margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                    <span><strong>Total Monthly Savings:</strong></span>
                    <strong style="color: #27ae60; font-size: 18px;">$655K</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #ddd6fe 0%, #c4b5fd 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #5b21b6; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ 32% Readmission Reduction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">vs industry average</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ $2,450 Savings/Patient</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Monthly cost reduction</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ 94.2% Compliance</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Device usage rate</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ 4.8/5 Satisfaction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Patient rating</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Kaigo Health</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashini Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)