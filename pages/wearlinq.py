"""
Wearlinq - Wireless 6-Lead Cardiac Monitoring Platform
FDA-cleared eWave device with AI arrhythmia detection
Built for Wearlinq by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Wearlinq - Cardiac Monitoring", page_icon="❤️", layout="wide")

# Arrhythmia types detected
ARRHYTHMIA_TYPES = {
    'Atrial Fibrillation (AFib)': {'prevalence': 12.3, 'detection_rate': 94.7, 'severity': 'High'},
    'Bradycardia': {'prevalence': 8.5, 'detection_rate': 97.2, 'severity': 'Medium'},
    'Tachycardia': {'prevalence': 15.2, 'detection_rate': 96.8, 'severity': 'Medium'},
    'Premature Ventricular Contractions': {'prevalence': 22.1, 'detection_rate': 92.3, 'severity': 'Low'},
    'Atrial Flutter': {'prevalence': 4.7, 'detection_rate': 93.5, 'severity': 'High'},
    'Normal Sinus Rhythm': {'prevalence': 37.2, 'detection_rate': 99.1, 'severity': 'None'}
}

# eWave device specs
DEVICE_SPECS = {
    'Leads': '6-lead (vs 1-lead standard)',
    'Battery Life': '5+ days continuous',
    'Data Transmission': 'Near real-time via smartphone',
    'Report Time': '<48 hours to clinician',
    'Weight': 'Lightest wireless 6-lead ECG',
    'Connectivity': 'Pairs with personal phone (no hub)'
}

# Clinical outcomes
CLINICAL_OUTCOMES = {
    'Early Detection': {'improvement': 47, 'baseline': 58, 'wearlinq': 85},
    'Patient Compliance': {'improvement': 68, 'baseline': 42, 'wearlinq': 89},
    'Diagnosis Accuracy': {'improvement': 32, 'baseline': 72, 'wearlinq': 95},
    'Time to Treatment': {'improvement': 55, 'baseline': '14 days', 'wearlinq': '2.3 days'}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">❤️</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Wearlinq</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Wireless 6-Lead Cardiac Monitoring</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">FDA-cleared eWave • Real-time AI detection • 5+ day battery</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["❤️ eWave Monitor", "📊 Arrhythmia Detection", "📈 Clinical Outcomes", "💡 Technology"])

with tab1:
    st.markdown("### eWave - 6-Lead Wireless ECG Monitor")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Patient Profile**")
        
        patient_name = st.text_input("Patient Name", "Robert Martinez")
        age = st.number_input("Age", 18, 100, 67)
        
        st.markdown("**Clinical Indication**")
        
        indication = st.selectbox("Reason for Monitoring",
                                 ["Palpitations", "Syncope/Pre-syncope", "Post-AFib ablation", "Suspected arrhythmia", "Stroke workup"])
        
        duration = st.selectbox("Monitoring Duration", ["24 hours", "48 hours", "72 hours", "5 days", "7 days", "14 days"])
        
        st.markdown("**Medical History**")
        
        conditions = st.multiselect("Cardiac History",
                                    ["Hypertension", "CAD", "Prior MI", "Heart failure", "Valve disease"],
                                    ["Hypertension", "CAD"])
        
        medications = st.text_area("Current Medications", "Metoprolol 50mg, Aspirin 81mg, Atorvastatin 40mg", height=60)
        
        st.markdown("**eWave Device Status**")
        
        device_applied = st.checkbox("Device applied to patient", value=True)
        app_paired = st.checkbox("Paired with smartphone app", value=True)
        baseline_ecg = st.checkbox("Baseline ECG recorded", value=True)
        
        monitor_btn = st.button("❤️ Start Monitoring", type="primary", use_container_width=True)
    
    with col2:
        if monitor_btn:
            st.markdown("**Live ECG Monitoring Dashboard**")
            
            import time
            with st.spinner("Initializing 6-lead ECG acquisition..."):
                time.sleep(1.0)
            
            st.success("✅ eWave monitoring active - Real-time data streaming!")
            
            # Simulate ECG waveform
            t = np.linspace(0, 2, 500)
            ecg_signal = (np.sin(2 * np.pi * 1.2 * t) * 0.3 + 
                         np.sin(2 * np.pi * 18 * t) * 0.05 +
                         np.random.randn(len(t)) * 0.02)
            
            # Add QRS complexes
            for i in [125, 312]:
                ecg_signal[i-10:i+10] += np.concatenate([
                    np.linspace(0, 1.2, 10),
                    np.linspace(1.2, 0, 10)
                ])
            
            fig1 = go.Figure()
            
            fig1.add_trace(go.Scatter(
                x=t,
                y=ecg_signal,
                mode='lines',
                line=dict(color='#dc2626', width=2),
                name='Lead II'
            ))
            
            fig1.update_layout(
                title="Real-Time ECG - Lead II",
                xaxis_title="Time (seconds)",
                yaxis_title="mV",
                height=200,
                showlegend=False,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Current Status</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Heart Rate</p>
                        <p style="font-size: 28px; color: white; font-weight: 900; margin: 0;">72 bpm</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Rhythm</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Normal Sinus</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Battery</p>
                        <p style="font-size: 28px; color: white; font-weight: 900; margin: 0;">87%</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Signal Quality</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Excellent</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Data Captured</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">12.3 hours</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Events Logged</p>
                        <p style="font-size: 28px; color: white; font-weight: 900; margin: 0;">3</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### All 6 Leads Monitoring")
            
            leads_data = pd.DataFrame({
                'Lead': ['Lead I', 'Lead II', 'Lead III', 'Lead aVR', 'Lead aVL', 'Lead aVF'],
                'Signal Quality': ['Excellent', 'Excellent', 'Good', 'Excellent', 'Good', 'Excellent'],
                'Status': ['✅ Active', '✅ Active', '✅ Active', '✅ Active', '✅ Active', '✅ Active']
            })
            
            st.dataframe(leads_data, hide_index=True, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Monitoring Time", "12.3 hrs", "Continuous")
            col2.metric("Data Points", "2.65M", "High res")
            col3.metric("AI Scans", "847", "Real-time")
            col4.metric("Alerts", "0", "Normal")

with tab2:
    st.markdown("### AI-Powered Arrhythmia Detection")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Patients Monitored", "4,847", "Active")
    col2.metric("Arrhythmias Detected", "1,245", "This month")
    col3.metric("AI Detection Rate", "95.3%", "Validated")
    col4.metric("Avg Report Time", "36 hours", "<48hrs SLA")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Detection Capabilities by Arrhythmia")
        
        arrhythmia_data = []
        for arrhythmia, data in ARRHYTHMIA_TYPES.items():
            severity_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢', 'None': '⚪'}[data['severity']]
            arrhythmia_data.append({
                'Arrhythmia': arrhythmia,
                'Detection Rate': f"{data['detection_rate']}%",
                'Prevalence': f"{data['prevalence']}%",
                'Severity': f"{severity_color} {data['severity']}"
            })
        
        st.dataframe(pd.DataFrame(arrhythmia_data), hide_index=True, use_container_width=True)
        
        st.markdown("### 6-Lead vs Single-Lead Comparison")
        
        comparison_data = pd.DataFrame({
            'Metric': ['Arrhythmia Detection', 'False Positives', 'Diagnostic Confidence', 'Clinical Utility'],
            'Single-Lead': ['72%', '18%', 'Moderate', 'Limited'],
            '6-Lead (eWave)': ['95%', '4%', 'High', 'Hospital-grade']
        })
        
        st.dataframe(comparison_data, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### Monthly Detection Trends")
        
        months = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        afib_detected = [145, 178, 223, 267, 312, 358, 402]
        other_detected = [89, 112, 145, 178, 201, 234, 267]
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=months,
            y=afib_detected,
            name='AFib',
            marker_color='#dc2626'
        ))
        
        fig2.add_trace(go.Bar(
            x=months,
            y=other_detected,
            name='Other Arrhythmias',
            marker_color='#f97316'
        ))
        
        fig2.update_layout(
            barmode='stack',
            yaxis_title='Arrhythmias Detected',
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### AI Performance")
        
        ai_metrics = {
            'Metric': ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1 Score'],
            'Value': ['94.7%', '96.2%', '92.8%', '97.3%', '93.7%']
        }
        
        st.dataframe(pd.DataFrame(ai_metrics), hide_index=True, use_container_width=True)

with tab3:
    st.markdown("### Clinical Impact & Outcomes")
    
    # Key impact metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">47%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Early Detection Improvement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ef4444 0%, #f97316 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">68%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Compliance Increase</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">55%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Faster Treatment</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Clinical Outcomes Improvement")
        
        outcomes_data = []
        for outcome, data in CLINICAL_OUTCOMES.items():
            outcomes_data.append({
                'Outcome': outcome,
                'Baseline': data['baseline'] if isinstance(data['baseline'], str) else f"{data['baseline']}%",
                'With eWave': data['wearlinq'] if isinstance(data['wearlinq'], str) else f"{data['wearlinq']}%",
                'Improvement': f"+{data['improvement']}%"
            })
        
        st.dataframe(pd.DataFrame(outcomes_data), hide_index=True, use_container_width=True)
        
        st.markdown("### Lives Saved Impact")
        
        st.markdown("""
        <div style="background: #dcfce7; padding: 20px; border-radius: 12px; border-left: 5px solid #22c55e;">
            <h4 style="margin: 0 0 12px 0; color: #166534;">💚 Potential Impact</h4>
            <p style="margin: 8px 0; color: #15803d; font-size: 15px;">
                <strong>Heart disease kills 1 in 3 Americans</strong>
            </p>
            <p style="margin: 8px 0; color: #15803d; font-size: 15px;">
                Even a <strong>10% reduction</strong> in early heart disease death = <strong>100,000+ lives saved</strong>
            </p>
            <p style="margin: 8px 0; color: #15803d; font-size: 15px;">
                Early arrhythmia detection enables proactive treatment, preventing strokes and sudden cardiac death
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Cost Savings")
        
        cost_data = {
            'Category': ['Prevented strokes', 'Avoided hospitalizations', 'Shorter diagnostic workup', 'Earlier treatment'],
            'Annual Savings': ['$4.8M', '$2.3M', '$1.5M', '$1.9M']
        }
        
        st.dataframe(pd.DataFrame(cost_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### Patient Compliance Over Time")
        
        days = list(range(1, 8))
        compliance = [95, 93, 91, 89, 89, 88, 87]
        
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=days,
            y=compliance,
            mode='lines+markers',
            line=dict(color='#dc2626', width=3),
            marker=dict(size=10)
        ))
        
        fig3.update_layout(
            title="Device Compliance Rate",
            xaxis_title="Day of Monitoring",
            yaxis_title="Compliance (%)",
            height=250,
            yaxis_range=[80, 100]
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("### Detection Distribution")
        
        detection_dist = pd.DataFrame({
            'Finding': ['Normal', 'AFib', 'Other Arrhythmias', 'Inconclusive'],
            'Patients': [1803, 597, 648, 147]
        })
        
        fig4 = px.pie(detection_dist, values='Patients', names='Finding',
                     color_discrete_sequence=['#22c55e', '#dc2626', '#f97316', '#94a3b8'])
        fig4.update_traces(textposition='inside', textinfo='percent+label')
        fig4.update_layout(height=250, showlegend=False)
        
        st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.markdown("### eWave Technology Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Device Specifications**")
        
        for spec, value in DEVICE_SPECS.items():
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #dc2626;">
                <p style="margin: 0; font-weight: 700; color: #991b1b;">{spec}</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{value}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**AI Detection Engine**")
        st.markdown("""
        - ✅ Real-time signal processing
        - ✅ 6-lead multi-dimensional analysis
        - ✅ Deep learning arrhythmia classifier
        - ✅ Cloud-based AI inference
        - ✅ Continuous learning from outcomes
        - ✅ FDA 510(k) cleared algorithms
        - ✅ Cardiologist-validated models
        - ✅ <48 hour report generation
        """)
    
    with col2:
        st.markdown("**System Architecture**")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 20px; border-radius: 12px;">
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #991b1b;">1. eWave Device</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #7f1d1d;">6-lead ECG sensors, 5+ day battery, Bluetooth</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #991b1b;">2. Smartphone App</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #7f1d1d;">Real-time data relay, patient diary, event marking</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #991b1b;">3. Cloud AI Engine</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #7f1d1d;">Multi-lead ECG analysis, arrhythmia detection</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #991b1b;">4. Clinical Review</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #7f1d1d;">Cardiologist validation, <48hr report</p>
            </div>
            <div>
                <p style="margin: 0; font-weight: 700; color: #991b1b;">5. Physician Dashboard</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #7f1d1d;">EHR integration, treatment recommendations</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Key Differentiators**")
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; margin-top: 15px;">
            <h4 style="margin: 0 0 12px 0; color: #92400e;">🏆 Industry-First Innovations</h4>
            <ul style="margin: 0; padding-left: 20px; color: #78350f;">
                <li><strong>First wireless 6-lead continuous ECG</strong> (vs 1-lead standard)</li>
                <li><strong>No separate hub required</strong> - pairs with personal phone</li>
                <li><strong>5+ day battery</strong> - longest in category</li>
                <li><strong>Hospital-grade diagnostics</strong> at home comfort</li>
                <li><strong>AI + cardiologist review</strong> - best of both worlds</li>
                <li><strong>FDA 510(k) cleared</strong> - clinically validated</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Integration**")
        st.markdown("""
        - 🏥 EHR systems (Epic, Cerner)
        - 📱 iOS/Android mobile apps
        - 💻 Physician web portal
        - 📊 Cardiology practice management
        - 🔔 Real-time alert systems
        - 📞 Telemedicine platforms
        """)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #991b1b; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #dc2626; font-weight: 700; margin: 0 0 6px 0;">✓ 6-Lead ECG</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">First wireless continuous</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #dc2626; font-weight: 700; margin: 0 0 6px 0;">✓ 95.3% AI Detection</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">FDA-cleared algorithms</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #dc2626; font-weight: 700; margin: 0 0 6px 0;">✓ 5+ Day Battery</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Continuous monitoring</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #dc2626; font-weight: 700; margin: 0 0 6px 0;">✓ 100K Lives Saved</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Potential annual impact</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Wearlinq</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)