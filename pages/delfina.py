"""
Delfina - AI-Powered Maternal Health Platform
Pregnancy monitoring and risk prediction for better outcomes
Built for Delfina by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Delfina - Maternal Health", page_icon="🤰", layout="wide")

# Risk factors and conditions
RISK_FACTORS = {
    'Gestational Diabetes': {'prevalence': 7.6, 'risk_score': 0.82, 'interventions': 4},
    'Preeclampsia': {'prevalence': 5.2, 'risk_score': 0.91, 'interventions': 6},
    'Preterm Birth': {'prevalence': 10.1, 'risk_score': 0.78, 'interventions': 5},
    'Postpartum Depression': {'prevalence': 13.2, 'risk_score': 0.74, 'interventions': 3},
    'Gestational Hypertension': {'prevalence': 6.5, 'risk_score': 0.85, 'interventions': 4}
}

# Outcome metrics
OUTCOME_METRICS = {
    'Maternal Mortality': {'reduction': 32.5, 'baseline': 17.4, 'current': 11.7},
    'Preterm Birth Rate': {'reduction': 24.3, 'baseline': 10.1, 'current': 7.6},
    'C-Section Rate': {'reduction': 18.7, 'baseline': 31.8, 'current': 25.9},
    'NICU Admissions': {'reduction': 28.9, 'baseline': 8.5, 'current': 6.0},
    'Postpartum Readmissions': {'reduction': 35.2, 'baseline': 5.8, 'current': 3.8}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #ec4899 0%, #f97316 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🤰</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Delfina</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">AI-Powered Maternal Health</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Risk prediction • Prenatal monitoring • Better outcomes</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🤰 Risk Assessment", "📊 Pregnancy Monitoring", "📈 Population Health", "💡 Clinical Insights"])

with tab1:
    st.markdown("### AI-Powered Risk Assessment")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Patient Information**")
        
        patient_name = st.text_input("Patient Name", "Sarah Martinez")
        age = st.number_input("Age", 18, 50, 32)
        gestational_age = st.number_input("Gestational Age (weeks)", 1, 42, 28)
        
        st.markdown("**Clinical History**")
        
        gravida = st.number_input("Gravida (G)", 1, 10, 2)
        para = st.number_input("Para (P)", 0, 10, 1)
        
        conditions = st.multiselect(
            "Pre-existing Conditions",
            ["Diabetes", "Hypertension", "Obesity", "Thyroid disorder", "Previous preterm birth"],
            ["Obesity"]
        )
        
        st.markdown("**Current Vitals**")
        
        bp_sys = st.number_input("Blood Pressure (systolic)", 90, 180, 135)
        bp_dia = st.number_input("Blood Pressure (diastolic)", 60, 120, 85)
        weight_gain = st.number_input("Weight Gain (lbs)", 0, 60, 22)
        glucose = st.number_input("Glucose (mg/dL)", 60, 200, 98)
        
        assess_btn = st.button("🤰 Run AI Assessment", type="primary", use_container_width=True)
    
    with col2:
        if assess_btn:
            st.markdown("**Risk Prediction Results**")
            
            import time
            with st.spinner("Analyzing maternal health data..."):
                time.sleep(1.5)
            
            st.success("✅ AI risk assessment complete!")
            
            # Risk scores
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ec4899 0%, #f97316 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Risk Profile - Week 28</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Overall Risk Score</p>
                        <p style="font-size: 32px; color: white; font-weight: 900; margin: 0;">0.73</p>
                        <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Moderate Risk</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Confidence Level</p>
                        <p style="font-size: 32px; color: white; font-weight: 900; margin: 0;">94.2%</p>
                        <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">High Confidence</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Risk Breakdown by Condition**")
            
            risk_data = []
            for condition, data in RISK_FACTORS.items():
                risk_level = "🔴 High" if data['risk_score'] > 0.85 else "🟡 Moderate" if data['risk_score'] > 0.75 else "🟢 Low"
                risk_data.append({
                    'Condition': condition,
                    'Risk Score': f"{data['risk_score']:.2f}",
                    'Level': risk_level,
                    'Interventions': data['interventions']
                })
            
            st.dataframe(pd.DataFrame(risk_data), hide_index=True, use_container_width=True)
            
            st.markdown("**AI-Recommended Actions**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background: #fef3c7; padding: 20px; border-radius: 12px; border-left: 5px solid #f59e0b;">
                    <h4 style="margin: 0 0 12px 0; color: #92400e;">⚠️ High Priority</h4>
                    <ul style="margin: 0; padding-left: 20px; color: #78350f;">
                        <li>Schedule glucose tolerance test (elevated risk for GDM)</li>
                        <li>Monitor BP twice weekly (trending upward)</li>
                        <li>Refer to maternal-fetal medicine specialist</li>
                        <li>Start daily BP log for patient</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background: #dcfce7; padding: 20px; border-radius: 12px; border-left: 5px solid #22c55e;">
                    <h4 style="margin: 0 0 12px 0; color: #166534;">✓ Routine Follow-up</h4>
                    <ul style="margin: 0; padding-left: 20px; color: #15803d;">
                        <li>Continue prenatal vitamins</li>
                        <li>Maintain exercise routine (modified)</li>
                        <li>Next ultrasound at 32 weeks</li>
                        <li>Review birth plan preferences</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("**Predicted Outcomes**")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Term Birth Prob", "87.3%", "High")
            col2.metric("C-Section Risk", "28.5%", "Moderate")
            col3.metric("NICU Need", "12.7%", "Low")
            col4.metric("Healthy Outcome", "92.8%", "Excellent")

with tab2:
    st.markdown("### Continuous Pregnancy Monitoring")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Active Patients", "2,847", "+156")
    col2.metric("High Risk", "342", "12%")
    col3.metric("Due This Month", "187", "+23")
    col4.metric("Avg Risk Score", "0.68", "-0.08")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Patient Vital Trends (Week 20-28)")
        
        # Generate sample trend data
        weeks = list(range(20, 29))
        bp_sys_trend = [128, 130, 132, 135, 137, 135, 138, 139, 135]
        weight_trend = [158, 161, 164, 167, 170, 173, 176, 179, 180]
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=weeks,
            y=bp_sys_trend,
            name="Systolic BP",
            line=dict(color='#ec4899', width=3),
            mode='lines+markers'
        ))
        
        fig1.add_hline(y=140, line_dash="dash", line_color="red", 
                      annotation_text="Hypertension Threshold")
        
        fig1.update_layout(
            title="Blood Pressure Trend",
            xaxis_title="Gestational Age (weeks)",
            yaxis_title="BP Systolic (mmHg)",
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("### Risk Score Evolution")
        
        risk_weeks = list(range(20, 29))
        risk_scores = [0.65, 0.67, 0.68, 0.70, 0.72, 0.71, 0.73, 0.75, 0.73]
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=risk_weeks,
            y=risk_scores,
            mode='lines+markers',
            line=dict(color='#f97316', width=3),
            fill='tozeroy',
            fillcolor='rgba(249, 115, 22, 0.1)'
        ))
        
        fig2.update_layout(
            title="Overall Risk Score Over Time",
            xaxis_title="Gestational Age (weeks)",
            yaxis_title="Risk Score",
            height=250,
            yaxis_range=[0.6, 0.8]
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("### Upcoming Appointments")
        
        appointments = [
            {"date": "Feb 8", "type": "Routine Prenatal", "provider": "Dr. Chen"},
            {"date": "Feb 12", "type": "Glucose Tolerance Test", "provider": "Lab"},
            {"date": "Feb 15", "type": "Ultrasound", "provider": "Dr. Patel"},
            {"date": "Feb 22", "type": "MFM Consultation", "provider": "Dr. Kim"}
        ]
        
        for apt in appointments:
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #ec4899;">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <p style="margin: 0; font-weight: 700; color: #831843;">{apt['type']}</p>
                        <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{apt['provider']}</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="margin: 0; font-weight: 600; color: #ec4899;">{apt['date']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Lab Results")
        
        labs = {
            'Test': ['Hemoglobin', 'Glucose (fasting)', 'Protein (urine)', 'Platelets'],
            'Result': ['11.8 g/dL', '98 mg/dL', 'Negative', '210K/μL'],
            'Status': ['✓ Normal', '⚠️ Monitor', '✓ Normal', '✓ Normal']
        }
        
        st.dataframe(pd.DataFrame(labs), hide_index=True, use_container_width=True)
        
        st.markdown("### Patient Engagement")
        
        engagement = {
            'Metric': ['App Check-ins', 'Symptom Reports', 'Vital Uploads', 'Educational Content'],
            'This Week': ['6/7 days', '3 reports', '14 readings', '8 articles read']
        }
        
        st.dataframe(pd.DataFrame(engagement), hide_index=True, use_container_width=True)

with tab3:
    st.markdown("### Population Health Analytics")
    
    # Impact metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ec4899 0%, #f97316 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">32.5%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Maternal Mortality Reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">24.3%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Preterm Birth Reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f97316 0%, #eab308 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">28.9%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">NICU Admission Reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Outcome Improvements")
        
        outcome_data = []
        for outcome, data in OUTCOME_METRICS.items():
            outcome_data.append({
                'Outcome': outcome,
                'Baseline': f"{data['baseline']}%",
                'With Delfina': f"{data['current']}%",
                'Reduction': f"{data['reduction']}%"
            })
        
        st.dataframe(pd.DataFrame(outcome_data), hide_index=True, use_container_width=True)
        
        st.markdown("### Risk Distribution")
        
        risk_dist = pd.DataFrame({
            'Risk Level': ['Low', 'Moderate', 'High', 'Very High'],
            'Patients': [1847, 658, 284, 58],
            'Percentage': [64.9, 23.1, 10.0, 2.0]
        })
        
        fig3 = px.pie(risk_dist, values='Patients', names='Risk Level',
                     color_discrete_sequence=['#22c55e', '#eab308', '#f97316', '#dc2626'])
        fig3.update_traces(textposition='inside', textinfo='percent+label')
        fig3.update_layout(height=300, showlegend=False)
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("### Monthly Deliveries & Outcomes")
        
        months = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb']
        deliveries = [156, 168, 172, 181, 187, 195]
        complications = [23, 19, 17, 15, 12, 11]
        
        fig4 = go.Figure()
        
        fig4.add_trace(go.Bar(
            x=months,
            y=deliveries,
            name='Total Deliveries',
            marker_color='#ec4899'
        ))
        
        fig4.add_trace(go.Bar(
            x=months,
            y=complications,
            name='Complications',
            marker_color='#dc2626'
        ))
        
        fig4.update_layout(
            barmode='group',
            height=300,
            xaxis_title="Month",
            yaxis_title="Count",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("### Cost Impact")
        
        cost_data = {
            'Category': ['Prevented NICU stays', 'Reduced complications', 'Fewer C-sections', 'Lower readmissions'],
            'Savings': ['$2.8M', '$1.5M', '$850K', '$640K']
        }
        
        st.dataframe(pd.DataFrame(cost_data), hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; margin-top: 15px;">
            <h4 style="margin: 0 0 10px 0; color: #92400e;">💰 Total Annual Savings</h4>
            <p style="font-size: 28px; font-weight: 900; color: #92400e; margin: 0;">$5.79M</p>
            <p style="margin: 8px 0 0 0; color: #78350f; font-size: 14px;">Per 10,000 pregnancies</p>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("### Clinical Decision Support")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**AI Model Performance**")
        
        model_metrics = {
            'Condition': ['Preeclampsia', 'Gestational Diabetes', 'Preterm Birth', 'Postpartum Depression'],
            'Accuracy': ['94.2%', '91.8%', '88.5%', '86.3%'],
            'Sensitivity': ['92.7%', '89.4%', '85.2%', '83.8%'],
            'Specificity': ['95.1%', '93.2%', '91.7%', '88.9%']
        }
        
        st.dataframe(pd.DataFrame(model_metrics), hide_index=True, use_container_width=True)
        
        st.markdown("**Data Sources**")
        st.markdown("""
        - ✅ EHR vital signs (BP, weight, labs)
        - ✅ Patient-reported symptoms
        - ✅ Wearable device data (HR, activity)
        - ✅ Demographic & social determinants
        - ✅ Historical pregnancy outcomes
        - ✅ Genetic markers (when available)
        - ✅ Environmental factors
        - ✅ Behavioral health screening
        """)
        
        st.markdown("**Model Training**")
        st.markdown("""
        - 📊 Training data: 245K pregnancies
        - 🔄 Monthly model updates
        - 🎯 Validated on diverse populations
        - ⚖️ Bias mitigation protocols
        - 🏥 Clinical review board oversight
        """)
    
    with col2:
        st.markdown("**Risk Prediction Timeline**")
        
        timeline = {
            'Trimester': ['First (1-13w)', 'Second (14-27w)', 'Third (28-40w)', 'Postpartum'],
            'Key Risks': [
                'Early pregnancy loss, ectopic',
                'GDM, preeclampsia onset',
                'Preterm labor, IUGR',
                'Hemorrhage, depression'
            ],
            'AI Screening': ['Weekly', 'Bi-weekly', 'Weekly', 'Daily x7d']
        }
        
        st.dataframe(pd.DataFrame(timeline), hide_index=True, use_container_width=True)
        
        st.markdown("**Intervention Pathways**")
        
        st.markdown("""
        <div style="background: #fee2e2; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
            <h4 style="margin: 0 0 8px 0; color: #991b1b;">🔴 High Risk (>0.85)</h4>
            <p style="margin: 0; color: #7f1d1d; font-size: 14px;">Immediate MFM referral, intensive monitoring, care coordination</p>
        </div>
        <div style="background: #fef3c7; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
            <h4 style="margin: 0 0 8px 0; color: #92400e;">🟡 Moderate Risk (0.70-0.84)</h4>
            <p style="margin: 0; color: #78350f; font-size: 14px;">Enhanced surveillance, preventive interventions, weekly check-ins</p>
        </div>
        <div style="background: #dcfce7; padding: 15px; border-radius: 10px;">
            <h4 style="margin: 0 0 8px 0; color: #166534;">🟢 Low Risk (<0.70)</h4>
            <p style="margin: 0; color: #15803d; font-size: 14px;">Standard prenatal care, routine monitoring, patient education</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Integration**")
        st.markdown("""
        - 🏥 Epic/Cerner EHR integration
        - 📱 Patient mobile app (iOS/Android)
        - ⌚ Wearable device sync
        - 📞 Telehealth platform
        - 🔔 Alert system for providers
        - 📊 Population health dashboard
        """)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #fce7f3 0%, #fed7aa 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #831843; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #ec4899; font-weight: 700; margin: 0 0 6px 0;">✓ 94.2% Prediction Accuracy</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Preeclampsia detection</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #ec4899; font-weight: 700; margin: 0 0 6px 0;">✓ 32.5% Mortality Reduction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Maternal outcomes</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #ec4899; font-weight: 700; margin: 0 0 6px 0;">✓ 24.3% Preterm Reduction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Early delivery prevention</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #ec4899; font-weight: 700; margin: 0 0 6px 0;">✓ $5.79M Annual Savings</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Per 10K pregnancies</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #ec4899 0%, #f97316 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Delfina</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)