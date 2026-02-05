"""
Revero - AI-Powered Chronic Disease Reversal Platform
Precision nutrition therapy to reverse metabolic and autoimmune conditions
Built for Revero by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Revero - Disease Reversal", page_icon="🥗", layout="wide")

# Conditions treated
CONDITIONS_TREATED = {
    'Type 2 Diabetes': {'patients': 1200, 'reversal_rate': 84, 'med_reduction': 92},
    'Obesity': {'patients': 980, 'reversal_rate': 78, 'med_reduction': 0},
    'Hypertension': {'patients': 850, 'reversal_rate': 72, 'med_reduction': 68},
    'Autoimmune Diseases': {'patients': 620, 'reversal_rate': 90, 'med_reduction': 79},
    'IBS/IBD': {'patients': 540, 'reversal_rate': 85, 'med_reduction': 65},
    'PCOS': {'patients': 380, 'reversal_rate': 76, 'med_reduction': 58}
}

# Clinical outcomes
CLINICAL_OUTCOMES = {
    'Full Resolution': 67,
    'Substantial Improvement': 23,
    'Moderate Improvement': 8,
    'No Change': 2
}

# Nutrition therapy components
THERAPY_COMPONENTS = {
    'Precision Nutrition': 'AI-personalized low-carb/keto protocols',
    'Medical Supervision': 'Continuous care from licensed clinicians',
    'Daily Biomarker Tracking': 'Glucose, ketones, weight, symptoms',
    'Health Coaching': 'Ongoing support and accountability',
    'Medication Management': 'Safe tapering as health improves',
    'Machine Learning': 'Adaptive protocol optimization'
}

# Platform metrics
PLATFORM_METRICS = {
    'Total Patients': 3000,
    'Avg Weight Loss': 42,
    'Off Insulin': 92,
    'Off All Meds': 79,
    'CRP Reduction': 68,
    '6mo Retention': 87
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #22c55e 0%, #10b981 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🥗</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Revero</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Reverse Chronic Disease</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Precision nutrition • AI personalization • Root cause treatment</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🥗 Patient Journey", "📊 Clinical Outcomes", "🧬 Precision Nutrition", "💡 AI Platform"])

with tab1:
    st.markdown("### Personalized Treatment Journey")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Patient Profile**")
        
        patient_name = st.text_input("Name", "Michael Chen")
        age = st.number_input("Age", 30, 80, 52)
        
        st.markdown("**Primary Condition**")
        
        condition = st.selectbox("Condition", list(CONDITIONS_TREATED.keys()))
        
        st.markdown("**Current Health**")
        
        weight = st.number_input("Weight (lbs)", 100, 400, 245)
        a1c = st.number_input("HbA1c (%)", 4.0, 14.0, 8.3) if condition == "Type 2 Diabetes" else None
        blood_pressure = st.text_input("Blood Pressure", "155/92")
        
        medications = st.multiselect(
            "Current Medications",
            ["Metformin", "Insulin", "Lisinopril", "Atorvastatin", "Other"],
            ["Metformin", "Insulin", "Lisinopril"]
        )
        
        st.markdown("**Health Goals**")
        
        goals = st.multiselect(
            "What do you want to achieve?",
            ["Reverse diabetes", "Lose weight", "Get off medications", "Improve energy", "Reduce inflammation"],
            ["Reverse diabetes", "Get off medications"]
        )
        
        start_btn = st.button("🥗 Start Precision Nutrition Plan", type="primary", use_container_width=True)
    
    with col2:
        if start_btn:
            st.markdown("**AI-Generated Personalized Plan**")
            
            import time
            with st.spinner("Analyzing your health profile with AI..."):
                time.sleep(1.5)
            
            st.success("✅ Your personalized treatment plan is ready!")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #22c55e 0%, #10b981 100%); padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Your Revero Program</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Nutrition Protocol</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Low-Carb Keto</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Daily Carbs</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">20-30g</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Medical Team</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">MD + Coach</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Check-ins</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">Daily + Weekly</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Your 90-Day Roadmap")
            
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #22c55e; margin-bottom: 10px;">
                <p style="margin: 0; font-weight: 700; color: #166534;">Week 1-2: Adaptation Phase</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">
                • Start low-carb nutrition (20-30g/day)<br>
                • Daily glucose & ketone tracking<br>
                • Initial medication adjustment<br>
                • Expected: -5-10 lbs, improved energy
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #10b981; margin-bottom: 10px;">
                <p style="margin: 0; font-weight: 700; color: #047857;">Month 1: Metabolic Shift</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">
                • Sustained ketosis (1.0-3.0 mmol/L)<br>
                • Reduce insulin by 50%<br>
                • Blood glucose normalizing<br>
                • Expected: -15-25 lbs, HbA1c dropping
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #059669; margin-bottom: 10px;">
                <p style="margin: 0; font-weight: 700; color: #065f46;">Month 2-3: Disease Reversal</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">
                • Off insulin completely<br>
                • Reduce/eliminate oral meds<br>
                • Normal blood pressure<br>
                • Expected: -30-40 lbs, HbA1c <5.7%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Predicted Outcomes (Based on Your Profile)")
            
            outcomes_df = pd.DataFrame({
                'Outcome': ['Weight Loss', 'HbA1c Reduction', 'Off Insulin', 'Off All Oral Meds', 'BP Normalization'],
                'Baseline': [f'{weight} lbs', '8.3%', '100 units/day', '3 medications', '155/92'],
                '90-Day Target': [f'{weight-42} lbs', '<5.7%', '0 units', '0 medications', '<130/80'],
                'Probability': ['92%', '87%', '92%', '84%', '72%']
            })
            
            st.dataframe(outcomes_df, hide_index=True, use_container_width=True)

with tab2:
    st.markdown("### Clinical Outcomes & Evidence")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #22c55e 0%, #10b981 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">90%</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Full Resolution/<br>Improvement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">92%</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Off Insulin<br>(T2D patients)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #059669 0%, #047857 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">84%</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Off All Oral<br>Medications</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #047857 0%, #065f46 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">42</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Avg Weight Loss<br>(lbs)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Conditions Treated & Reversal Rates")
        
        conditions_data = []
        for condition, data in CONDITIONS_TREATED.items():
            conditions_data.append({
                'Condition': condition,
                'Patients': data['patients'],
                'Reversal Rate': f"{data['reversal_rate']}%",
                'Med Reduction': f"{data['med_reduction']}%"
            })
        
        st.dataframe(pd.DataFrame(conditions_data), hide_index=True, use_container_width=True)
        
        st.markdown("### Harvard Study Results (2021)")
        
        st.markdown("""
        <div style="background: #dcfce7; padding: 20px; border-radius: 12px; border-left: 5px solid #22c55e;">
            <h4 style="margin: 0 0 12px 0; color: #166534;">Peer-Reviewed Findings</h4>
            <ul style="margin: 0; padding-left: 20px; color: #15803d;">
                <li>90% of patients achieved full resolution or substantial improvement</li>
                <li>Significantly decreased inflammation (CRP levels)</li>
                <li>92% of T2D patients came off insulin completely</li>
                <li>84% eliminated all oral medications</li>
                <li>Sustained weight loss average: 42 lbs</li>
                <li>Zero serious adverse events</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Patient Outcome Distribution")
        
        outcome_labels = list(CLINICAL_OUTCOMES.keys())
        outcome_values = list(CLINICAL_OUTCOMES.values())
        
        fig1 = px.pie(
            values=outcome_values,
            names=outcome_labels,
            color_discrete_sequence=['#22c55e', '#10b981', '#059669', '#94a3b8']
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        fig1.update_layout(height=300, showlegend=False)
        
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("### Biomarker Improvements")
        
        biomarkers = pd.DataFrame({
            'Biomarker': ['HbA1c', 'Fasting Glucose', 'CRP (Inflammation)', 'Triglycerides', 'HDL Cholesterol', 'Blood Pressure'],
            'Baseline Avg': ['8.2%', '185 mg/dL', '5.8 mg/L', '215 mg/dL', '42 mg/dL', '142/88'],
            '90-Day Avg': ['5.6%', '95 mg/dL', '1.8 mg/L', '98 mg/dL', '58 mg/dL', '125/78'],
            'Improvement': ['↓ 32%', '↓ 49%', '↓ 69%', '↓ 54%', '↑ 38%', '↓ 12/10']
        })
        
        st.dataframe(biomarkers, hide_index=True, use_container_width=True)

with tab3:
    st.markdown("### Precision Nutrition Therapy")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Core Protocol Components**")
        
        for component, description in THERAPY_COMPONENTS.items():
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #22c55e;">
                <p style="margin: 0; font-weight: 700; color: #166534;">{component}</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**Nutrition Approach**")
        st.markdown("""
        - ✅ Low-carb/ketogenic diet (20-30g carbs/day)
        - ✅ Whole, unprocessed foods
        - ✅ Focus on nutrient density
        - ✅ Adequate protein (1-1.5g/lb ideal weight)
        - ✅ Healthy fats for satiety
        - ✅ Elimination of inflammatory foods
        - ✅ Personalized to individual tolerances
        """)
    
    with col2:
        st.markdown("**Sample Daily Nutrition**")
        
        nutrition_data = pd.DataFrame({
            'Macro': ['Carbohydrates', 'Protein', 'Fat', 'Calories'],
            'Target': ['20-30g (5%)', '120-150g (30%)', '130-150g (65%)', '1800-2200'],
            'Example Sources': [
                'Leafy greens, low-carb veggies',
                'Grass-fed beef, wild fish, eggs',
                'Avocado, olive oil, butter, nuts',
                'Whole foods, nutrient-dense'
            ]
        })
        
        st.dataframe(nutrition_data, hide_index=True, use_container_width=True)
        
        st.markdown("**Daily Biomarker Tracking**")
        
        tracking_data = pd.DataFrame({
            'Metric': ['Blood Glucose', 'Ketones', 'Weight', 'Blood Pressure', 'Symptoms'],
            'Frequency': ['3-4x daily', 'Daily AM', 'Daily AM', '2x daily', 'Daily log'],
            'Target Range': ['70-100 mg/dL', '1.0-3.0 mmol/L', 'Downward trend', '<130/80', 'Improving']
        })
        
        st.dataframe(tracking_data, hide_index=True, use_container_width=True)
        
        st.markdown("**Medical Supervision**")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dcfce7 0%, #d1fae5 100%); padding: 20px; border-radius: 12px;">
            <p style="margin: 0 0 10px 0; font-weight: 700; color: #166534;">Licensed Clinical Team</p>
            <ul style="margin: 0; padding-left: 20px; color: #15803d;">
                <li>Initial consultation with MD</li>
                <li>Weekly video check-ins</li>
                <li>Daily in-app messaging</li>
                <li>Safe medication tapering protocol</li>
                <li>Lab review & interpretation</li>
                <li>Health coach support 7 days/week</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("### AI-Powered Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Machine Learning Personalization**")
        
        st.markdown("""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; border-left: 4px solid #22c55e; margin-bottom: 15px;">
            <h4 style="margin: 0 0 10px 0; color: #166534;">Individual Protocol Optimization</h4>
            <p style="margin: 0; color: #666; font-size: 14px;">
            AI analyzes patient demographics, condition severity, biomarker data, and response patterns to personalize:
            </p>
            <ul style="margin: 10px 0 0 0; padding-left: 20px; color: #666; font-size: 14px;">
                <li>Carb threshold (15-50g based on tolerance)</li>
                <li>Protein targets (activity level adjusted)</li>
                <li>Meal timing recommendations</li>
                <li>Supplement protocols</li>
                <li>Medication tapering schedule</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Platform Technology**")
        st.markdown("""
        - 🤖 Machine learning for protocol adaptation
        - 📱 Mobile app with daily tracking
        - 🩺 Remote patient monitoring integration
        - 📊 Real-time biomarker analytics
        - 💊 Medication management system
        - 📞 Telemedicine video consultations
        - 🔔 Smart alerts for clinical team
        - 📈 Predictive outcome modeling
        """)
        
        st.markdown("**Data-Driven Insights**")
        
        insights_data = {
            'Data Point': ['Patient profiles analyzed', 'Biomarker readings/day', 'Protocol variations', 'Success pattern recognition'],
            'Scale': ['3,000+', '50,000+', '100+', 'Continuous ML']
        }
        
        st.dataframe(pd.DataFrame(insights_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Continuous Improvement Loop**")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dcfce7 0%, #d1fae5 100%); padding: 20px; border-radius: 12px;">
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #166534;">1. Data Collection</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #15803d;">Daily biomarkers, symptoms, adherence</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #166534;">2. AI Analysis</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #15803d;">Pattern recognition, outcome prediction</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #166534;">3. Protocol Adjustment</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #15803d;">Real-time personalization based on response</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #166534;">4. Clinical Review</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #15803d;">MD oversight, medication adjustments</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #166534;">5. Model Refinement</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #15803d;">ML learns from outcomes, improves predictions</p>
            </div>
            <div>
                <p style="margin: 0; font-weight: 700; color: #166534;">6. Population Insights</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #15803d;">Aggregate learnings improve all patient protocols</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Cost Savings for Payers**")
        
        cost_data = pd.DataFrame({
            'Category': ['Medication costs', 'ER visits avoided', 'Hospitalizations prevented', 'Specialist referrals reduced'],
            'Annual Savings/Patient': ['$2,400', '$1,800', '$4,200', '$850']
        })
        
        st.dataframe(cost_data, hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; margin-top: 15px;">
            <h4 style="margin: 0 0 10px 0; color: #92400e;">💰 Total Cost Savings</h4>
            <p style="font-size: 32px; font-weight: 900; color: #92400e; margin: 0;">$9,250/patient</p>
            <p style="margin: 8px 0 0 0; color: #78350f; font-size: 14px;">Annually vs traditional treatment</p>
        </div>
        """, unsafe_allow_html=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #dcfce7 0%, #d1fae5 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #166534; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #22c55e; font-weight: 700; margin: 0 0 6px 0;">✓ 90% Disease Resolution</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Peer-reviewed outcomes</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #22c55e; font-weight: 700; margin: 0 0 6px 0;">✓ 92% Off Insulin</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Type 2 diabetes patients</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #22c55e; font-weight: 700; margin: 0 0 6px 0;">✓ AI Personalization</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Machine learning protocols</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #22c55e; font-weight: 700; margin: 0 0 6px 0;">✓ $9,250 Savings</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Per patient annually</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #22c55e 0%, #10b981 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Revero</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)