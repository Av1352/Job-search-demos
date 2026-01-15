"""
Akute Health - Patient Analytics & Clinical Decision Support Dashboard
EMR analytics for digital health platforms
Built for Akute Health by Anju Nandhakumar
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np

# Page config
st.set_page_config(
    page_title="Akute Health Demo - Anju Vilashni",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px 24px;
}
.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px 32px;
    font-size: 16px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# Sample patient data
def generate_patient_cohort(n_patients=100):
    """Generate synthetic patient data for analytics"""
    
    conditions = ['Diabetes', 'Hypertension', 'Asthma', 'CHF', 'COPD', 'Depression', 'Arthritis']
    risk_levels = ['Low', 'Medium', 'High', 'Critical']
    
    patients = []
    for i in range(n_patients):
        age = random.randint(25, 85)
        
        num_conditions = random.choices([1, 2, 3, 4], weights=[50, 30, 15, 5])[0]
        patient_conditions = random.sample(conditions, num_conditions)
        
        base_risk = (age / 100) * 30
        condition_risk = num_conditions * 15
        total_risk = min(base_risk + condition_risk + random.randint(-10, 10), 100)
        
        if total_risk < 30:
            risk = 'Low'
        elif total_risk < 60:
            risk = 'Medium'
        elif total_risk < 85:
            risk = 'High'
        else:
            risk = 'Critical'
        
        last_visit_days = random.randint(1, 180)
        next_visit_days = random.randint(7, 90)
        
        patients.append({
            'patient_id': f'PT-{i+1001:04d}',
            'age': age,
            'conditions': patient_conditions,
            'num_conditions': num_conditions,
            'risk_score': total_risk,
            'risk_level': risk,
            'last_visit': last_visit_days,
            'next_visit': next_visit_days,
            'medications': random.randint(0, 8),
            'er_visits': random.choices([0, 1, 2, 3, 4], weights=[60, 25, 10, 3, 2])[0],
            'hospitalizations': random.choices([0, 1, 2], weights=[70, 25, 5])[0]
        })
    
    return pd.DataFrame(patients)

# Header
components.html(
    """
    <div style="
        text-align: center;
        padding: 20px 30px 70px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 25px;
        box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);
    ">
        <div style="
            width: 100px;
            height: 100px;
            background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
            border-radius: 50%;
            margin: 0 auto 25px auto;
            border: 5px solid white;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5);
        ">
            <span style="font-size: 56px;">🏥</span>
        </div>

        <h1 style="
            font-size: 58px;
            font-weight: 900;
            color: white;
            margin: 0 0 18px 0;
            text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            Akute Health Analytics
        </h1>

        <p style="
            font-size: 28px;
            color: rgba(255,255,255,0.95);
            font-weight: 700;
            margin: 15px 0;
        ">
            Patient Analytics & Clinical Decision Support
        </p>

        <p style="
            font-size: 18px;
            color: rgba(255,255,255,0.85);
            font-weight: 500;
            margin-bottom: 25px;
        ">
            EMR analytics for digital health platforms
        </p>

        <div style="
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
            max-width: 800px;
            margin: 28px auto 0 auto;
        ">
            <span style="background:#ec4899;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Population Health</span>
            <span style="background:#8b5cf6;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Risk Stratification</span>
            <span style="background:#f59e0b;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Clinical Insights</span>
            <span style="background:#10b981;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">YC Backed</span>
        </div>

        <p style="
            font-size: 16px;
            color: rgba(255,255,255,0.9);
            margin-top: 28px;
            font-weight: 600;
        ">
            Built for <strong style="color:white;">Akute Health</strong>
            by <strong style="color:white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """,
    height=520,
)

st.markdown("---")

# Initialize session state for dataframe
if 'patient_df' not in st.session_state:
    st.session_state.patient_df = None

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Population Health", "👤 Patient Details", "🚨 Priority Alerts"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Population Health Analytics</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Overview of patient population, risk distribution, and chronic conditions</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📈 Analyze Population Health", key="analyze_pop"):
        # Generate patient cohort
        df = generate_patient_cohort(250)
        st.session_state.patient_df = df
        
        # Population summary
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Population Health Overview</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Total Patients</p>
                <p style="font-size: 42px; color: #667eea; font-weight: 900; margin: 0;">{len(df)}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Active in system</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Avg Age</p>
                <p style="font-size: 42px; color: #667eea; font-weight: 900; margin: 0;">{df['age'].mean():.0f}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">years old</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            high_risk_count = len(df[df['risk_level'].isin(['High', 'Critical'])])
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(239, 68, 68, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">High Risk</p>
                <p style="font-size: 42px; color: #ef4444; font-weight: 900; margin: 0;">{high_risk_count}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">{high_risk_count/len(df)*100:.0f}% of population</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">ER Visits/Year</p>
                <p style="font-size: 42px; color: #667eea; font-weight: 900; margin: 0;">{df['er_visits'].sum()}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Avg {df['er_visits'].mean():.1f} per patient</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Risk stratification
        risk_counts = df['risk_level'].value_counts()
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #991b1b; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">⚠️ Risk Stratification</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 24px;">✓</span>
                </div>
                <p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">{risk_counts.get('Low', 0)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Low Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 24px;">⚡</span>
                </div>
                <p style="font-size: 32px; color: #f59e0b; font-weight: 900; margin: 0;">{risk_counts.get('Medium', 0)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Medium Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 24px;">⚠️</span>
                </div>
                <p style="font-size: 32px; color: #f97316; font-weight: 900; margin: 0;">{risk_counts.get('High', 0)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">High Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 24px;">🚨</span>
                </div>
                <p style="font-size: 32px; color: #dc2626; font-weight: 900; margin: 0;">{risk_counts.get('Critical', 0)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Critical Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Top conditions
        all_conditions = []
        for conds in df['conditions']:
            all_conditions.extend(conds)
        
        condition_counts = pd.Series(all_conditions).value_counts().head(7)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🏥 Top Chronic Conditions</h3>
        </div>
        """, unsafe_allow_html=True)
        
        colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#8b5cf6']
        for idx, (condition, count) in enumerate(condition_counts.items()):
            pct = (count / len(df)) * 100
            st.markdown(f"""
            <div style="background: white; border-radius: 12px; padding: 16px; margin-bottom: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 16px; color: #1f2937; font-weight: 700;">{condition}</span>
                    <span style="font-size: 18px; color: {colors[idx]}; font-weight: 900;">{count} <span style="font-size: 13px; color: #6b7280;">({pct:.0f}%)</span></span>
                </div>
                <div style="background: #e5e7eb; border-radius: 8px; height: 10px; overflow: hidden;">
                    <div style="background: {colors[idx]}; height: 100%; width: {pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_risk = go.Figure(data=[go.Pie(
                labels=['Low', 'Medium', 'High', 'Critical'],
                values=[risk_counts.get(r, 0) for r in ['Low', 'Medium', 'High', 'Critical']],
                marker=dict(colors=['#10b981', '#f59e0b', '#f97316', '#dc2626']),
                hole=0.4,
                textinfo='label+percent'
            )])
            fig_risk.update_layout(title="Patient Risk Distribution", height=400)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            fig_age = go.Figure(data=[go.Histogram(
                x=df['age'],
                nbinsx=15,
                marker_color='#3b82f6'
            )])
            fig_age.update_layout(
                title="Age Distribution",
                xaxis_title="Age",
                yaxis_title="Number of Patients",
                height=400
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        # Scatter plot
        fig_scatter = go.Figure(data=[go.Scatter(
            x=df['num_conditions'],
            y=df['er_visits'],
            mode='markers',
            marker=dict(
                size=df['risk_score']/5,
                color=df['risk_score'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Risk Score")
            ),
            text=[f"Patient: {pid}<br>Age: {age}<br>Risk: {risk}" 
                  for pid, age, risk in zip(df['patient_id'], df['age'], df['risk_level'])],
            hovertemplate='%{text}<br>Conditions: %{x}<br>ER Visits: %{y}<extra></extra>'
        )])
        fig_scatter.update_layout(
            title="ER Visits vs Chronic Conditions",
            xaxis_title="Number of Chronic Conditions",
            yaxis_title="ER Visits (Past Year)",
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Individual Patient Analysis</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Detailed clinical information and personalized recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        patient_id = st.selectbox(
            "Select Patient ID",
            [f"PT-{i:04d}" for i in range(1001, 1051)],
            key="patient_select"
        )
    
    with col2:
        view_btn = st.button("👁️ View Patient", key="view_patient", use_container_width=True)
    
    if view_btn:
        if st.session_state.patient_df is None:
            st.session_state.patient_df = generate_patient_cohort(250)
        
        df = st.session_state.patient_df
        
        if patient_id in df['patient_id'].values:
            patient = df[df['patient_id'] == patient_id].iloc[0]
            
            risk_colors = {
                'Low': '#10b981',
                'Medium': '#f59e0b',
                'High': '#f97316',
                'Critical': '#dc2626'
            }
            risk_color = risk_colors.get(patient['risk_level'], '#6b7280')
            
            # Patient header
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 20px; padding: 32px; margin-bottom: 25px;">
                <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 25px;">
                    <div style="background: #a855f7; width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 4px solid white;">
                        <span style="font-size: 40px;">👤</span>
                    </div>
                    <div>
                        <h2 style="color: #6b21a8; font-size: 32px; font-weight: 900; margin: 0;">Patient {patient_id}</h2>
                        <p style="color: #a855f7; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">{patient['age']} years old • {len(patient['conditions'])} chronic conditions</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background: white; border-radius: 14px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <p style="font-size: 13px; color: #6b7280; margin: 0 0 8px 0; font-weight: 600;">Risk Score</p>
                    <p style="font-size: 36px; color: {risk_color}; font-weight: 900; margin: 0;">{patient['risk_score']:.0f}</p>
                    <span style="display: inline-block; background: {risk_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 700; margin-top: 8px;">{patient['risk_level'].upper()}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: white; border-radius: 14px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <p style="font-size: 13px; color: #6b7280; margin: 0 0 8px 0; font-weight: 600;">Last Visit</p>
                    <p style="font-size: 36px; color: #3b82f6; font-weight: 900; margin: 0;">{patient['last_visit']}</p>
                    <p style="font-size: 12px; color: #9ca3af; margin: 8px 0 0 0;">days ago</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background: white; border-radius: 14px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <p style="font-size: 13px; color: #6b7280; margin: 0 0 8px 0; font-weight: 600;">Next Visit</p>
                    <p style="font-size: 36px; color: #10b981; font-weight: 900; margin: 0;">{patient['next_visit']}</p>
                    <p style="font-size: 12px; color: #9ca3af; margin: 8px 0 0 0;">days from now</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Conditions
            conditions_tags = ''.join([f'<span style="background: linear-gradient(135deg, #ec4899 0%, #db2777 100%); color: white; padding: 8px 18px; border-radius: 20px; font-size: 13px; font-weight: 700; display: inline-block; margin: 5px;">{cond}</span>' for cond in patient['conditions']])
            
            st.markdown(f"""
            <div style="background: white; border-radius: 16px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 15px 0;">🏥 Chronic Conditions</h4>
                <div>{conditions_tags}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 2px solid #f59e0b; border-radius: 14px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: #92400e; margin: 0 0 8px 0; font-weight: 600;">Medications</p>
                    <p style="font-size: 32px; color: #d97706; font-weight: 900; margin: 0;">{patient['medications']}</p>
                    <p style="font-size: 12px; color: #92400e; margin: 8px 0 0 0;">active prescriptions</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%); border: 2px solid #f97316; border-radius: 14px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: #9a3412; margin: 0 0 8px 0; font-weight: 600;">ER Visits</p>
                    <p style="font-size: 32px; color: #ea580c; font-weight: 900; margin: 0;">{patient['er_visits']}</p>
                    <p style="font-size: 12px; color: #9a3412; margin: 8px 0 0 0;">past 12 months</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 2px solid #ef4444; border-radius: 14px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: #991b1b; margin: 0 0 8px 0; font-weight: 600;">Hospitalizations</p>
                    <p style="font-size: 32px; color: #dc2626; font-weight: 900; margin: 0;">{patient['hospitalizations']}</p>
                    <p style="font-size: 12px; color: #991b1b; margin: 8px 0 0 0;">past 12 months</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            recommendations = []
            
            if patient['risk_level'] in ['High', 'Critical']:
                recommendations.append("🚨 Schedule immediate care coordination call")
                recommendations.append("📋 Review medication adherence")
            
            if patient['er_visits'] >= 2:
                recommendations.append("⚠️ High ER utilization - consider care management program")
            
            if patient['last_visit'] > 90:
                recommendations.append("📅 Overdue for wellness check - schedule appointment")
            
            if patient['medications'] >= 5:
                recommendations.append("💊 Polypharmacy risk - review for drug interactions")
            
            if patient['num_conditions'] >= 3:
                recommendations.append("🏥 Multiple comorbidities - refer to specialist care team")
            
            if not recommendations:
                recommendations.append("✅ Patient stable - continue routine monitoring")
            
            rec_list = ''.join([f'<li style="color: #1f2937; font-size: 15px; font-weight: 600;">{rec}</li>' for rec in recommendations])
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px;">
                <h3 style="color: #065f46; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">💡 Clinical Recommendations</h3>
                <div style="background: white; border-radius: 12px; padding: 20px;">
                    <ul style="margin: 0; padding-left: 24px; line-height: 2.2;">
                        {rec_list}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #991b1b; font-size: 22px; font-weight: 800; margin: 0;">High-Priority Patient Alerts</h3>
        <p style="color: #dc2626; font-size: 14px; margin: 8px 0 0 0;">Patients requiring immediate clinical attention and intervention</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚨 Generate Priority Alerts", key="alerts_btn"):
        df = generate_patient_cohort(250)
        st.session_state.patient_df = df
        
        critical = df[df['risk_level'] == 'Critical']
        high_risk = df[df['risk_level'] == 'High']
        high_er = df[df['er_visits'] >= 3]
        overdue = df[df['last_visit'] > 120]
        
        # Alerts header
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 4px solid #dc2626; border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 25px;">
                <div style="background: #dc2626; width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 4px solid white;">
                    <span style="font-size: 36px;">🚨</span>
                </div>
                <div>
                    <h2 style="color: #991b1b; font-size: 32px; font-weight: 900; margin: 0;">Priority Alerts</h2>
                    <p style="color: #dc2626; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">Patients requiring immediate attention</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 40px; color: #dc2626; font-weight: 900; margin: 0;">{len(critical)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Critical Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 40px; color: #f97316; font-weight: 900; margin: 0;">{len(high_risk)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">High Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 40px; color: #f59e0b; font-weight: 900; margin: 0;">{len(high_er)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">High ER Use</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 40px; color: #8b5cf6; font-weight: 900; margin: 0;">{len(overdue)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Overdue Visits</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Critical patients list
        if len(critical) > 0:
            st.markdown("""
            <div style="background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <h4 style="color: #991b1b; font-size: 20px; font-weight: 800; margin: 0 0 18px 0;">🚨 Critical Risk Patients (Top 5)</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for _, patient in critical.head(5).iterrows():
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left: 5px solid #dc2626; border-radius: 12px; padding: 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <p style="font-size: 16px; color: #1f2937; font-weight: 800; margin: 0 0 6px 0;">{patient['patient_id']} • Age {patient['age']}</p>
                            <p style="font-size: 13px; color: #6b7280; margin: 0;">{', '.join(patient['conditions'][:3])}</p>
                        </div>
                        <div style="text-align: right;">
                            <p style="font-size: 24px; color: #dc2626; font-weight: 900; margin: 0;">{patient['risk_score']:.0f}</p>
                            <p style="font-size: 12px; color: #991b1b; margin: 4px 0 0 0; font-weight: 600;">Risk Score</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Alert trend
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        critical_trend = [random.randint(8, 15) for _ in range(30)]
        high_trend = [random.randint(25, 40) for _ in range(30)]
        
        fig_alerts = go.Figure()
        fig_alerts.add_trace(go.Scatter(
            x=dates, y=critical_trend, name='Critical Risk',
            line=dict(color='#dc2626', width=3), mode='lines+markers'
        ))
        fig_alerts.add_trace(go.Scatter(
            x=dates, y=high_trend, name='High Risk',
            line=dict(color='#f97316', width=3), mode='lines+markers'
        ))
        fig_alerts.update_layout(
            title="High-Risk Patient Trend (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Number of Patients",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_alerts, use_container_width=True)

# Footer
st.markdown("<hr style='border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Akute Health</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: rgba(102, 126, 234, 0.1); border-radius: 16px; padding: 24px; margin-top: 20px; text-align: center;">
    <p style="margin: 8px 0; font-size: 16px;">
        📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: #667eea; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a>
    </p>
    <p style="margin: 8px 0; font-size: 16px;">
        💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: #667eea; font-weight: 700; text-decoration: none;">LinkedIn</a> | 
        💻 <a href="https://github.com/Av1352" target="_blank" style="color: #667eea; font-weight: 700; text-decoration: none;">GitHub</a> | 
        🌐 <a href="https://vxanju.com" target="_blank" style="color: #667eea; font-weight: 700; text-decoration: none;">Portfolio</a>
    </p>
    <p style="font-size: 15px; margin: 18px 0 0 0; font-weight: 700; color: #1f2937;">
        <strong>Tech Stack:</strong> Python • Streamlit • Plotly • Pandas • NumPy
    </p>
</div>
""", unsafe_allow_html=True)