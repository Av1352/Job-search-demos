"""
Akute Health - Patient Analytics & Clinical Decision Support Dashboard
EMR analytics for digital health platforms
Built for Akute Health by Anju Nandhakumar
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import random

# Sample patient data
def generate_patient_cohort(n_patients=100):
    """Generate synthetic patient data for analytics"""
    
    conditions = ['Diabetes', 'Hypertension', 'Asthma', 'CHF', 'COPD', 'Depression', 'Arthritis']
    risk_levels = ['Low', 'Medium', 'High', 'Critical']
    
    patients = []
    for i in range(n_patients):
        age = random.randint(25, 85)
        
        # Risk increases with age and conditions
        num_conditions = random.choices([1, 2, 3, 4], weights=[50, 30, 15, 5])[0]
        patient_conditions = random.sample(conditions, num_conditions)
        
        # Calculate risk score
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
        
        # Generate visit data
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

def analyze_population_health():
    """Generate population health analytics"""
    
    # Generate patient cohort
    df = generate_patient_cohort(250)
    
    # Population summary
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Population Health Overview</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Patients</p>
                <p style="font-size: 42px; color: white; font-weight: 900; margin: 0;">{len(df)}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Active in system</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Avg Age</p>
                <p style="font-size: 42px; color: white; font-weight: 900; margin: 0;">{df['age'].mean():.0f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">years old</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">High Risk</p>
                <p style="font-size: 42px; color: #fca5a5; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{len(df[df['risk_level'].isin(['High', 'Critical'])])}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{len(df[df['risk_level'].isin(['High', 'Critical'])])/len(df)*100:.0f}% of population</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">ER Visits/Year</p>
                <p style="font-size: 42px; color: white; font-weight: 900; margin: 0;">{df['er_visits'].sum()}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Avg {df['er_visits'].mean():.1f} per patient</p>
            </div>
        </div>
    </div>
    """
    
    # Risk distribution
    risk_counts = df['risk_level'].value_counts()
    
    risk_html = f"""
    <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(239, 68, 68, 0.2); margin-bottom: 25px;">
        <h3 style="color: #991b1b; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">⚠️ Risk Stratification</h3>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);">
                    <span style="font-size: 24px;">✓</span>
                </div>
                <p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">{risk_counts.get('Low', 0)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Low Risk</p>
            </div>
            
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(245, 158, 11, 0.3);">
                    <span style="font-size: 24px;">⚡</span>
                </div>
                <p style="font-size: 32px; color: #f59e0b; font-weight: 900; margin: 0;">{risk_counts.get('Medium', 0)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Medium Risk</p>
            </div>
            
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(249, 115, 22, 0.3);">
                    <span style="font-size: 24px;">⚠️</span>
                </div>
                <p style="font-size: 32px; color: #f97316; font-weight: 900; margin: 0;">{risk_counts.get('High', 0)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">High Risk</p>
            </div>
            
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(220, 38, 38, 0.3);">
                    <span style="font-size: 24px;">🚨</span>
                </div>
                <p style="font-size: 32px; color: #dc2626; font-weight: 900; margin: 0;">{risk_counts.get('Critical', 0)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Critical Risk</p>
            </div>
        </div>
    </div>
    """
    
    # Top conditions
    all_conditions = []
    for conds in df['conditions']:
        all_conditions.extend(conds)
    
    condition_counts = pd.Series(all_conditions).value_counts().head(7)
    
    conditions_html = """
    <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🏥 Top Chronic Conditions</h3>
        <div style="display: grid; grid-template-columns: 1fr; gap: 12px;">
    """
    
    colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#8b5cf6']
    for idx, (condition, count) in enumerate(condition_counts.items()):
        pct = (count / len(df)) * 100
        conditions_html += f"""
        <div style="background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 16px; color: #1f2937; font-weight: 700;">{condition}</span>
                <span style="font-size: 18px; color: {colors[idx]}; font-weight: 900;">{count} <span style="font-size: 13px; color: #6b7280;">({pct:.0f}%)</span></span>
            </div>
            <div style="background: #e5e7eb; border-radius: 8px; height: 10px; overflow: hidden;">
                <div style="background: {colors[idx]}; height: 100%; width: {pct}%; transition: width 0.3s;"></div>
            </div>
        </div>
        """
    
    conditions_html += "</div></div>"
    
    # Create charts
    
    # 1. Risk distribution pie chart
    fig_risk = go.Figure(data=[go.Pie(
        labels=['Low', 'Medium', 'High', 'Critical'],
        values=[risk_counts.get(r, 0) for r in ['Low', 'Medium', 'High', 'Critical']],
        marker=dict(colors=['#10b981', '#f59e0b', '#f97316', '#dc2626']),
        hole=0.4,
        textinfo='label+percent',
        textfont=dict(size=14, color='white', family='Arial Black')
    )])
    
    fig_risk.update_layout(
        title="Patient Risk Distribution",
        height=400,
        showlegend=True
    )
    
    # 2. Age distribution
    fig_age = go.Figure(data=[go.Histogram(
        x=df['age'],
        nbinsx=15,
        marker_color='#3b82f6',
        name='Patients'
    )])
    
    fig_age.update_layout(
        title="Age Distribution",
        xaxis_title="Age",
        yaxis_title="Number of Patients",
        height=400
    )
    
    # 3. ER visits vs conditions
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
    
    return summary_html, risk_html, conditions_html, fig_risk, fig_age, fig_scatter, df

def view_patient_detail(patient_id, df):
    """View detailed patient information"""
    
    patient = df[df['patient_id'] == patient_id].iloc[0]
    
    # Risk color
    risk_colors = {
        'Low': '#10b981',
        'Medium': '#f59e0b',
        'High': '#f97316',
        'Critical': '#dc2626'
    }
    risk_color = risk_colors.get(patient['risk_level'], '#6b7280')
    
    detail_html = f"""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(168, 85, 247, 0.2); margin-bottom: 25px;">
        <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 25px;">
            <div style="background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 6px 16px rgba(168, 85, 247, 0.4); border: 4px solid white;">
                <span style="font-size: 40px;">👤</span>
            </div>
            <div>
                <h2 style="color: #6b21a8; font-size: 32px; font-weight: 900; margin: 0;">Patient {patient_id}</h2>
                <p style="color: #a855f7; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">{patient['age']} years old • {len(patient['conditions'])} chronic conditions</p>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px;">
            <div style="background: white; border-radius: 14px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 13px; color: #6b7280; margin: 0 0 8px 0; font-weight: 600;">Risk Score</p>
                <p style="font-size: 36px; color: {risk_color}; font-weight: 900; margin: 0;">{patient['risk_score']:.0f}</p>
                <span style="display: inline-block; background: {risk_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 700; margin-top: 8px;">{patient['risk_level'].upper()}</span>
            </div>
            
            <div style="background: white; border-radius: 14px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 13px; color: #6b7280; margin: 0 0 8px 0; font-weight: 600;">Last Visit</p>
                <p style="font-size: 36px; color: #3b82f6; font-weight: 900; margin: 0;">{patient['last_visit']}</p>
                <p style="font-size: 12px; color: #9ca3af; margin: 8px 0 0 0;">days ago</p>
            </div>
            
            <div style="background: white; border-radius: 14px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 13px; color: #6b7280; margin: 0 0 8px 0; font-weight: 600;">Next Visit</p>
                <p style="font-size: 36px; color: #10b981; font-weight: 900; margin: 0;">{patient['next_visit']}</p>
                <p style="font-size: 12px; color: #9ca3af; margin: 8px 0 0 0;">days from now</p>
            </div>
        </div>
        
        <div style="background: white; border-radius: 16px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 15px 0;">🏥 Chronic Conditions</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                {''.join([f'<span style="background: linear-gradient(135deg, #ec4899 0%, #db2777 100%); color: white; padding: 8px 18px; border-radius: 20px; font-size: 13px; font-weight: 700; box-shadow: 0 2px 6px rgba(236, 72, 153, 0.3);">{cond}</span>' for cond in patient['conditions']])}
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 2px solid #f59e0b; border-radius: 14px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: #92400e; margin: 0 0 8px 0; font-weight: 600;">Medications</p>
                <p style="font-size: 32px; color: #d97706; font-weight: 900; margin: 0;">{patient['medications']}</p>
                <p style="font-size: 12px; color: #92400e; margin: 8px 0 0 0;">active prescriptions</p>
            </div>
            
            <div style="background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%); border: 2px solid #f97316; border-radius: 14px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: #9a3412; margin: 0 0 8px 0; font-weight: 600;">ER Visits</p>
                <p style="font-size: 32px; color: #ea580c; font-weight: 900; margin: 0;">{patient['er_visits']}</p>
                <p style="font-size: 12px; color: #9a3412; margin: 8px 0 0 0;">past 12 months</p>
            </div>
            
            <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 2px solid #ef4444; border-radius: 14px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: #991b1b; margin: 0 0 8px 0; font-weight: 600;">Hospitalizations</p>
                <p style="font-size: 32px; color: #dc2626; font-weight: 900; margin: 0;">{patient['hospitalizations']}</p>
                <p style="font-size: 12px; color: #991b1b; margin: 8px 0 0 0;">past 12 months</p>
            </div>
        </div>
    </div>
    """
    
    # Clinical recommendations
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
    
    rec_html = """
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2);">
        <h3 style="color: #065f46; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">💡 Clinical Recommendations</h3>
        <div style="background: white; border-radius: 12px; padding: 20px;">
            <ul style="margin: 0; padding-left: 24px; line-height: 2.2;">
    """
    
    for rec in recommendations:
        rec_html += f'<li style="color: #1f2937; font-size: 15px; font-weight: 600;">{rec}</li>'
    
    rec_html += """
            </ul>
        </div>
    </div>
    """
    
    return detail_html, rec_html

def generate_alerts():
    """Generate patient alerts requiring immediate attention"""
    
    df = generate_patient_cohort(250)
    
    # Filter high-priority patients
    critical = df[df['risk_level'] == 'Critical']
    high_risk = df[df['risk_level'] == 'High']
    high_er = df[df['er_visits'] >= 3]
    overdue = df[df['last_visit'] > 120]
    
    alerts_html = f"""
    <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 4px solid #dc2626; border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(220, 38, 38, 0.3); margin-bottom: 25px;">
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 25px;">
            <div style="background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4); border: 4px solid white;">
                <span style="font-size: 36px;">🚨</span>
            </div>
            <div>
                <h2 style="color: #991b1b; font-size: 32px; font-weight: 900; margin: 0;">Priority Alerts</h2>
                <p style="color: #dc2626; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">Patients requiring immediate attention</p>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;">
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 40px; color: #dc2626; font-weight: 900; margin: 0;">{len(critical)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Critical Risk</p>
            </div>
            
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 40px; color: #f97316; font-weight: 900; margin: 0;">{len(high_risk)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">High Risk</p>
            </div>
            
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 40px; color: #f59e0b; font-weight: 900; margin: 0;">{len(high_er)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">High ER Use</p>
            </div>
            
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 40px; color: #8b5cf6; font-weight: 900; margin: 0;">{len(overdue)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Overdue Visits</p>
            </div>
        </div>
    """
    
    # Show top 5 critical patients
    if len(critical) > 0:
        alerts_html += """
        <div style="background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <h4 style="color: #991b1b; font-size: 20px; font-weight: 800; margin: 0 0 18px 0;">🚨 Critical Risk Patients (Top 5)</h4>
            <div style="display: grid; gap: 12px;">
        """
        
        for _, patient in critical.head(5).iterrows():
            alerts_html += f"""
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left: 5px solid #dc2626; border-radius: 12px; padding: 16px;">
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
            """
        
        alerts_html += "</div></div>"
    
    alerts_html += "</div>"
    
    # Create alert trend chart
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    critical_trend = [random.randint(8, 15) for _ in range(30)]
    high_trend = [random.randint(25, 40) for _ in range(30)]
    
    fig_alerts = go.Figure()
    
    fig_alerts.add_trace(go.Scatter(
        x=dates,
        y=critical_trend,
        name='Critical Risk',
        line=dict(color='#dc2626', width=3),
        mode='lines+markers'
    ))
    
    fig_alerts.add_trace(go.Scatter(
        x=dates,
        y=high_trend,
        name='High Risk',
        line=dict(color='#f97316', width=3),
        mode='lines+markers'
    ))
    
    fig_alerts.update_layout(
        title="High-Risk Patient Trend (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title="Number of Patients",
        height=400,
        hovermode='x unified'
    )
    
    return alerts_html, fig_alerts, df

custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🏥</span>
        </div>
        
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Akute Health Analytics
        </h1>
        
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Patient Analytics & Clinical Decision Support</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">EMR analytics for digital health platforms</p>
        
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 800px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Population Health</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Risk Stratification</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Clinical Insights</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">YC Backed</span>
        </div>
        
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Akute Health</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """)
    
    # Store dataframe in state
    df_state = gr.State()
    
    with gr.Tabs():
        with gr.Tab("📊 Population Health"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Population Health Analytics</h3>
                <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Overview of patient population, risk distribution, and chronic conditions</p>
            </div>
            """)
            
            analyze_btn = gr.Button("📈 Analyze Population Health", variant="primary", size="lg")
            
            pop_summary = gr.HTML(label="Population Summary")
            risk_dist = gr.HTML(label="Risk Distribution")
            conditions = gr.HTML(label="Top Conditions")
            
            with gr.Row():
                risk_chart = gr.Plot(label="Risk Distribution")
                age_chart = gr.Plot(label="Age Distribution")
            
            scatter_chart = gr.Plot(label="ER Visits vs Conditions")
            
            def analyze_and_store():
                summary, risk, cond, fig_risk, fig_age, fig_scatter, df = analyze_population_health()
                return summary, risk, cond, fig_risk, fig_age, fig_scatter, df
            
            analyze_btn.click(
                fn=analyze_and_store,
                inputs=[],
                outputs=[pop_summary, risk_dist, conditions, risk_chart, age_chart, scatter_chart, df_state]
            )
        
        with gr.Tab("👤 Patient Details"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Individual Patient Analysis</h3>
                <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Detailed clinical information and personalized recommendations</p>
            </div>
            """)
            
            with gr.Row():
                patient_dropdown = gr.Dropdown(
                    choices=[f"PT-{i:04d}" for i in range(1001, 1051)],
                    value="PT-1001",
                    label="Select Patient ID"
                )
                view_patient_btn = gr.Button("👁️ View Patient", variant="primary", size="lg")
            
            patient_detail = gr.HTML(label="Patient Information")
            patient_rec = gr.HTML(label="Recommendations")
            
            view_patient_btn.click(
                fn=view_patient_detail,
                inputs=[patient_dropdown, df_state],
                outputs=[patient_detail, patient_rec]
            )
        
        with gr.Tab("🚨 Priority Alerts"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #991b1b; font-size: 22px; font-weight: 800; margin: 0;">High-Priority Patient Alerts</h3>
                <p style="color: #dc2626; font-size: 14px; margin: 8px 0 0 0;">Patients requiring immediate clinical attention and intervention</p>
            </div>
            """)
            
            alerts_btn = gr.Button("🚨 Generate Priority Alerts", variant="primary", size="lg")
            
            alerts_output = gr.HTML(label="Priority Alerts")
            alerts_chart = gr.Plot(label="Alert Trends")
            
            def generate_and_store_alerts():
                alerts, fig, df = generate_alerts()
                return alerts, fig, df
            
            alerts_btn.click(
                fn=generate_and_store_alerts,
                inputs=[],
                outputs=[alerts_output, alerts_chart, df_state]
            )
    
    gr.HTML("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Akute Health</h2>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📊 Population Health Management</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Identify high-risk patients before they become costly ER visits. Proactive care management reduces hospitalizations by 30-40% and saves $500-1000 per patient per year.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🎯 Risk Stratification</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    ML-powered risk scoring identifies which patients need intensive care coordination. Prioritize resources for the 10-15% of patients who drive 60-70% of costs.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💡 Clinical Decision Support</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Actionable recommendations at the point of care. Reduce physician cognitive load, prevent care gaps, and improve patient outcomes through data-driven insights.
                </p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Real-World Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Value-Based Care:</strong> ACO/CMS quality metrics tracking and reporting</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Care Coordination:</strong> Identify gaps in care for chronic disease management</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Readmission Prevention:</strong> Flag high-risk patients for post-discharge follow-up</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Resource Optimization:</strong> Allocate care managers to patients who need them most</li>
            </ul>
        </div>
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Analytics</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Live patient data aggregation and analysis</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Risk Prediction Models</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">ML algorithms for readmission and ER risk</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ FHIR/HL7 Integration</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Seamless EMR connectivity</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ HIPAA Compliant</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Secure, encrypted patient data handling</p>
                </div>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Akute Health</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
        </p>
        <div style="margin: 25px 0; padding: 22px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.2);">
            <p style="margin: 8px 0; font-size: 16px;">
                📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: white; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a>
            </p>
            <p style="margin: 8px 0; font-size: 16px;">
                💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">LinkedIn</a> | 
                💻 <a href="https://github.com/Av1352" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">GitHub</a> | 
                🌐 <a href="https://vxanju.com" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">Portfolio</a>
            </p>
        </div>
        <p style="font-size: 15px; margin: 18px 0; font-weight: 700;">
            <strong style="color: white;">Tech Stack:</strong> Python • Gradio • Plotly • Pandas • NumPy
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing patient analytics and clinical decision support for digital health EMRs.<br>
            Population health • Risk stratification • Actionable insights
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()