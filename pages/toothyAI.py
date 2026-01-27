"""
Toothy AI - Insurance Verification & Billing Automation
AI-powered dental claim processing and eligibility verification
Built for Toothy AI by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Toothy AI", page_icon="🦷", layout="wide")

# Insurance plans database
INSURANCE_PLANS = {
    'Delta Dental PPO': {'coverage': 80, 'deductible': 50, 'annual_max': 1500},
    'MetLife Enhanced': {'coverage': 70, 'deductible': 75, 'annual_max': 1200},
    'Cigna Basic': {'coverage': 60, 'deductible': 100, 'annual_max': 1000},
    'Aetna Premier': {'coverage': 85, 'deductible': 25, 'annual_max': 2000},
    'United Healthcare': {'coverage': 75, 'deductible': 50, 'annual_max': 1500}
}

# Procedure codes and costs
PROCEDURES = {
    'D0150': {'name': 'Comprehensive Oral Exam', 'cost': 95, 'category': 'Preventive'},
    'D0210': {'name': 'X-rays (Full Mouth)', 'cost': 150, 'category': 'Preventive'},
    'D1110': {'name': 'Teeth Cleaning', 'cost': 120, 'category': 'Preventive'},
    'D2140': {'name': 'Amalgam Filling (1 surface)', 'cost': 180, 'category': 'Basic'},
    'D2740': {'name': 'Crown (Porcelain)', 'cost': 1250, 'category': 'Major'},
    'D3310': {'name': 'Root Canal (Anterior)', 'cost': 850, 'category': 'Major'},
    'D4341': {'name': 'Periodontal Scaling (per quadrant)', 'cost': 200, 'category': 'Basic'},
    'D5110': {'name': 'Complete Denture (Upper)', 'cost': 1800, 'category': 'Major'},
    'D6010': {'name': 'Surgical Implant Placement', 'cost': 2400, 'category': 'Major'}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #10b981 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🦷</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Toothy AI</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Insurance Verification & Billing Automation</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Instant eligibility checks • Automated claim processing • Real-time cost estimates</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🔍 Eligibility Verification", "💰 Claim Calculator"])

with tab1:
    st.markdown("### Check Patient Insurance Status")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        patient_name = st.text_input("Patient Name", placeholder="John Doe")
        insurance_plan = st.selectbox("Insurance Plan", list(INSURANCE_PLANS.keys()))
        member_id = st.text_input("Member ID", placeholder="ABC123456789")
        verify_btn = st.button("🔍 Verify Eligibility", type="primary", use_container_width=True)
    
    with col2:
        if verify_btn and patient_name and member_id:
            # Simulate eligibility check
            plan_info = INSURANCE_PLANS[insurance_plan]
            is_active = np.random.random() > 0.1
            
            if is_active:
                status = "✅ ACTIVE"
                status_color = "#10b981"
                remaining_benefit = plan_info['annual_max'] * np.random.uniform(0.3, 0.9)
            else:
                status = "❌ INACTIVE"
                status_color = "#ef4444"
                remaining_benefit = 0
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {status_color} 0%, #73BA9B 100%); padding: 30px; border-radius: 16px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <div style="text-align: center; color: white;">
                    <h2 style="margin: 0 0 20px 0; font-size: 32px; font-weight: 900;">Eligibility Status</h2>
                    <div style="background: rgba(255,255,255,0.2); backdrop-filter: blur(10px); border-radius: 12px; padding: 25px; margin-bottom: 20px;">
                        <p style="font-size: 48px; margin: 0; font-weight: 900;">{status}</p>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 25px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Patient</p>
                        <p style="font-size: 20px; color: white; font-weight: 700; margin: 0;">{patient_name}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Member ID</p>
                        <p style="font-size: 20px; color: white; font-weight: 700; margin: 0;">{member_id}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Coverage</p>
                        <p style="font-size: 20px; color: white; font-weight: 700; margin: 0;">{plan_info['coverage']}%</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Remaining Benefits</p>
                        <p style="font-size: 20px; color: white; font-weight: 700; margin: 0;">${remaining_benefit:.0f}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Deductible</p>
                        <p style="font-size: 20px; color: white; font-weight: 700; margin: 0;">${plan_info['deductible']}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Annual Max</p>
                        <p style="font-size: 20px; color: white; font-weight: 700; margin: 0;">${plan_info['annual_max']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Calculate Claim Breakdown")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        claim_insurance = st.selectbox("Insurance Plan", list(INSURANCE_PLANS.keys()), key="claim_insurance")
        
        st.markdown("**Select Procedures:**")
        selected_procedures = []
        for code, proc in PROCEDURES.items():
            if st.checkbox(f"{code}: {proc['name']} (${proc['cost']})", key=code):
                selected_procedures.append(code)
        
        calculate_btn = st.button("💰 Calculate Claim", type="primary", use_container_width=True)
    
    with col2:
        if calculate_btn and selected_procedures:
            plan_info = INSURANCE_PLANS[claim_insurance]
            
            total_cost = 0
            insurance_pays = 0
            patient_pays = 0
            procedure_details = []
            
            for code in selected_procedures:
                proc = PROCEDURES[code]
                cost = proc['cost']
                
                if proc['category'] == 'Preventive':
                    coverage_pct = min(100, plan_info['coverage'] + 20)
                elif proc['category'] == 'Basic':
                    coverage_pct = plan_info['coverage']
                else:
                    coverage_pct = max(50, plan_info['coverage'] - 10)
                
                insurance_amount = cost * coverage_pct / 100
                patient_amount = cost - insurance_amount
                
                total_cost += cost
                insurance_pays += insurance_amount
                patient_pays += patient_amount
                
                procedure_details.append({
                    'Code': code,
                    'Procedure': proc['name'],
                    'Category': proc['category'],
                    'Total': f'${cost:.2f}',
                    'Insurance': f'${insurance_amount:.2f}',
                    'Patient': f'${patient_amount:.2f}'
                })
            
            # Summary
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h2 style="color: white; text-align: center; margin: 0 0 25px 0; font-size: 28px; font-weight: 900;">Claim Summary</h2>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Total Cost</p>
                        <p style="font-size: 32px; color: white; font-weight: 900; margin: 0;">${total_cost:.2f}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Insurance Pays</p>
                        <p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">${insurance_pays:.2f}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Patient Pays</p>
                        <p style="font-size: 32px; color: #f59e0b; font-weight: 900; margin: 0;">${patient_pays:.2f}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Table
            df = pd.DataFrame(procedure_details)
            st.dataframe(df, use_container_width=True)
            
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Insurance Pays',
                x=[p['Code'] for p in procedure_details],
                y=[float(p['Insurance'].replace('$','')) for p in procedure_details],
                marker=dict(color='#10b981'),
                text=[p['Insurance'] for p in procedure_details],
                textposition='auto'
            ))
            fig.add_trace(go.Bar(
                name='Patient Pays',
                x=[p['Code'] for p in procedure_details],
                y=[float(p['Patient'].replace('$','')) for p in procedure_details],
                marker=dict(color='#f59e0b'),
                text=[p['Patient'] for p in procedure_details],
                textposition='auto'
            ))
            fig.update_layout(
                title='Cost Breakdown by Procedure',
                barmode='stack',
                xaxis_title='Procedure Code',
                yaxis_title='Amount ($)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #065f46; font-size: 24px; font-weight: 800;">💡 System Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Real-time Eligibility</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Instant insurance verification</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Automated Claims</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Pre-calculated cost breakdowns</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ 5 Major Insurers</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Delta, MetLife, Cigna, Aetna, UHC</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ 9 Procedure Codes</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Preventive, basic, major coverage</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #10b981 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Toothy AI</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)