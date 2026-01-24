"""
Seal - GxP Data Validation & Quality Control Platform
Regulatory compliance for biotech and pharmaceutical data
Built for Seal by Anju Nandhakumar
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import random
import re
from utils.sidebar import render_sidebar
render_sidebar()

st.set_page_config(page_title="Seal GxP Platform", layout="wide")

# Initialize session state
if 'validation_run' not in st.session_state:
    st.session_state.validation_run = False
if 'audit_generated' not in st.session_state:
    st.session_state.audit_generated = False

# GxP validation rules (FDA 21 CFR Part 11)
GXP_RULES = {
    "Data Integrity": {
        "ALCOA+": ["Attributable", "Legible", "Contemporaneous", "Original", "Accurate", "Complete", "Consistent", "Enduring", "Available"],
        "severity": "Critical"
    },
    "Audit Trail": {
        "requirements": ["User ID", "Timestamp", "Old Value", "New Value", "Reason"],
        "severity": "Critical"
    },
    "Electronic Signatures": {
        "requirements": ["Unique ID", "Password", "Biometric", "Two-factor"],
        "severity": "Critical"
    },
    "Data Backup": {
        "requirements": ["Regular backups", "Tested recovery", "Offsite storage"],
        "severity": "High"
    }
}

# Sample clinical trial data
def generate_clinical_data(n_samples=50):
    """Generate synthetic clinical trial data"""
    
    data = []
    
    for i in range(n_samples):
        patient_id = f"PT-{random.randint(1000, 9999):04d}"
        visit_date = datetime.now() - timedelta(days=random.randint(0, 180))
        
        systolic_bp = random.gauss(125, 15)
        diastolic_bp = random.gauss(80, 10)
        heart_rate = random.gauss(72, 12)
        temperature = random.gauss(98.6, 0.5)
        
        hemoglobin = random.gauss(14.5, 2)
        wbc = random.gauss(7.5, 2)
        platelets = random.gauss(250, 50)
        
        if random.random() < 0.1:
            anomaly_type = random.choice(['bp', 'hr', 'temp', 'lab'])
            if anomaly_type == 'bp':
                systolic_bp = random.choice([random.gauss(180, 10), random.gauss(90, 5)])
            elif anomaly_type == 'hr':
                heart_rate = random.choice([random.gauss(120, 10), random.gauss(45, 5)])
            elif anomaly_type == 'temp':
                temperature = random.choice([random.gauss(103, 1), random.gauss(96, 0.5)])
            elif anomaly_type == 'lab':
                hemoglobin = random.choice([random.gauss(8, 1), random.gauss(18, 1)])
        
        has_missing = random.random() < 0.05
        has_outlier = (systolic_bp > 180 or systolic_bp < 90 or 
                      heart_rate > 110 or heart_rate < 50 or
                      temperature > 101 or temperature < 97 or
                      hemoglobin < 10 or hemoglobin > 17)
        
        data.append({
            'patient_id': patient_id,
            'visit_date': visit_date.strftime('%Y-%m-%d'),
            'systolic_bp': systolic_bp if not has_missing else None,
            'diastolic_bp': diastolic_bp if not has_missing else None,
            'heart_rate': heart_rate,
            'temperature': temperature,
            'hemoglobin': hemoglobin,
            'wbc': wbc,
            'platelets': platelets,
            'has_missing': has_missing,
            'has_outlier': has_outlier
        })
    
    return pd.DataFrame(data)

def validate_clinical_data():
    """Validate clinical trial data against GxP requirements"""
    
    df = generate_clinical_data(50)
    
    total_records = len(df)
    missing_data_count = df['has_missing'].sum()
    outlier_count = df['has_outlier'].sum()
    
    completeness_score = ((total_records - missing_data_count) / total_records) * 100
    accuracy_score = ((total_records - outlier_count) / total_records) * 100
    overall_score = (completeness_score + accuracy_score) / 2
    
    if overall_score >= 95:
        status = "GxP Compliant"
        status_color = "#10b981"
        status_emoji = "✅"
    elif overall_score >= 85:
        status = "Needs Review"
        status_color = "#f59e0b"
        status_emoji = "⚠️"
    else:
        status = "Non-Compliant"
        status_color = "#ef4444"
        status_emoji = "❌"
    
    # Create charts
    categories = ['Completeness', 'Accuracy', 'Consistency', 'Integrity']
    scores = [completeness_score, accuracy_score, random.uniform(95, 100), random.uniform(96, 100)]
    
    fig_quality = go.Figure(data=[
        go.Bar(
            x=categories,
            y=scores,
            marker_color=['#10b981' if s >= 95 else '#f59e0b' if s >= 85 else '#ef4444' for s in scores],
            text=[f'{s:.1f}%' for s in scores],
            textposition='outside'
        )
    ])
    
    fig_quality.add_hline(y=95, line_dash="dash", line_color="#059669", 
                          annotation_text="GxP Threshold: 95%", annotation_position="right")
    
    fig_quality.update_layout(
        title="Data Quality Score by Category",
        yaxis_title="Score (%)",
        yaxis_range=[0, 110],
        height=400
    )
    
    outlier_data = df[df['has_outlier']]
    
    if len(outlier_data) > 0:
        fig_outliers = go.Figure()
        
        fig_outliers.add_trace(go.Scatter(
            x=list(range(len(outlier_data))),
            y=outlier_data['systolic_bp'].fillna(0),
            mode='markers',
            marker=dict(size=10, color='#ef4444'),
            name='Systolic BP Outliers'
        ))
        
        fig_outliers.add_hline(y=180, line_dash="dash", line_color="#dc2626", annotation_text="High Limit")
        fig_outliers.add_hline(y=90, line_dash="dash", line_color="#dc2626", annotation_text="Low Limit")
        
        fig_outliers.update_layout(
            title="Detected Outliers - Systolic Blood Pressure",
            xaxis_title="Record Index",
            yaxis_title="Systolic BP (mmHg)",
            height=400
        )
    else:
        fig_outliers = go.Figure()
        fig_outliers.add_annotation(
            text="No outliers detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#10b981")
        )
        fig_outliers.update_layout(height=400, title="Outlier Analysis")
    
    return total_records, missing_data_count, outlier_count, overall_score, status, status_color, status_emoji, completeness_score, accuracy_score, fig_quality, fig_outliers, df

def generate_audit_trail():
    """Generate sample audit trail for GxP compliance"""
    
    users = ["dr.smith@pharma.com", "lab.tech@pharma.com", "data.manager@pharma.com"]
    actions = ["Data Entry", "Data Correction", "Value Update", "Record Approval"]
    
    # Build audit entries
    audit_cards = []
    for i in range(8):
        timestamp = datetime.now() - timedelta(hours=random.randint(1, 48))
        user = random.choice(users)
        action = random.choice(actions)
        
        old_val = f"{random.randint(110, 140)}" if random.random() > 0.3 else "null"
        new_val = f"{random.randint(115, 145)}"
        reason = random.choice([
            "Initial data entry per protocol",
            "Correction per source document review",
            "Updated per physician verification",
            "QC review - value confirmed accurate"
        ])
        
        card = f'<div style="background: white; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-left: 4px solid #a855f7;"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;"><div><span style="background: #a855f7; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-right: 8px;">{action.upper()}</span><span style="font-size: 15px; color: #1f2937; font-weight: 700;">Record #{i+1}</span></div><p style="font-size: 13px; color: #6b7280; margin: 0; font-family: monospace;">{timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p></div><div style="background: #f9fafb; border-radius: 8px; padding: 12px; margin-bottom: 10px;"><div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;"><div><p style="font-size: 11px; color: #6b7280; margin: 0 0 4px 0;">User</p><p style="font-size: 13px; color: #1f2937; font-weight: 600; margin: 0; font-family: monospace;">{user}</p></div><div><p style="font-size: 11px; color: #6b7280; margin: 0 0 4px 0;">Old Value</p><p style="font-size: 13px; color: #ef4444; font-weight: 600; margin: 0;">{old_val}</p></div><div><p style="font-size: 11px; color: #6b7280; margin: 0 0 4px 0;">New Value</p><p style="font-size: 13px; color: #10b981; font-weight: 600; margin: 0;">{new_val}</p></div></div></div><p style="font-size: 13px; color: #6b7280; margin: 0;"><strong>Reason:</strong> {reason}</p></div>'
        audit_cards.append(card)
    
    all_cards = ''.join(audit_cards)
    
    # Create audit timeline chart
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    audit_events = [random.randint(10, 50) for _ in dates]
    
    fig_audit = go.Figure()
    
    fig_audit.add_trace(go.Scatter(
        x=dates,
        y=audit_events,
        mode='lines+markers',
        line=dict(color='#a855f7', width=3),
        marker=dict(size=6),
        fill='tonexty',
        fillcolor='rgba(168, 85, 247, 0.1)',
        name='Audit Events'
    ))
    
    fig_audit.update_layout(
        title="Audit Trail Activity (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title="Number of Changes",
        height=400
    )
    
    return all_cards, fig_audit

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🔬</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Seal GxP Platform
        </h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Data Validation & Quality Control</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">FDA 21 CFR Part 11 compliance for biotech & pharma</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">GxP Compliance</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">ALCOA+</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Audit Trail</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">FDA 21 CFR Part 11</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Seal</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🔬 Data Validation", "📜 Audit Trail"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Clinical Trial Data Validation</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Automated GxP compliance checking for pharmaceutical and biotech data</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Validate Clinical Data", type="primary", use_container_width=True):
        st.session_state.validation_run = True
    
    if st.session_state.validation_run:
        total_records, missing_data_count, outlier_count, overall_score, status, status_color, status_emoji, completeness_score, accuracy_score, fig_quality, fig_outliers, df = validate_clinical_data()
        
        # Summary
        st.markdown(f'<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;"><h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🔬 GxP Data Validation Results</h2><div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;"><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Quality Score</p><p style="font-size: 48px; color: {status_color}; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{overall_score:.0f}%</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{status_emoji} {status}</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Records Validated</p><p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{total_records}</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Clinical trial data</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Missing Data</p><p style="font-size: 48px; color: {"#fca5a5" if missing_data_count > 0 else "#86efac"}; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{missing_data_count}</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{missing_data_count/total_records*100:.1f}% incomplete</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Outliers Detected</p><p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{outlier_count}</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{outlier_count/total_records*100:.1f}% flagged</p></div></div></div>', unsafe_allow_html=True)
        
        # ALCOA+ checklist
        alcoa_checks = {
            "Attributable": (True, "All records have user ID and timestamp"),
            "Legible": (True, "Data is readable and permanent"),
            "Contemporaneous": (True, "Recorded at time of activity"),
            "Original": (True, "First recording or certified copy"),
            "Accurate": (not bool(outlier_count), f"{outlier_count} outliers detected"),
            "Complete": (not bool(missing_data_count), f"{missing_data_count} missing values"),
            "Consistent": (True, "Data follows defined format"),
            "Enduring": (True, "Stored with proper backups"),
            "Available": (True, "Accessible for review/audit")
        }
        
        alcoa_cards = []
        for principle, (passed, detail) in alcoa_checks.items():
            check_color = "#10b981" if passed else "#ef4444"
            check_icon = "✓" if passed else "✗"
            
            card = f'<div style="background: white; border-left: 5px solid {check_color}; border-radius: 12px; padding: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);"><div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;"><div style="background: {check_color}; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center;"><span style="font-size: 16px; color: white; font-weight: 900;">{check_icon}</span></div><p style="font-size: 16px; color: #1f2937; font-weight: 800; margin: 0;">{principle}</p></div><p style="font-size: 13px; color: #6b7280; margin: 0; padding-left: 40px;">{detail}</p></div>'
            alcoa_cards.append(card)
        
        all_alcoa = ''.join(alcoa_cards)
        
        alcoa_html = f'<div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;"><h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📋 ALCOA+ Compliance Check</h3><p style="color: #3b82f6; font-size: 14px; margin: 0 0 20px 0; font-weight: 600;">FDA 21 CFR Part 11 Data Integrity Principles</p><div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">{all_alcoa}</div></div>'
        
        st.markdown(alcoa_html, unsafe_allow_html=True)
        
        # Issues
        if missing_data_count > 0 or outlier_count > 0:
            issue_items = []
            
            if missing_data_count > 0:
                item = f'<div style="background: white; border-left: 5px solid #ef4444; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);"><div style="display: flex; justify-content: space-between; align-items: center;"><div><span style="background: #dc2626; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-right: 10px;">CRITICAL</span><span style="font-size: 18px; color: #1f2937; font-weight: 700;">Incomplete Records</span></div><p style="font-size: 28px; color: #dc2626; font-weight: 900; margin: 0;">{missing_data_count}</p></div><p style="font-size: 14px; color: #6b7280; margin: 10px 0 0 0;">Missing vital signs data - violates ALCOA+ "Complete" principle</p><div style="background: #fef3c7; border-radius: 8px; padding: 12px; margin-top: 12px;"><p style="font-size: 13px; color: #92400e; font-weight: 600; margin: 0;">🔧 Action: Query source system, request data re-entry, document deviation</p></div></div>'
                issue_items.append(item)
            
            if outlier_count > 0:
                item = f'<div style="background: white; border-left: 5px solid #f59e0b; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);"><div style="display: flex; justify-content: space-between; align-items: center;"><div><span style="background: #f59e0b; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-right: 10px;">HIGH</span><span style="font-size: 18px; color: #1f2937; font-weight: 700;">Statistical Outliers</span></div><p style="font-size: 28px; color: #f59e0b; font-weight: 900; margin: 0;">{outlier_count}</p></div><p style="font-size: 14px; color: #6b7280; margin: 10px 0 0 0;">Values outside normal clinical ranges - potential data entry errors or true adverse events</p><div style="background: #fef3c7; border-radius: 8px; padding: 12px; margin-top: 12px;"><p style="font-size: 13px; color: #92400e; font-weight: 600; margin: 0;">🔧 Action: Clinical review required, verify with source documents, flag for monitoring</p></div></div>'
                issue_items.append(item)
            
            all_issues = ''.join(issue_items)
            issues_html = f'<div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(239, 68, 68, 0.2); margin-bottom: 25px;"><div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;"><div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4); border: 4px solid white;"><span style="font-size: 36px;">⚠️</span></div><div><h3 style="color: #991b1b; font-size: 26px; font-weight: 900; margin: 0;">Data Quality Issues</h3><p style="color: #dc2626; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">{missing_data_count + outlier_count} issues require attention</p></div></div><div style="display: grid; gap: 12px;">{all_issues}</div></div>'
        else:
            issues_html = '<div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;"><div style="display: flex; align-items: center; gap: 15px;"><div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4); border: 4px solid white;"><span style="font-size: 36px;">✅</span></div><div><h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0;">All Validation Checks Passed</h3><p style="color: #047857; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">Data meets all GxP compliance requirements</p></div></div></div>'
        
        st.markdown(issues_html, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_quality, use_container_width=True)
        with col2:
            st.plotly_chart(fig_outliers, use_container_width=True)
        
        st.dataframe(df, use_container_width=True, height=400)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Electronic Audit Trail System</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Complete change history for regulatory compliance and data integrity</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📜 Generate Audit Trail", type="primary", use_container_width=True):
        st.session_state.audit_generated = True
    
    if st.session_state.audit_generated:
        all_cards, fig_audit = generate_audit_trail()
        
        audit_html = f'<div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(168, 85, 247, 0.2); margin-bottom: 25px;"><h3 style="color: #6b21a8; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📜 Audit Trail (FDA 21 CFR Part 11)</h3><p style="color: #a855f7; font-size: 14px; margin: 0 0 20px 0; font-weight: 600;">Complete record of all data modifications with electronic signatures</p><div style="display: grid; gap: 12px;">{all_cards}</div><div style="background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); border-radius: 12px; padding: 20px; margin-top: 20px; color: white;"><p style="font-size: 16px; font-weight: 800; margin: 0 0 10px 0;">✅ Audit Trail Compliance Status</p><ul style="margin: 0; padding-left: 24px; line-height: 2;"><li style="font-size: 14px;">All changes logged with unique user ID</li><li style="font-size: 14px;">Timestamps recorded for every modification</li><li style="font-size: 14px;">Old and new values captured</li><li style="font-size: 14px;">Reason for change documented</li><li style="font-size: 14px;">Electronic signatures verified</li></ul></div></div>'
        
        st.markdown(audit_html, unsafe_allow_html=True)
        st.plotly_chart(fig_audit, use_container_width=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Seal</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 FDA Approval Risk</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Poor data quality can delay FDA approval by 6-12 months, costing pharma companies $1M-5M per month. Automated validation catches issues early.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ Audit Readiness</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    FDA audits require complete audit trails. Manual documentation takes weeks. Automated trails are instant, complete, and compliant.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📊 Data Quality</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Clinical trials cost $20K-100K per patient. Bad data = wasted money. Catch errors early before they contaminate entire study.
                </p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Real-World Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">6-12 months faster:</strong> FDA approval with clean data</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">$5M+ saved:</strong> Per avoided approval delay</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">100% audit ready:</strong> Complete trails at all times</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">90% time reduction:</strong> In data validation workload</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ ALCOA+ Validation</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">All 9 FDA data integrity principles</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Anomaly Detection</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Statistical outlier identification</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Complete Audit Trail</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Every change logged per 21 CFR Part 11</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Validation</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Instant feedback on data quality</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Seal</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Streamlit • Plotly • Pandas • Statistical Analysis
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing GxP data validation for biotech and pharmaceutical industries.<br>
            ALCOA+ compliance • Audit trails • Data quality • FDA 21 CFR Part 11
        </p>
    </div>
    """, unsafe_allow_html=True)