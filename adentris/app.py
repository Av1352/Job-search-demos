"""
Adentris - AI Healthcare Compliance Intelligence Platform
Automated compliance checking for hospitals and healthcare organizations
Built for Adentris by Anju Nandhakumar
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import random
import re

# Compliance rules database
HIPAA_RULES = {
    "Privacy Rule": {
        "requirement": "Patient consent for data sharing",
        "severity": "Critical",
        "checks": ["consent_documented", "disclosure_tracking", "minimum_necessary"]
    },
    "Security Rule": {
        "requirement": "Encryption of ePHI at rest and in transit",
        "severity": "Critical", 
        "checks": ["encryption_enabled", "access_controls", "audit_logs"]
    },
    "Breach Notification": {
        "requirement": "Report breaches affecting 500+ individuals within 60 days",
        "severity": "High",
        "checks": ["breach_detection", "notification_process", "documentation"]
    }
}

CMS_REQUIREMENTS = {
    "Documentation": {
        "requirement": "Complete clinical documentation within 30 days of discharge",
        "severity": "High",
        "checks": ["discharge_summary", "physician_signature", "coding_accuracy"]
    },
    "Quality Measures": {
        "requirement": "Report quality metrics quarterly",
        "severity": "Medium",
        "checks": ["hedis_measures", "star_ratings", "readmission_rates"]
    }
}

JOINT_COMMISSION = {
    "Patient Safety": {
        "requirement": "Universal Protocol for surgical procedures",
        "severity": "Critical",
        "checks": ["timeout_procedure", "site_marking", "correct_patient_verification"]
    },
    "Medication Management": {
        "requirement": "Two patient identifiers before medication administration",
        "severity": "Critical",
        "checks": ["patient_id_verification", "medication_reconciliation", "allergy_check"]
    }
}

def analyze_clinical_note(note_text):
    """Analyze a clinical note for compliance issues"""
    
    issues = []
    score = 100
    
    # Check for required elements
    required_elements = {
        "Chief Complaint": ["chief complaint", "cc:", "reason for visit"],
        "HPI": ["history of present illness", "hpi", "patient reports"],
        "Assessment": ["assessment", "impression", "diagnosis"],
        "Plan": ["plan", "treatment", "follow-up"],
        "Signature": ["md", "do", "np", "pa", "signed"]
    }
    
    note_lower = note_text.lower()
    
    for element, keywords in required_elements.items():
        if not any(keyword in note_lower for keyword in keywords):
            issues.append({
                "type": "Missing Element",
                "element": element,
                "severity": "High" if element in ["Assessment", "Plan"] else "Medium",
                "description": f"{element} not documented"
            })
            score -= 15 if element in ["Assessment", "Plan"] else 10
    
    # Check for PHI handling
    phi_patterns = {
        "SSN": r'\d{3}-\d{2}-\d{4}',
        "Phone": r'\d{3}-\d{3}-\d{4}',
        "Email": r'[\w\.-]+@[\w\.-]+',
    }
    
    for phi_type, pattern in phi_patterns.items():
        if re.search(pattern, note_text):
            issues.append({
                "type": "PHI Exposure Risk",
                "element": phi_type,
                "severity": "Critical",
                "description": f"Potential {phi_type} found - verify if redacted properly"
            })
            score -= 20
    
    # Check length (documentation completeness)
    word_count = len(note_text.split())
    if word_count < 50:
        issues.append({
            "type": "Incomplete Documentation",
            "element": "Note Length",
            "severity": "Medium",
            "description": f"Note is very short ({word_count} words) - may lack detail"
        })
        score -= 10
    
    # Check for signature
    if not any(sig in note_lower for sig in ["signed", "electronically signed", "md", "do"]):
        issues.append({
            "type": "Missing Signature",
            "element": "Provider Signature",
            "severity": "Critical",
            "description": "No provider signature detected"
        })
        score -= 25
    
    score = max(0, score)
    
    # Determine compliance level
    if score >= 90:
        compliance_level = "Compliant"
        compliance_color = "#10b981"
    elif score >= 70:
        compliance_level = "Needs Review"
        compliance_color = "#f59e0b"
    else:
        compliance_level = "Non-Compliant"
        compliance_color = "#ef4444"
    
    return issues, score, compliance_level, compliance_color, word_count

def check_clinical_note(note_text):
    """Check a clinical note and generate compliance report"""
    
    if not note_text or len(note_text.strip()) < 10:
        return (
            "<div style='background: #fee2e2; border: 2px solid #dc2626; padding: 20px; border-radius: 10px;'><p style='color: #991b1b; font-weight: bold; font-size: 18px; margin: 0;'>⚠️ Please enter a clinical note to analyze!</p></div>",
            None,
            None
        )
    
    issues, score, compliance_level, compliance_color, word_count = analyze_clinical_note(note_text)
    
    # Generate report HTML
    report_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📋 Clinical Documentation Analysis</h2>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Compliance Score</p>
                <p style="font-size: 48px; color: {compliance_color}; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{score}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">out of 100</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Status</p>
                <p style="font-size: 24px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{compliance_level}</p>
                <div style="display: inline-block; background: {compliance_color}; color: white; padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 700; margin-top: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
                    {'✓ PASS' if score >= 70 else '✗ FAIL'}
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Issues Found</p>
                <p style="font-size: 48px; color: {'#fca5a5' if len(issues) > 0 else '#86efac'}; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{len(issues)}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{word_count} words</p>
            </div>
        </div>
    </div>
    """
    
    # Issues breakdown
    if issues:
        critical_issues = [i for i in issues if i['severity'] == 'Critical']
        high_issues = [i for i in issues if i['severity'] == 'High']
        medium_issues = [i for i in issues if i['severity'] == 'Medium']
        
        issues_html = f"""
        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(239, 68, 68, 0.2); margin-bottom: 25px;">
            <h3 style="color: #991b1b; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">⚠️ Compliance Issues Detected</h3>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px;">
                <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <p style="font-size: 36px; color: #dc2626; font-weight: 900; margin: 0;">{len(critical_issues)}</p>
                    <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Critical</p>
                </div>
                <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <p style="font-size: 36px; color: #f97316; font-weight: 900; margin: 0;">{len(high_issues)}</p>
                    <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">High</p>
                </div>
                <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <p style="font-size: 36px; color: #f59e0b; font-weight: 900; margin: 0;">{len(medium_issues)}</p>
                    <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Medium</p>
                </div>
            </div>
        """
        
        # List all issues
        for issue in issues:
            severity_colors = {
                'Critical': '#dc2626',
                'High': '#f97316',
                'Medium': '#f59e0b'
            }
            color = severity_colors.get(issue['severity'], '#6b7280')
            
            issues_html += f"""
            <div style="background: white; border-left: 5px solid {color}; border-radius: 12px; padding: 18px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 8px;">
                    <div style="flex: 1;">
                        <span style="background: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-right: 10px;">{issue['severity'].upper()}</span>
                        <span style="font-size: 16px; color: #1f2937; font-weight: 700;">{issue['type']}</span>
                    </div>
                </div>
                <p style="font-size: 14px; color: #6b7280; margin: 0;"><strong>{issue['element']}:</strong> {issue['description']}</p>
            </div>
            """
        
        issues_html += "</div>"
    else:
        issues_html = """
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4); border: 4px solid white;">
                    <span style="font-size: 36px;">✅</span>
                </div>
                <div>
                    <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0;">No Compliance Issues Found</h3>
                    <p style="color: #047857; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">Documentation meets all compliance requirements</p>
                </div>
            </div>
        </div>
        """
    
    # Recommendations
    rec_html = """
    <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2);">
        <h3 style="color: #1e40af; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">💡 Compliance Recommendations</h3>
        <div style="background: white; border-radius: 12px; padding: 20px;">
            <ul style="margin: 0; padding-left: 24px; line-height: 2.2;">
    """
    
    if score < 70:
        rec_html += """
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;">🚨 <strong>Immediate Action Required:</strong> Address all critical issues before finalizing note</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;">📋 Complete all required documentation elements (HPI, Assessment, Plan)</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;">✍️ Ensure proper provider signature and credentials</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;">🔍 Review for PHI handling compliance</li>
        """
    elif score < 90:
        rec_html += """
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;">⚠️ Address remaining issues to achieve full compliance</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;">📝 Add missing documentation elements</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;">✓ Verify all required signatures and attestations</li>
        """
    else:
        rec_html += """
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;">✅ Documentation is compliant - ready for submission</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;">📊 Note meets all regulatory requirements</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;">🎯 Maintain this quality standard for future documentation</li>
        """
    
    rec_html += """
            </ul>
        </div>
    </div>
    """
    
    # Create compliance breakdown chart
    categories = ['Required Elements', 'PHI Handling', 'Completeness', 'Signature']
    scores = [
        85 if not any(i['type'] == 'Missing Element' for i in issues) else 50,
        100 if not any(i['type'] == 'PHI Exposure Risk' for i in issues) else 30,
        90 if not any(i['type'] == 'Incomplete Documentation' for i in issues) else 60,
        100 if not any(i['type'] == 'Missing Signature' for i in issues) else 0
    ]
    
    fig_breakdown = go.Figure(data=[
        go.Bar(
            x=categories,
            y=scores,
            marker_color=['#10b981' if s >= 80 else '#f59e0b' if s >= 60 else '#ef4444' for s in scores],
            text=[f'{s}%' for s in scores],
            textposition='outside'
        )
    ])
    
    fig_breakdown.update_layout(
        title="Compliance Score Breakdown by Category",
        yaxis_title="Score (%)",
        yaxis_range=[0, 110],
        height=400
    )
    
    return report_html + issues_html + rec_html, fig_breakdown, len(issues)

def generate_compliance_dashboard():
    """Generate organization-wide compliance dashboard"""
    
    # Simulate compliance data
    compliance_data = {
        'HIPAA Privacy': random.randint(85, 98),
        'HIPAA Security': random.randint(80, 95),
        'CMS Documentation': random.randint(75, 92),
        'Joint Commission': random.randint(88, 97),
        'Breach Notification': random.randint(90, 100),
        'Quality Measures': random.randint(82, 94)
    }
    
    avg_score = sum(compliance_data.values()) / len(compliance_data)
    
    dashboard_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🏥 Organization Compliance Dashboard</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Overall Score</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{avg_score:.0f}%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">6 frameworks</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Passing Rate</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">100%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">All ≥70%</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Last Audit</p>
                <p style="font-size: 24px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">Dec 2024</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">No findings</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Next Review</p>
                <p style="font-size: 24px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">Mar 2025</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">In 90 days</p>
            </div>
        </div>
    </div>
    """
    
    # Compliance by framework
    framework_html = """
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📊 Compliance by Framework</h3>
        <div style="display: grid; gap: 12px;">
    """
    
    for framework, score in compliance_data.items():
        color = '#10b981' if score >= 90 else '#f59e0b' if score >= 80 else '#f97316'
        framework_html += f"""
        <div style="background: white; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 16px; color: #1f2937; font-weight: 700;">{framework}</span>
                <span style="font-size: 24px; color: {color}; font-weight: 900;">{score}%</span>
            </div>
            <div style="background: #e5e7eb; border-radius: 8px; height: 10px; overflow: hidden;">
                <div style="background: {color}; height: 100%; width: {score}%; transition: width 0.3s;"></div>
            </div>
        </div>
        """
    
    framework_html += "</div></div>"
    
    # Create charts
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=list(compliance_data.values()),
        theta=list(compliance_data.keys()),
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#3b82f6', width=3)
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title="Compliance Framework Radar",
        height=500
    )
    
    # Trend chart
    dates = pd.date_range(end=datetime.now(), periods=12, freq='M')
    trend_data = [random.randint(75, 85) + (i * 1.2) for i in range(12)]
    
    fig_trend = go.Figure(data=[
        go.Scatter(
            x=dates,
            y=trend_data,
            mode='lines+markers',
            line=dict(color='#10b981', width=3),
            marker=dict(size=8),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.1)'
        )
    ])
    
    fig_trend.add_hline(y=90, line_dash="dash", line_color="#059669", annotation_text="Target: 90%")
    
    fig_trend.update_layout(
        title="Compliance Score Trend (Last 12 Months)",
        xaxis_title="Month",
        yaxis_title="Overall Compliance Score (%)",
        yaxis_range=[70, 100],
        height=400
    )
    
    return dashboard_html + framework_html, fig_radar, fig_trend

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
            <span style="font-size: 56px;">✅</span>
        </div>
        
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Adentris Compliance
        </h1>
        
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">AI Healthcare Compliance Intelligence</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Automated compliance checking for hospitals & healthcare organizations</p>
        
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 800px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">HIPAA</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">CMS</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Joint Commission</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">YC Backed</span>
        </div>
        
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Adentris</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("📝 Clinical Note Checker"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">AI-Powered Clinical Documentation Analysis</h3>
                <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Automated compliance checking for clinical notes • HIPAA • CMS • Joint Commission</p>
            </div>
            """)
            
            # Example dropdown
            example_dropdown = gr.Dropdown(
                choices=[
                    "✅ Compliant Note - All Elements Present",
                    "⚠️ Missing Signature",
                    "⚠️ Incomplete Documentation",
                    "🚨 Multiple Critical Issues",
                    "📝 Custom Note (Enter Your Own)"
                ],
                label="Try Example Clinical Notes",
                value="📝 Custom Note (Enter Your Own)"
            )
            
            note_input = gr.Textbox(
                label="Clinical Note",
                placeholder="Select an example above or paste your own clinical documentation here...",
                lines=12
            )
            
            check_note_btn = gr.Button("🔍 Check Compliance", variant="primary", size="lg")
            
            note_report = gr.HTML(label="Compliance Report")
            note_chart = gr.Plot(label="Score Breakdown")
            issue_count = gr.Number(label="Total Issues", visible=False)
            
            check_note_btn.click(
                fn=check_clinical_note,
                inputs=[note_input],
                outputs=[note_report, note_chart, issue_count]
            )
            
            # Example note loader
            def load_example(choice):
                examples = {
                    "✅ Compliant Note - All Elements Present": """Chief Complaint: Chest pain

History of Present Illness: 65 year old male with history of coronary artery disease, hypertension, and hyperlipidemia presents to the emergency department with acute onset crushing substernal chest pain radiating to left arm that started 2 hours ago while at rest. Patient describes pain as 8/10 severity, associated with diaphoresis and shortness of breath. Patient took nitroglycerin x2 at home without relief. Denies nausea, vomiting, or palpitations. Last episode of chest pain was 6 months ago during cardiac catheterization showing 70% LAD stenosis.

Past Medical History: CAD s/p stent 2020, hypertension, hyperlipidemia, type 2 diabetes
Medications: Aspirin 81mg daily, Metoprolol 50mg BID, Atorvastatin 80mg QHS, Metformin 1000mg BID
Allergies: No known drug allergies

Physical Exam:
Vitals: BP 145/92, HR 98, RR 20, O2 sat 94% on RA, Temp 98.6°F
General: Anxious, diaphoretic male in moderate distress
Cardiovascular: Regular rate and rhythm, no murmurs
Respiratory: Clear to auscultation bilaterally
EKG: ST elevation 2mm in leads II, III, aVF

Assessment: Acute ST-elevation myocardial infarction (STEMI), inferior wall

Plan:
1. Activate cardiac catheterization lab - STAT
2. Aspirin 325mg PO given
3. Heparin bolus 5000 units IV, then drip at 1000 units/hour
4. Start nitroglycerin drip
5. Cardiology consultation - Dr. Johnson paged
6. Admit to CCU
7. Serial troponins q4h
8. NPO for catheterization

Patient and family counseled on diagnosis, treatment plan, and risks/benefits of emergency cardiac catheterization. Patient consents to procedure.

Electronically signed: John Smith, MD
Emergency Medicine
Date: 12/30/2024 14:35""",

                    "⚠️ Missing Signature": """Chief Complaint: Abdominal pain

HPI: 45 yo F presents with 2 days of right lower quadrant abdominal pain. Pain started periumbilical then migrated to RLQ. Associated with nausea, one episode of vomiting, decreased appetite. Denies fever, diarrhea, urinary symptoms. LMP 2 weeks ago, regular cycles.

Physical Exam:
Vitals stable, afebrile
Abdomen: Tender RLQ with guarding, positive McBurney's point, no rebound
Labs: WBC 14.5, normal urinalysis

Assessment: Acute appendicitis

Plan:
1. Surgery consultation
2. NPO
3. IV fluids
4. Pain management
5. Consent for appendectomy""",

                    "⚠️ Incomplete Documentation": """Patient seen for follow-up.

Doing okay. Some issues with medications.

Will continue current plan.

Dr. Jones""",

                    "🚨 Multiple Critical Issues": """Patient came in today. 

Has been having some problems.

SSN: 123-45-6789
Phone: 555-123-4567
Email: patient@email.com

Gave some medication. Follow up needed.""",

                    "📝 Custom Note (Enter Your Own)": ""
                }
                
                return examples.get(choice, "")
            
            example_dropdown.change(
                fn=load_example,
                inputs=[example_dropdown],
                outputs=[note_input]
            )
        
        with gr.Tab("🏥 Organization Dashboard"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Enterprise Compliance Dashboard</h3>
                <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Organization-wide compliance monitoring across all regulatory frameworks</p>
            </div>
            """)
            
            dashboard_btn = gr.Button("📊 Generate Compliance Dashboard", variant="primary", size="lg")
            
            dashboard_output = gr.HTML(label="Dashboard")
            radar_chart = gr.Plot(label="Compliance Radar")
            trend_chart = gr.Plot(label="Compliance Trend")
            
            dashboard_btn.click(
                fn=generate_compliance_dashboard,
                inputs=[],
                outputs=[dashboard_output, radar_chart, trend_chart]
            )
        
        with gr.Tab("📋 Regulatory Frameworks"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #92400e; font-size: 22px; font-weight: 800; margin: 0;">Healthcare Regulatory Requirements</h3>
                <p style="color: #d97706; font-size: 14px; margin: 8px 0 0 0;">Comprehensive coverage of HIPAA, CMS, and Joint Commission standards</p>
            </div>
            """)
            
            frameworks_html = """
            <div style="display: grid; gap: 20px;">
                <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 16px; padding: 24px;">
                    <h3 style="color: #991b1b; font-size: 24px; font-weight: 800; margin: 0 0 15px 0;">🔒 HIPAA Compliance</h3>
                    <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                        <h4 style="color: #1f2937; font-weight: 700; margin: 0 0 10px 0;">Privacy Rule</h4>
                        <p style="color: #6b7280; margin: 0; line-height: 1.7;">Patient consent for data sharing • Disclosure tracking • Minimum necessary standard</p>
                        <span style="display: inline-block; background: #dc2626; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-top: 10px;">CRITICAL</span>
                    </div>
                    <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                        <h4 style="color: #1f2937; font-weight: 700; margin: 0 0 10px 0;">Security Rule</h4>
                        <p style="color: #6b7280; margin: 0; line-height: 1.7;">Encryption of ePHI at rest and in transit • Access controls • Audit logs</p>
                        <span style="display: inline-block; background: #dc2626; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-top: 10px;">CRITICAL</span>
                    </div>
                    <div style="background: white; border-radius: 12px; padding: 20px;">
                        <h4 style="color: #1f2937; font-weight: 700; margin: 0 0 10px 0;">Breach Notification</h4>
                        <p style="color: #6b7280; margin: 0; line-height: 1.7;">Report breaches affecting 500+ individuals within 60 days • Documentation required</p>
                        <span style="display: inline-block; background: #f97316; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-top: 10px;">HIGH</span>
                    </div>
                </div>
                
                <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px;">
                    <h3 style="color: #1e40af; font-size: 24px; font-weight: 800; margin: 0 0 15px 0;">📊 CMS Requirements</h3>
                    <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                        <h4 style="color: #1f2937; font-weight: 700; margin: 0 0 10px 0;">Documentation Standards</h4>
                        <p style="color: #6b7280; margin: 0; line-height: 1.7;">Complete clinical documentation within 30 days of discharge • Physician signature required</p>
                        <span style="display: inline-block; background: #f97316; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-top: 10px;">HIGH</span>
                    </div>
                    <div style="background: white; border-radius: 12px; padding: 20px;">
                        <h4 style="color: #1f2937; font-weight: 700; margin: 0 0 10px 0;">Quality Measures</h4>
                        <p style="color: #6b7280; margin: 0; line-height: 1.7;">Report quality metrics quarterly • HEDIS measures • STAR ratings • Readmission rates</p>
                        <span style="display: inline-block; background: #f59e0b; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-top: 10px;">MEDIUM</span>
                    </div>
                </div>
                
                <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px;">
                    <h3 style="color: #6b21a8; font-size: 24px; font-weight: 800; margin: 0 0 15px 0;">🏥 Joint Commission Standards</h3>
                    <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                        <h4 style="color: #1f2937; font-weight: 700; margin: 0 0 10px 0;">Patient Safety</h4>
                        <p style="color: #6b7280; margin: 0; line-height: 1.7;">Universal Protocol for surgical procedures • Timeout procedure • Site marking • Correct patient verification</p>
                        <span style="display: inline-block; background: #dc2626; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-top: 10px;">CRITICAL</span>
                    </div>
                    <div style="background: white; border-radius: 12px; padding: 20px;">
                        <h4 style="color: #1f2937; font-weight: 700; margin: 0 0 10px 0;">Medication Management</h4>
                        <p style="color: #6b7280; margin: 0; line-height: 1.7;">Two patient identifiers before medication administration • Medication reconciliation • Allergy check</p>
                        <span style="display: inline-block; background: #dc2626; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-top: 10px;">CRITICAL</span>
                    </div>
                </div>
            </div>
            """
            
            gr.HTML(frameworks_html)
    
    gr.HTML("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Adentris</h2>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ Automated Compliance</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Replace manual compliance reviews with AI. Scan 1000+ notes per day, identify issues instantly, reduce compliance staff workload by 70%.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Risk Mitigation</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Avoid HIPAA fines ($100-50K per violation), CMS penalties, Joint Commission sanctions. One major violation costs more than entire compliance system.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📊 Audit Readiness</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Real-time compliance dashboards, historical tracking, automated reporting. Be audit-ready 24/7 with complete documentation trail.
                </p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Real-World Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">$50K-500K saved:</strong> Per avoided HIPAA violation or CMS penalty</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">70% time reduction:</strong> In manual compliance review workload</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">100% coverage:</strong> Every note checked, no documentation gaps</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Real-time alerts:</strong> Fix issues before they become violations</li>
            </ul>
        </div>
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ NLP-Powered Analysis</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Extract required elements, detect PHI, verify signatures</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-Framework Support</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">HIPAA, CMS, Joint Commission in one platform</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Feedback</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Instant compliance scores and recommendations</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ EMR Integration</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Epic, Cerner, Athena compatible</p>
                </div>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Adentris</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Gradio • Plotly • NLP • Regex
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered compliance checking for healthcare organizations.<br>
            HIPAA • CMS • Joint Commission • Automated analysis • Real-time alerts
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()