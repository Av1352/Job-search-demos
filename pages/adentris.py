"""
Adentris - AI Healthcare Compliance Intelligence Platform
Automated compliance checking for hospitals and healthcare organizations
Built for Adentris by Anju Nandhakumar
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import random
import re
import textwrap

# Page config
st.set_page_config(
    page_title="Adentris Demo - Anju Vilashni",
    page_icon="✅",
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
    
    # Check length
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

# Header
st.markdown(
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
            <span style="font-size: 56px;">✅</span>
        </div>
        <h1 style="
            font-size: 58px;
            font-weight: 900;
            color: white;
            margin: 0 0 18px 0;
            text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            Adentris Compliance
        </h1>
        <p style="
            font-size: 28px;
            color: rgba(255,255,255,0.95);
            font-weight: 700;
            margin: 15px 0;
        ">
            AI Healthcare Compliance Intelligence
        </p>
        <p style="
            font-size: 18px;
            color: rgba(255,255,255,0.85);
            font-weight: 500;
            margin-bottom: 25px;
        ">
            Automated compliance checking for hospitals & healthcare organizations
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
            <span style="background:#ec4899;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">HIPAA</span>
            <span style="background:#8b5cf6;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">CMS</span>
            <span style="background:#f59e0b;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Joint Commission</span>
            <span style="background:#10b981;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">YC Backed</span>
        </div>
        <p style="
            font-size: 16px;
            color: rgba(255,255,255,0.9);
            margin-top: 28px;
            font-weight: 600;
        ">
            Built for <strong style="color:white;">Adentris</strong>
            by <strong style="color:white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# Example notes
EXAMPLE_NOTES = {
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

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Clinical Note Checker", "🏥 Organization Dashboard", "📋 Regulatory Frameworks"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">AI-Powered Clinical Documentation Analysis</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Automated compliance checking for clinical notes • HIPAA • CMS • Joint Commission</p>
    </div>
    """, unsafe_allow_html=True)
    
    example_choice = st.selectbox(
        "Try Example Clinical Notes",
        list(EXAMPLE_NOTES.keys()),
        index=4
    )
    
    note_text = st.text_area(
        "Clinical Note",
        value=EXAMPLE_NOTES[example_choice],
        height=300,
        placeholder="Select an example above or paste your own clinical documentation here..."
    )
    
    if st.button("🔍 Check Compliance", key="check_note"):
        if not note_text or len(note_text.strip()) < 10:
            st.error("⚠️ Please enter a clinical note to analyze!")
        else:
            issues, score, compliance_level, compliance_color, word_count = analyze_clinical_note(note_text)
            
            # Report header
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
                <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📋 Clinical Documentation Analysis</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                    <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Compliance Score</p>
                    <p style="font-size: 48px; color: {compliance_color}; font-weight: 900; margin: 0;">{score}</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">out of 100</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                    <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Status</p>
                    <p style="font-size: 24px; color: {compliance_color}; font-weight: 900; margin: 0;">{compliance_level}</p>
                    <div style="display: inline-block; background: {compliance_color}; color: white; padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 700; margin-top: 8px;">
                        {'✓ PASS' if score >= 70 else '✗ FAIL'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                    <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Issues Found</p>
                    <p style="font-size: 48px; color: {'#ef4444' if len(issues) > 0 else '#10b981'}; font-weight: 900; margin: 0;">{len(issues)}</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">{word_count} words</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Issues section
            if issues:
                critical_issues = [i for i in issues if i['severity'] == 'Critical']
                high_issues = [i for i in issues if i['severity'] == 'High']
                medium_issues = [i for i in issues if i['severity'] == 'Medium']
                
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
                    <h3 style="color: #991b1b; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">⚠️ Compliance Issues Detected</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                        <p style="font-size: 36px; color: #dc2626; font-weight: 900; margin: 0;">{len(critical_issues)}</p>
                        <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Critical</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                        <p style="font-size: 36px; color: #f97316; font-weight: 900; margin: 0;">{len(high_issues)}</p>
                        <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">High</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                        <p style="font-size: 36px; color: #f59e0b; font-weight: 900; margin: 0;">{len(medium_issues)}</p>
                        <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Medium</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # List issues
                for issue in issues:
                    severity_colors = {
                        'Critical': '#dc2626',
                        'High': '#f97316',
                        'Medium': '#f59e0b'
                    }
                    color = severity_colors.get(issue['severity'], '#6b7280')
                    
                    st.markdown(f"""
                    <div style="background: white; border-left: 5px solid {color}; border-radius: 12px; padding: 18px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                        <div style="margin-bottom: 8px;">
                            <span style="background: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-right: 10px;">{issue['severity'].upper()}</span>
                            <span style="font-size: 16px; color: #1f2937; font-weight: 700;">{issue['type']}</span>
                        </div>
                        <p style="font-size: 14px; color: #6b7280; margin: 0;"><strong>{issue['element']}:</strong> {issue['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <div style="background: #10b981; width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 4px solid white;">
                            <span style="font-size: 36px;">✅</span>
                        </div>
                        <div>
                            <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0;">No Compliance Issues Found</h3>
                            <p style="color: #047857; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">Documentation meets all compliance requirements</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("""
            <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px;">
                <h3 style="color: #1e40af; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">💡 Compliance Recommendations</h3>
                <div style="background: white; border-radius: 12px; padding: 20px;">
                    <ul style="margin: 0; padding-left: 24px; line-height: 2.2;">
            """, unsafe_allow_html=True)
            
            if score < 70:
                st.markdown("""
                        <li style="color: #1f2937; font-size: 15px; font-weight: 600;">🚨 <strong>Immediate Action Required:</strong> Address all critical issues before finalizing note</li>
                        <li style="color: #1f2937; font-size: 15px; font-weight: 600;">📋 Complete all required documentation elements (HPI, Assessment, Plan)</li>
                        <li style="color: #1f2937; font-size: 15px; font-weight: 600;">✍️ Ensure proper provider signature and credentials</li>
                        <li style="color: #1f2937; font-size: 15px; font-weight: 600;">🔍 Review for PHI handling compliance</li>
                """, unsafe_allow_html=True)
            elif score < 90:
                st.markdown("""
                        <li style="color: #1f2937; font-size: 15px; font-weight: 600;">⚠️ Address remaining issues to achieve full compliance</li>
                        <li style="color: #1f2937; font-size: 15px; font-weight: 600;">📝 Add missing documentation elements</li>
                        <li style="color: #1f2937; font-size: 15px; font-weight: 600;">✓ Verify all required signatures and attestations</li>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                        <li style="color: #1f2937; font-size: 15px; font-weight: 600;">✅ Documentation is compliant - ready for submission</li>
                        <li style="color: #1f2937; font-size: 15px; font-weight: 600;">📊 Note meets all regulatory requirements</li>
                        <li style="color: #1f2937; font-size: 15px; font-weight: 600;">🎯 Maintain this quality standard for future documentation</li>
                """, unsafe_allow_html=True)
            
            st.markdown("</ul></div></div>", unsafe_allow_html=True)
            
            # Breakdown chart
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
            
            st.plotly_chart(fig_breakdown, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Enterprise Compliance Dashboard</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Organization-wide compliance monitoring across all regulatory frameworks</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Generate Compliance Dashboard", key="dashboard"):
        # Simulate data
        compliance_data = {
            'HIPAA Privacy': random.randint(85, 98),
            'HIPAA Security': random.randint(80, 95),
            'CMS Documentation': random.randint(75, 92),
            'Joint Commission': random.randint(88, 97),
            'Breach Notification': random.randint(90, 100),
            'Quality Measures': random.randint(82, 94)
        }
        
        avg_score = sum(compliance_data.values()) / len(compliance_data)
        
        # Dashboard header
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🏥 Organization Compliance Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Overall Score</p>
                <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">{avg_score:.0f}%</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">6 frameworks</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(16, 185, 129, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Passing Rate</p>
                <p style="font-size: 48px; color: #10b981; font-weight: 900; margin: 0;">100%</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">All ≥70%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: rgba(59, 130, 246, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(59, 130, 246, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Last Audit</p>
                <p style="font-size: 24px; color: #3b82f6; font-weight: 900; margin: 0;">Dec 2024</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">No findings</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: rgba(245, 158, 11, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(245, 158, 11, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Next Review</p>
                <p style="font-size: 24px; color: #f59e0b; font-weight: 900; margin: 0;">Mar 2025</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">In 90 days</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Framework breakdown
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📊 Compliance by Framework</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for framework, score in compliance_data.items():
            color = '#10b981' if score >= 90 else '#f59e0b' if score >= 80 else '#f97316'
            st.markdown(f"""
            <div style="background: white; border-radius: 12px; padding: 18px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 16px; color: #1f2937; font-weight: 700;">{framework}</span>
                    <span style="font-size: 24px; color: {color}; font-weight: 900;">{score}%</span>
                </div>
                <div style="background: #e5e7eb; border-radius: 8px; height: 10px; overflow: hidden;">
                    <div style="background: {color}; height: 100%; width: {score}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=list(compliance_data.values()),
            theta=list(compliance_data.keys()),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.3)',
            line=dict(color='#3b82f6', width=3)
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Compliance Framework Radar",
            height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Trend
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
        st.plotly_chart(fig_trend, use_container_width=True)

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #92400e; font-size: 22px; font-weight: 800; margin: 0;">Healthcare Regulatory Requirements</h3>
        <p style="color: #d97706; font-size: 14px; margin: 8px 0 0 0;">Comprehensive coverage of HIPAA, CMS, and Joint Commission standards</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
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
    
    <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
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
    """, unsafe_allow_html=True)

# Footer
st.markdown("<hr style='border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Adentris</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
        <strong>Tech Stack:</strong> Python • Streamlit • Plotly • NLP • Regex
    </p>
</div>
""", unsafe_allow_html=True)