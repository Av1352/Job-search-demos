"""
Dioptra AI - Contract Intelligence Platform
Automated contract analysis and negotiation support
Built for Dioptra AI by Anju Nandhakumar
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import random
import re

# Page config
st.set_page_config(
    page_title="Dioptra AI Demo - Anju Vilashni",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main { background: white; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
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

# Sample contracts (shortened for brevity)
SAMPLE_CONTRACTS = {
    "SaaS Vendor Agreement": """MASTER SUBSCRIPTION AGREEMENT
Payment Terms: Client shall pay $50,000 annually, due Net 15 days.
Term: 3 years with auto-renewal. Early termination requires 6 months notice and 50% fee.
Liability: Vendor's liability capped at $10,000. Client's liability is unlimited.
IP: All improvements become Vendor's exclusive property.""",
    
    "Employment Agreement": """EMPLOYMENT AGREEMENT
Compensation: $150,000 base + 20% discretionary bonus.
At-will employment. Either party may terminate at any time.
Non-compete for 2 years post-termination in any state where Company operates.
Employee assigns all work product to Company."""
}

def analyze_contract(contract_text, contract_type):
    """Analyze contract for risks"""
    
    if not contract_text or len(contract_text.strip()) < 50:
        st.error("⚠️ Please enter a contract to analyze!")
        return
    
    contract_lower = contract_text.lower()
    
    # Extract info
    payment_match = re.search(r'net (\d+)', contract_lower)
    payment_days = int(payment_match.group(1)) if payment_match else None
    amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', contract_text)
    
    # Detect risks
    risks = []
    risk_score = 100
    
    if payment_days and payment_days < 30:
        risks.append({
            "clause": "Payment Terms",
            "issue": f"Net {payment_days} is aggressive (market: Net 30)",
            "severity": "Medium",
            "recommendation": "Negotiate for Net 30 or Net 45"
        })
        risk_score -= 10
    
    if "unlimited liability" in contract_lower:
        risks.append({
            "clause": "Liability",
            "issue": "Unlimited liability exposure",
            "severity": "Critical",
            "recommendation": "Cap at 1-2x contract value"
        })
        risk_score -= 25
    
    if "termination fee" in contract_lower:
        risks.append({
            "clause": "Termination",
            "issue": "Termination fees detected",
            "severity": "High",
            "recommendation": "Negotiate for mutual termination, no fees"
        })
        risk_score -= 20
    
    risk_score = max(0, risk_score)
    
    if risk_score >= 80:
        risk_level, risk_color, risk_emoji = "Low Risk", "#10b981", "✅"
    elif risk_score >= 60:
        risk_level, risk_color, risk_emoji = "Medium Risk", "#f59e0b", "⚠️"
    else:
        risk_level, risk_color, risk_emoji = "High Risk", "#ef4444", "🚨"
    
    # Summary
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Contract Analysis Complete</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
            <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Risk Score</p>
            <p style="font-size: 48px; color: {risk_color}; font-weight: 900; margin: 0;">{risk_score}</p>
            <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">{risk_emoji} {risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
            <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Issues Found</p>
            <p style="font-size: 48px; color: {'#ef4444' if len(risks) > 0 else '#10b981'}; font-weight: 900; margin: 0;">{len(risks)}</p>
            <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">risks detected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
            <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Key Terms</p>
            <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">{len(amounts)}</p>
            <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">amounts identified</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
            <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Contract Type</p>
            <p style="font-size: 20px; color: #667eea; font-weight: 900; margin: 0; line-height: 1.2;">{contract_type}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Risks
    if risks:
        critical = [r for r in risks if r['severity'] == 'Critical']
        high = [r for r in risks if r['severity'] == 'High']
        medium = [r for r in risks if r['severity'] == 'Medium']
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
                <div style="background: #ef4444; width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 4px solid white;">
                    <span style="font-size: 36px;">⚠️</span>
                </div>
                <div>
                    <h3 style="color: #991b1b; font-size: 26px; font-weight: 900; margin: 0;">Contract Risks Detected</h3>
                    <p style="color: #dc2626; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">{len(risks)} issues require negotiation</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 36px; color: #dc2626; font-weight: 900; margin: 0;">{len(critical)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Critical</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 36px; color: #f97316; font-weight: 900; margin: 0;">{len(high)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">High</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <p style="font-size: 36px; color: #f59e0b; font-weight: 900; margin: 0;">{len(medium)}</p>
                <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Medium</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # List risks
        severity_colors = {'Critical': '#dc2626', 'High': '#f97316', 'Medium': '#f59e0b'}
        
        for risk in risks:
            color = severity_colors.get(risk['severity'], '#6b7280')
            st.markdown(f"""
            <div style="background: white; border-left: 5px solid {color}; border-radius: 12px; padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <span style="background: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; margin-right: 10px;">{risk['severity'].upper()}</span>
                        <span style="font-size: 18px; color: #1f2937; font-weight: 700;">{risk['clause']}</span>
                    </div>
                </div>
                <p style="font-size: 15px; color: #6b7280; margin: 0 0 12px 0; line-height: 1.6;"><strong>Issue:</strong> {risk['issue']}</p>
                <div style="background: #fef3c7; border: 2px solid #f59e0b; border-radius: 10px; padding: 14px;">
                    <p style="font-size: 14px; color: #92400e; font-weight: 700; margin: 0;">💡 Negotiation Recommendation:</p>
                    <p style="font-size: 14px; color: #78350f; margin: 6px 0 0 0; line-height: 1.6;">{risk['recommendation']}</p>
                </div>
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
                    <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0;">No Major Risks Detected</h3>
                    <p style="color: #047857; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">Contract appears reasonable - standard commercial terms</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
            <span style="font-size: 56px;">⚖️</span>
        </div>
        <h1 style="
            font-size: 58px;
            font-weight: 900;
            color: white;
            margin: 0 0 18px 0;
            text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            Dioptra Contract AI
        </h1>
        <p style="
            font-size: 28px;
            color: rgba(255,255,255,0.95);
            font-weight: 700;
            margin: 15px 0;
        ">
            AI-Powered Contract Analysis & Negotiation
        </p>
        <p style="
            font-size: 18px;
            color: rgba(255,255,255,0.85);
            font-weight: 500;
            margin-bottom: 25px;
        ">
            Risk detection • Market comparison • Negotiation recommendations
        </p>
        <div style="
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
            max-width: 850px;
            margin: 28px auto 0 auto;
        ">
            <span style="background:#ec4899;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Legal AI</span>
            <span style="background:#8b5cf6;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Risk Detection</span>
            <span style="background:#f59e0b;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">NLP Analysis</span>
            <span style="background:#10b981;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">YC Backed</span>
        </div>
        <p style="
            font-size: 16px;
            color: rgba(255,255,255,0.9);
            margin-top: 28px;
            font-weight: 600;
        ">
            Built for <strong style="color:white;">Dioptra AI</strong>
            by <strong style="color:white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["📄 Analyze Contract", "📊 Market Standards"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">AI Contract Analysis</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Upload or paste contract for instant risk assessment and negotiation recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    example_choice = st.selectbox(
        "Try Example Contracts",
        [
            "📝 Custom Contract (Paste Your Own)",
            "💼 SaaS Vendor Agreement (High Risk)",
            "👔 Employment Agreement (Medium Risk)"
        ]
    )
    
    contract_text = st.text_area(
        "Contract Text",
        value=SAMPLE_CONTRACTS.get(example_choice.split(' (')[0].split(' ', 1)[1], "") if example_choice != "📝 Custom Contract (Paste Your Own)" else "",
        height=300,
        placeholder="Paste contract text here..."
    )
    
    contract_type = st.selectbox(
        "Contract Type",
        ["SaaS Agreement", "Employment Agreement", "Consulting Agreement", "NDA", "Partnership Agreement"]
    )
    
    if st.button("🔍 Analyze Contract", use_container_width=True, type="primary"):
        analyze_contract(contract_text, contract_type)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Industry Benchmark Comparison</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Compare your contract terms against market standards</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: white; border-radius: 20px; padding: 28px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
        <h4 style="color: #1f2937; font-size: 20px; font-weight: 800; margin: 0 0 20px 0;">📊 Standard Contract Terms</h4>
        <div style="display: grid; gap: 15px;">
            <div style="background: #f9fafb; border-radius: 12px; padding: 18px; border-left: 4px solid #dc2626;">
                <h4 style="color: #1f2937; margin: 0 0 8px 0;">Payment Terms</h4>
                <p style="color: #6b7280; margin: 0; font-size: 14px;"><strong>Market Standard:</strong> Net 30 days</p>
                <p style="color: #dc2626; margin: 8px 0 0 0; font-size: 13px;"><strong>Importance:</strong> Critical</p>
            </div>
            <div style="background: #f9fafb; border-radius: 12px; padding: 18px; border-left: 4px solid #dc2626;">
                <h4 style="color: #1f2937; margin: 0 0 8px 0;">Liability Cap</h4>
                <p style="color: #6b7280; margin: 0; font-size: 14px;"><strong>Market Standard:</strong> 1-2x contract value</p>
                <p style="color: #dc2626; margin: 8px 0 0 0; font-size: 13px;"><strong>Importance:</strong> Critical</p>
            </div>
            <div style="background: #f9fafb; border-radius: 12px; padding: 18px; border-left: 4px solid #f97316;">
                <h4 style="color: #1f2937; margin: 0 0 8px 0;">Termination</h4>
                <p style="color: #6b7280; margin: 0; font-size: 14px;"><strong>Market Standard:</strong> 30-90 days notice</p>
                <p style="color: #f97316; margin: 8px 0 0 0; font-size: 13px;"><strong>Importance:</strong> High</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<hr style='border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Dioptra AI</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
        <strong>Tech Stack:</strong> Python • Streamlit • NLP • Regex • Contract Analysis
    </p>
</div>
""", unsafe_allow_html=True)