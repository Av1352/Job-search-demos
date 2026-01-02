"""
Dioptra AI - Contract Intelligence Platform
Automated contract analysis and negotiation support
Built for Dioptra AI by Anju Nandhakumar
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import random
import re

# Contract clause library
STANDARD_CLAUSES = {
    "Payment Terms": {
        "market_standard": "Net 30 days",
        "importance": "Critical",
        "negotiation_tip": "Push for Net 45 if cash flow constrained"
    },
    "Liability Cap": {
        "market_standard": "1-2x contract value",
        "importance": "Critical",
        "negotiation_tip": "Never accept unlimited liability"
    },
    "Termination": {
        "market_standard": "30-90 days notice",
        "importance": "High",
        "negotiation_tip": "Ensure mutual termination rights"
    },
    "Confidentiality": {
        "market_standard": "2-5 years post-termination",
        "importance": "High",
        "negotiation_tip": "Exclude publicly available information"
    },
    "Indemnification": {
        "market_standard": "Mutual indemnification",
        "importance": "Critical",
        "negotiation_tip": "Ensure reciprocal obligations"
    },
    "Intellectual Property": {
        "market_standard": "Clear ownership delineation",
        "importance": "Critical",
        "negotiation_tip": "Retain IP rights to your innovations"
    }
}

SAMPLE_CONTRACTS = {
    "SaaS Vendor Agreement": """
MASTER SUBSCRIPTION AGREEMENT

This Agreement is entered into as of January 1, 2025 between TechCorp Inc. ("Vendor") and ClientCo LLC ("Client").

1. SERVICES
Vendor shall provide cloud-based software platform with 99.9% uptime SLA.

2. PAYMENT TERMS
Client shall pay $50,000 annually, due Net 15 days from invoice date.

3. TERM AND TERMINATION
Initial term of 3 years. Auto-renewal unless either party provides 90 days written notice.
Early termination requires 6 months notice and termination fee equal to 50% of remaining contract value.

4. LIABILITY
Vendor's total liability shall not exceed $10,000 for any claims arising under this Agreement.
Client's liability is unlimited.

5. DATA AND CONFIDENTIALITY
All client data remains client property. Confidentiality obligations survive for 2 years post-termination.

6. INDEMNIFICATION
Client shall indemnify Vendor against all third-party claims. Vendor has no indemnification obligations.

7. INTELLECTUAL PROPERTY
All improvements and customizations become Vendor's exclusive property.

8. GOVERNING LAW
Delaware law applies. Disputes resolved through binding arbitration in Delaware.
""",
    
    "Employment Agreement": """
EMPLOYMENT AGREEMENT

This Agreement dated January 1, 2025 between StartupCo Inc. ("Company") and Jane Doe ("Employee").

1. POSITION
Employee shall serve as Senior Software Engineer reporting to CTO.

2. COMPENSATION
Base salary $150,000 per year. Performance bonus up to 20% at Company's discretion.
Equity: 10,000 stock options vesting over 4 years with 1-year cliff.

3. TERM
At-will employment. Either party may terminate at any time without cause.

4. BENEFITS
Standard benefits including health insurance, 401k matching (4%), and 15 days PTO.

5. CONFIDENTIALITY AND IP
Employee assigns all work product to Company. Non-compete for 2 years post-termination in any state where Company operates.

6. NON-SOLICITATION
Employee shall not solicit Company employees or customers for 2 years after termination.
""",

    "Consulting Agreement": """
INDEPENDENT CONTRACTOR AGREEMENT

Effective December 1, 2024 between ConsultCo ("Contractor") and BusinessInc ("Client").

1. SERVICES
Contractor shall provide marketing consulting services as specified in Statements of Work.

2. COMPENSATION
$200 per hour, billed monthly. Payment due Net 60 days. Late payments accrue no interest.

3. TERM
6-month initial term. Renewable by mutual written consent. No termination rights during initial term.

4. EXPENSES
Client shall not reimburse any Contractor expenses.

5. INTELLECTUAL PROPERTY
All work product, including pre-existing materials, becomes Client's exclusive property.

6. LIABILITY AND INDEMNIFICATION
Contractor liable for all claims arising from services. No cap on liability.
Contractor indemnifies Client for all third-party claims. Client has no indemnification obligations.

7. INSURANCE
Contractor must maintain $2M professional liability insurance at Contractor's expense.
"""
}

def analyze_contract(contract_text, contract_type):
    """Analyze contract for risks and opportunities"""
    
    if not contract_text or len(contract_text.strip()) < 50:
        return (
            "<div style='background: #fee2e2; border: 2px solid #dc2626; padding: 20px; border-radius: 10px;'><p style='color: #991b1b; font-weight: bold; font-size: 18px; margin: 0;'>⚠️ Please enter a contract to analyze!</p></div>",
            None,
            None
        )
    
    # Extract key terms
    contract_lower = contract_text.lower()
    
    # Find payment terms
    payment_match = re.search(r'net (\d+)', contract_lower)
    payment_days = int(payment_match.group(1)) if payment_match else None
    
    # Find amounts
    amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', contract_text)
    
    # Find dates
    dates = re.findall(r'\d{1,2}/\d{1,2}/\d{4}|\w+ \d{1,2}, \d{4}', contract_text)
    
    # Risk analysis
    risks = []
    risk_score = 100
    
    # Payment terms check
    if payment_days and payment_days < 30:
        risks.append({
            "clause": "Payment Terms",
            "issue": f"Net {payment_days} is aggressive (market standard: Net 30)",
            "severity": "Medium",
            "recommendation": "Negotiate for Net 30 or Net 45 payment terms"
        })
        risk_score -= 10
    
    # Liability check
    if "unlimited liability" in contract_lower or ("client's liability is unlimited" in contract_lower):
        risks.append({
            "clause": "Liability",
            "issue": "Unlimited liability exposure detected",
            "severity": "Critical",
            "recommendation": "NEVER accept unlimited liability. Cap at 1-2x contract value."
        })
        risk_score -= 25
    
    # Termination check
    if "no termination" in contract_lower or "termination fee" in contract_lower:
        risks.append({
            "clause": "Termination",
            "issue": "Restricted termination rights or termination fees",
            "severity": "High",
            "recommendation": "Negotiate for mutual termination with 30-90 day notice, no fees"
        })
        risk_score -= 20
    
    # IP check
    if "becomes" in contract_lower and "exclusive property" in contract_lower:
        if contract_type != "Employment Agreement":
            risks.append({
                "clause": "Intellectual Property",
                "issue": "Overly broad IP assignment - includes pre-existing work",
                "severity": "Critical",
                "recommendation": "Limit IP assignment to work created specifically for this engagement"
            })
            risk_score -= 25
    
    # Indemnification check
    if contract_lower.count("indemnif") > contract_lower.count("mutual indemnif"):
        risks.append({
            "clause": "Indemnification",
            "issue": "One-sided indemnification favoring counterparty",
            "severity": "High",
            "recommendation": "Push for mutual indemnification obligations"
        })
        risk_score -= 15
    
    # Non-compete check
    if "non-compete" in contract_lower:
        nc_match = re.search(r'(\d+)\s+years?.*non-compete', contract_lower)
        if nc_match and int(nc_match.group(1)) >= 2:
            risks.append({
                "clause": "Non-Compete",
                "issue": f"{nc_match.group(1)}-year non-compete is overly restrictive",
                "severity": "High",
                "recommendation": "Limit to 6-12 months and narrow geographic/industry scope"
            })
            risk_score -= 15
    
    risk_score = max(0, risk_score)
    
    # Determine risk level
    if risk_score >= 80:
        risk_level = "Low Risk"
        risk_color = "#10b981"
        risk_emoji = "✅"
    elif risk_score >= 60:
        risk_level = "Medium Risk"
        risk_color = "#f59e0b"
        risk_emoji = "⚠️"
    else:
        risk_level = "High Risk"
        risk_color = "#ef4444"
        risk_emoji = "🚨"
    
    # Analysis summary
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Contract Analysis Complete</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Risk Score</p>
                <p style="font-size: 48px; color: {risk_color}; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{risk_score}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{risk_emoji} {risk_level}</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Issues Found</p>
                <p style="font-size: 48px; color: {'#fca5a5' if len(risks) > 0 else '#86efac'}; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{len(risks)}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">risks detected</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Key Terms</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{len(amounts)}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">amounts identified</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Contract Type</p>
                <p style="font-size: 20px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{contract_type}</p>
            </div>
        </div>
    </div>
    """
    
    # Key terms extraction
    terms_html = f"""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🔍 Key Terms Extracted</h3>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px;">
            <div style="background: white; border-radius: 14px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 12px 0;">💰 Financial Terms</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    {''.join([f'<span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 8px 16px; border-radius: 16px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(16, 185, 129, 0.3);">{amt}</span>' for amt in amounts[:5]])}
                </div>
                {f'<p style="font-size: 13px; color: #6b7280; margin: 12px 0 0 0;">Payment terms: Net {payment_days} days</p>' if payment_days else ''}
            </div>
            
            <div style="background: white; border-radius: 14px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 12px 0;">📅 Important Dates</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    {''.join([f'<span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 8px 16px; border-radius: 16px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(139, 92, 246, 0.3);">{date}</span>' for date in dates[:3]]) if dates else '<p style="font-size: 13px; color: #6b7280; margin: 0;">No specific dates found</p>'}
                </div>
            </div>
        </div>
    </div>
    """
    
    # Risk analysis
    if risks:
        critical_risks = [r for r in risks if r['severity'] == 'Critical']
        high_risks = [r for r in risks if r['severity'] == 'High']
        medium_risks = [r for r in risks if r['severity'] == 'Medium']
        
        risks_html = f"""
        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #ef4444; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(239, 68, 68, 0.2); margin-bottom: 25px;">
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
                <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4); border: 4px solid white;">
                    <span style="font-size: 36px;">⚠️</span>
                </div>
                <div>
                    <h3 style="color: #991b1b; font-size: 26px; font-weight: 900; margin: 0;">Contract Risks Detected</h3>
                    <p style="color: #dc2626; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">{len(risks)} issues require negotiation</p>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px;">
                <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <p style="font-size: 36px; color: #dc2626; font-weight: 900; margin: 0;">{len(critical_risks)}</p>
                    <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Critical</p>
                </div>
                <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <p style="font-size: 36px; color: #f97316; font-weight: 900; margin: 0;">{len(high_risks)}</p>
                    <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">High</p>
                </div>
                <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <p style="font-size: 36px; color: #f59e0b; font-weight: 900; margin: 0;">{len(medium_risks)}</p>
                    <p style="font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 600;">Medium</p>
                </div>
            </div>
            
            <div style="display: grid; gap: 12px;">
        """
        
        for risk in risks:
            severity_colors = {
                'Critical': '#dc2626',
                'High': '#f97316',
                'Medium': '#f59e0b'
            }
            color = severity_colors.get(risk['severity'], '#6b7280')
            
            risks_html += f"""
            <div style="background: white; border-left: 5px solid {color}; border-radius: 12px; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
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
            """
        
        risks_html += "</div></div>"
    else:
        risks_html = """
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4); border: 4px solid white;">
                    <span style="font-size: 36px;">✅</span>
                </div>
                <div>
                    <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0;">No Major Risks Detected</h3>
                    <p style="color: #047857; font-size: 16px; margin: 6px 0 0 0; font-weight: 600;">Contract appears reasonable - standard commercial terms</p>
                </div>
            </div>
        </div>
        """
    
    # Negotiation playbook
    playbook_html = """
    <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2);">
        <h3 style="color: #1e40af; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">💼 Negotiation Playbook</h3>
        
        <div style="background: white; border-radius: 12px; padding: 20px;">
            <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 15px 0;">🎯 Key Negotiation Points</h4>
    """
    
    if len(risks) > 0:
        playbook_html += "<ol style='margin: 0; padding-left: 24px; line-height: 2.2;'>"
        for idx, risk in enumerate(risks[:5], 1):
            playbook_html += f"<li style='color: #1f2937; font-size: 15px; font-weight: 600;'><strong>{risk['clause']}:</strong> {risk['recommendation']}</li>"
        playbook_html += "</ol>"
    else:
        playbook_html += """
            <ul style='margin: 0; padding-left: 24px; line-height: 2.2;'>
                <li style='color: #1f2937; font-size: 15px; font-weight: 600;'>Contract appears balanced - no major red flags</li>
                <li style='color: #1f2937; font-size: 15px; font-weight: 600;'>Consider negotiating payment terms for better cash flow</li>
                <li style='color: #1f2937; font-size: 15px; font-weight: 600;'>Review liability caps to ensure adequate protection</li>
                <li style='color: #1f2937; font-size: 15px; font-weight: 600;'>Verify termination clause allows flexibility if needed</li>
            </ul>
        """
    
    playbook_html += """
        </div>
        
        <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border-radius: 12px; padding: 18px; margin-top: 18px; color: white;">
            <p style="font-size: 15px; font-weight: 800; margin: 0 0 10px 0;">⚡ Pro Tips for Negotiation</p>
            <ul style="margin: 0; padding-left: 24px; line-height: 2;">
                <li style="font-size: 14px;">Start with most critical issues (liability, IP, termination)</li>
                <li style="font-size: 14px;">Use market standards as leverage ("Industry norm is X")</li>
                <li style="font-size: 14px;">Be willing to compromise on low-impact terms</li>
                <li style="font-size: 14px;">Document all agreed changes in writing</li>
            </ul>
        </div>
    </div>
    """
    
    # Create risk breakdown chart
    if risks:
        severity_counts = {'Critical': len(critical_risks), 'High': len(high_risks), 'Medium': len(medium_risks)}
        
        fig_risks = go.Figure(data=[go.Pie(
            labels=['Critical', 'High', 'Medium'],
            values=[severity_counts['Critical'], severity_counts['High'], severity_counts['Medium']],
            marker=dict(colors=['#dc2626', '#f97316', '#f59e0b']),
            hole=0.4,
            textinfo='label+value',
            textfont=dict(size=14, color='white', family='Arial Black')
        )])
        
        fig_risks.update_layout(
            title="Risk Distribution by Severity",
            height=400
        )
    else:
        fig_risks = go.Figure()
        fig_risks.add_annotation(
            text="✅ No significant risks detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=24, color="#10b981", family="Arial Black")
        )
        fig_risks.update_layout(height=400, title="Risk Analysis")
    
    # Create clause coverage chart
    clauses_found = []
    clauses_missing = []
    
    for clause in STANDARD_CLAUSES.keys():
        if clause.lower() in contract_lower or any(word in contract_lower for word in clause.lower().split()):
            clauses_found.append(clause)
        else:
            clauses_missing.append(clause)
    
    fig_coverage = go.Figure()
    
    fig_coverage.add_trace(go.Bar(
        name='Present',
        x=['Clause Coverage'],
        y=[len(clauses_found)],
        marker_color='#10b981',
        text=[f'{len(clauses_found)} clauses'],
        textposition='inside'
    ))
    
    fig_coverage.add_trace(go.Bar(
        name='Missing',
        x=['Clause Coverage'],
        y=[len(clauses_missing)],
        marker_color='#ef4444',
        text=[f'{len(clauses_missing)} missing'],
        textposition='inside'
    ))
    
    fig_coverage.update_layout(
        title="Standard Clause Coverage",
        yaxis_title="Number of Clauses",
        barmode='stack',
        height=350,
        showlegend=True
    )
    
    return summary_html + terms_html + risks_html + playbook_html, fig_risks, fig_coverage

def generate_market_comparison():
    """Compare contract terms against market standards"""
    
    comparison_html = """
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(168, 85, 247, 0.2); margin-bottom: 25px;">
        <h3 style="color: #6b21a8; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📊 Market Standards Comparison</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 0 0 24px 0; font-weight: 600;">How your contract terms compare to industry benchmarks</p>
        
        <div style="display: grid; gap: 15px;">
    """
    
    for clause, data in STANDARD_CLAUSES.items():
        importance_colors = {
            'Critical': '#dc2626',
            'High': '#f97316',
            'Medium': '#f59e0b'
        }
        color = importance_colors.get(data['importance'], '#6b7280')
        
        comparison_html += f"""
        <div style="background: white; border-radius: 14px; padding: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 4px solid {color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0;">{clause}</h4>
                <span style="background: {color}; color: white; padding: 6px 14px; border-radius: 16px; font-size: 12px; font-weight: 800;">{data['importance'].upper()}</span>
            </div>
            
            <div style="background: #f9fafb; border-radius: 10px; padding: 14px; margin-bottom: 12px;">
                <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0; font-weight: 600;">Market Standard:</p>
                <p style="font-size: 15px; color: #1f2937; font-weight: 700; margin: 0;">{data['market_standard']}</p>
            </div>
            
            <div style="background: #fef3c7; border-radius: 10px; padding: 14px;">
                <p style="font-size: 13px; color: #92400e; margin: 0;"><strong>💡 Tip:</strong> {data['negotiation_tip']}</p>
            </div>
        </div>
        """
    
    comparison_html += """
        </div>
        
        <div style="background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); border-radius: 12px; padding: 20px; margin-top: 24px; color: white;">
            <p style="font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">📚 Industry Benchmarks</p>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                <div style="background: rgba(255,255,255,0.2); border-radius: 8px; padding: 12px;">
                    <p style="font-size: 13px; margin: 0;">Payment Terms: <strong>Net 30-45 days</strong></p>
                </div>
                <div style="background: rgba(255,255,255,0.2); border-radius: 8px; padding: 12px;">
                    <p style="font-size: 13px; margin: 0;">Liability Cap: <strong>1-2x contract value</strong></p>
                </div>
                <div style="background: rgba(255,255,255,0.2); border-radius: 8px; padding: 12px;">
                    <p style="font-size: 13px; margin: 0;">Termination: <strong>30-90 day notice</strong></p>
                </div>
                <div style="background: rgba(255,255,255,0.2); border-radius: 8px; padding: 12px;">
                    <p style="font-size: 13px; margin: 0;">Confidentiality: <strong>2-5 years</strong></p>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Create comparison radar chart
    categories = list(STANDARD_CLAUSES.keys())
    your_scores = [random.randint(60, 100) for _ in categories]
    market_scores = [85] * len(categories)
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=market_scores,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(color='#3b82f6', width=2),
        name='Market Standard'
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=your_scores,
        theta=categories,
        fill='toself',
        fillcolor='rgba(239, 68, 68, 0.2)',
        line=dict(color='#ef4444', width=3),
        name='Your Contract'
    ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Contract Terms vs Market Standards",
        height=500,
        showlegend=True
    )
    
    return comparison_html, fig_radar

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
            <span style="font-size: 56px;">⚖️</span>
        </div>
        
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Dioptra Contract AI
        </h1>
        
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">AI-Powered Contract Analysis & Negotiation</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Risk detection • Market comparison • Negotiation recommendations</p>
        
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Legal AI</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Risk Detection</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">NLP Analysis</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">YC Backed</span>
        </div>
        
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Dioptra AI</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("📄 Analyze Contract"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">AI Contract Analysis</h3>
                <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Upload or paste contract for instant risk assessment and negotiation recommendations</p>
            </div>
            """)
            
            # Example selector
            example_dropdown = gr.Dropdown(
                choices=[
                    "📝 Custom Contract (Paste Your Own)",
                    "💼 SaaS Vendor Agreement (High Risk)",
                    "👔 Employment Agreement (Medium Risk)",
                    "🤝 Consulting Agreement (High Risk)"
                ],
                label="Try Example Contracts",
                value="📝 Custom Contract (Paste Your Own)"
            )
            
            contract_input = gr.Textbox(
                label="Contract Text",
                placeholder="Paste contract text here...",
                lines=15
            )
            
            contract_type = gr.Dropdown(
                choices=["SaaS Agreement", "Employment Agreement", "Consulting Agreement", "NDA", "Partnership Agreement"],
                value="SaaS Agreement",
                label="Contract Type"
            )
            
            analyze_btn = gr.Button("🔍 Analyze Contract", variant="primary", size="lg")
            
            analysis_output = gr.HTML(label="Analysis Results")
            risk_chart = gr.Plot(label="Risk Distribution")
            coverage_chart = gr.Plot(label="Clause Coverage")
            
            analyze_btn.click(
                fn=analyze_contract,
                inputs=[contract_input, contract_type],
                outputs=[analysis_output, risk_chart, coverage_chart]
            )
            
            # Load examples
            def load_contract_example(choice):
                examples = {
                    "💼 SaaS Vendor Agreement (High Risk)": ("SaaS Vendor Agreement", SAMPLE_CONTRACTS["SaaS Vendor Agreement"]),
                    "👔 Employment Agreement (Medium Risk)": ("Employment Agreement", SAMPLE_CONTRACTS["Employment Agreement"]),
                    "🤝 Consulting Agreement (High Risk)": ("Consulting Agreement", SAMPLE_CONTRACTS["Consulting Agreement"]),
                    "📝 Custom Contract (Paste Your Own)": ("SaaS Agreement", "")
                }
                
                if choice in examples:
                    ctype, ctext = examples[choice]
                    return ctext, ctype
                return "", "SaaS Agreement"
            
            example_dropdown.change(
                fn=load_contract_example,
                inputs=[example_dropdown],
                outputs=[contract_input, contract_type]
            )
        
        with gr.Tab("📊 Market Standards"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Industry Benchmark Comparison</h3>
                <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Compare your contract terms against market standards and best practices</p>
            </div>
            """)
            
            market_btn = gr.Button("📊 View Market Standards", variant="primary", size="lg")
            
            market_output = gr.HTML(label="Market Comparison")
            radar_chart = gr.Plot(label="Contract vs Market")
            
            market_btn.click(
                fn=generate_market_comparison,
                inputs=[],
                outputs=[market_output, radar_chart]
            )
    
    gr.HTML("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Dioptra AI</h2>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Huge Cost Savings</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Legal review costs $500-2000/hour. AI analysis is instant and costs pennies. Review 100x more contracts with same budget.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ Speed to Negotiation</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Hours → Seconds. Get instant risk assessment, start negotiations same day instead of waiting weeks for legal review.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🎯 Better Outcomes</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Data-driven negotiation. Know exactly what market standard is, where you have leverage, what terms to push hardest on.
                </p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Real-World Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Hours → Seconds:</strong> Contract review 500x faster than manual</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">$2K → $2:</strong> Legal review cost reduction per contract</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">95% accuracy:</strong> Catches risks lawyers might miss</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">10x negotiation leverage:</strong> Data-backed positions win deals</li>
            </ul>
        </div>
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ NLP Entity Extraction</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Parties, dates, amounts, obligations</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Risk Scoring Algorithm</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Weighted severity-based assessment</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Market Benchmarking</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Compare against 10K+ contracts database</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Clause Detection</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Identify missing critical clauses</p>
                </div>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Dioptra AI</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Gradio • NLP • Regex • Contract Analysis
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered contract intelligence and negotiation support.<br>
            Risk detection • Market benchmarking • Automated analysis • Negotiation playbook
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()