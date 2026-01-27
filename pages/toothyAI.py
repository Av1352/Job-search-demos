"""
Toothy AI - Insurance Verification & Billing Automation
AI-powered dental claim processing and eligibility verification
Built for Toothy AI by Anju Nandhakumar
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

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

def verify_eligibility(patient_name, insurance_plan, member_id):
    """Verify insurance eligibility"""
    
    # Simulate eligibility check
    plan_info = INSURANCE_PLANS.get(insurance_plan, {})
    
    is_active = np.random.random() > 0.1  # 90% chance active
    
    if is_active:
        status = "✅ ACTIVE"
        status_color = "#10b981"
        remaining_benefit = plan_info['annual_max'] * np.random.uniform(0.3, 0.9)
    else:
        status = "❌ INACTIVE"
        status_color = "#ef4444"
        remaining_benefit = 0
    
    result_html = f"""
    <div style="background: linear-gradient(135deg, {status_color} 0%, #73BA9B 100%); padding: 30px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
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
    """
    
    return result_html

def calculate_claim(procedure_codes, insurance_plan):
    """Calculate claim breakdown"""
    
    plan_info = INSURANCE_PLANS.get(insurance_plan, {})
    
    total_cost = 0
    insurance_pays = 0
    patient_pays = 0
    
    procedure_details = []
    
    for code in procedure_codes:
        if code in PROCEDURES:
            proc = PROCEDURES[code]
            cost = proc['cost']
            
            # Coverage varies by category
            if proc['category'] == 'Preventive':
                coverage_pct = min(100, plan_info['coverage'] + 20)
            elif proc['category'] == 'Basic':
                coverage_pct = plan_info['coverage']
            else:  # Major
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
    
    df = pd.DataFrame(procedure_details)
    
    # Create visualization
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Insurance Pays',
        x=[proc['Code'] for proc in procedure_details],
        y=[float(proc['Insurance'].replace('$','')) for proc in procedure_details],
        marker=dict(color='#10b981'),
        text=[proc['Insurance'] for proc in procedure_details],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='Patient Pays',
        x=[proc['Code'] for proc in procedure_details],
        y=[float(proc['Patient'].replace('$','')) for proc in procedure_details],
        marker=dict(color='#f59e0b'),
        text=[proc['Patient'] for proc in procedure_details],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Cost Breakdown by Procedure',
        barmode='stack',
        xaxis_title='Procedure Code',
        yaxis_title='Amount ($)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        height=400
    )
    
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
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
    """
    
    return df, fig, summary_html

# Gradio Interface
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="emerald"),
    css="""
    .gradio-container {font-family: 'Inter', sans-serif;}
    .gr-button-primary {background: linear-gradient(135deg, #10b981 0%, #73BA9B 100%) !important; border: none !important; font-weight: 700 !important;}
    .gr-button-primary:hover {transform: translateY(-2px) !important; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4) !important;}
    """
) as demo:
    
    gr.HTML("""
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
    """)
    
    with gr.Tabs():
        with gr.Tab("🔍 Eligibility Verification"):
            gr.Markdown("### Check Patient Insurance Status")
            
            with gr.Row():
                with gr.Column():
                    patient_name = gr.Textbox(label="Patient Name", placeholder="John Doe")
                    insurance_plan = gr.Dropdown(
                        choices=list(INSURANCE_PLANS.keys()),
                        label="Insurance Plan",
                        value="Delta Dental PPO"
                    )
                    member_id = gr.Textbox(label="Member ID", placeholder="ABC123456789")
                    verify_btn = gr.Button("🔍 Verify Eligibility", variant="primary", size="lg")
            
            eligibility_result = gr.HTML()
            
            verify_btn.click(
                fn=verify_eligibility,
                inputs=[patient_name, insurance_plan, member_id],
                outputs=[eligibility_result]
            )
        
        with gr.Tab("💰 Claim Calculator"):
            gr.Markdown("### Calculate Claim Breakdown")
            
            with gr.Row():
                with gr.Column():
                    claim_insurance = gr.Dropdown(
                        choices=list(INSURANCE_PLANS.keys()),
                        label="Insurance Plan",
                        value="Delta Dental PPO"
                    )
                    
                    procedures = gr.CheckboxGroup(
                        choices=[f"{code}: {PROCEDURES[code]['name']} (${PROCEDURES[code]['cost']})" 
                                for code in PROCEDURES.keys()],
                        label="Select Procedures",
                        value=["D0150: Comprehensive Oral Exam ($95)", "D1110: Teeth Cleaning ($120)"]
                    )
                    
                    calculate_btn = gr.Button("💰 Calculate Claim", variant="primary", size="lg")
            
            claim_summary = gr.HTML()
            claim_table = gr.Dataframe(label="Procedure Details")
            claim_chart = gr.Plot(label="Cost Visualization")
            
            def process_claim(selected_procs, insurance):
                codes = [proc.split(':')[0] for proc in selected_procs]
                return calculate_claim(codes, insurance)
            
            calculate_btn.click(
                fn=process_claim,
                inputs=[procedures, claim_insurance],
                outputs=[claim_table, claim_chart, claim_summary]
            )
    
    gr.HTML("""
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
    """)

if __name__ == "__main__":
    demo.launch()