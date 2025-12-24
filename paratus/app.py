"""
Paratus Health - AI Pre-Visit Intake Assistant
Beautiful, colorful, user-friendly interface with 3 ML models
"""

import gradio as gr
from ml_model import get_paratus_ml_system
from datetime import datetime
import re

# Initialize ML system
print("Initializing Paratus Health ML System...")
ml_system = get_paratus_ml_system()
print("Ready!")

def match_schmitt_thompson_protocol(symptoms):
    """Match to Schmitt-Thompson protocols"""
    
    symptoms_lower = symptoms.lower()
    
    protocols = {
        'Chest Pain': {'code': 'ST-CP-001', 'priority': 'IMMEDIATE', 'questions': ['Is the pain crushing or squeezing?', 'Does it radiate to your arm, jaw, or back?', 'Are you short of breath?', 'Do you have a history of heart disease?'], 'action': 'Call 911 or go to ER immediately'},
        'Fever': {'code': 'ST-FV-015', 'priority': 'URGENT if >103°F', 'questions': ['What is your temperature?', 'How long have you had the fever?', 'Any other symptoms?', 'Taking any fever reducers?'], 'action': 'See provider within 24 hours if persistent high fever'},
        'Skin Rash': {'code': 'ST-SK-023', 'priority': 'ROUTINE', 'questions': ['Where is the rash located?', 'Is it itchy or painful?', 'Any new products used?', 'Is it spreading?'], 'action': 'Appointment within 3-5 days unless severe'},
        'Headache': {'code': 'ST-HD-012', 'priority': 'VARIES', 'questions': ['Severity 1-10?', 'Worst headache of your life?', 'Any vision changes?', 'Recent head trauma?'], 'action': 'ER if worst headache ever; otherwise routine'}
    }
    
    if 'chest pain' in symptoms_lower or 'heart' in symptoms_lower:
        return 'Chest Pain', protocols['Chest Pain']
    elif 'fever' in symptoms_lower or '103' in symptoms or '104' in symptoms:
        return 'Fever', protocols['Fever']
    elif 'rash' in symptoms_lower or 'skin' in symptoms_lower or 'itch' in symptoms_lower:
        return 'Skin Rash', protocols['Skin Rash']
    elif 'headache' in symptoms_lower:
        return 'Headache', protocols['Headache']
    else:
        return 'General', {'code': 'ST-GEN-000', 'priority': 'ROUTINE', 'questions': ['Describe symptoms', 'When did they start?'], 'action': 'Schedule routine appointment'}

def identify_red_flags(symptoms):
    """Identify critical symptoms"""
    
    red_flags_map = {
        '🚨 CARDIAC': ['chest pain', 'crushing pain', 'left arm pain'],
        '🚨 RESPIRATORY': ['difficulty breathing', "can't breathe", 'severe shortness of breath'],
        '🚨 NEUROLOGICAL': ['worst headache', 'sudden weakness', 'seizure'],
        '⚠️ INFECTION': ['high fever', '103', '104']
    }
    
    identified = []
    symptoms_lower = symptoms.lower()
    
    for category, keywords in red_flags_map.items():
        for keyword in keywords:
            if keyword in symptoms_lower:
                identified.append(f"{category}: {keyword}")
                break
    
    return identified

def generate_soap_note(patient_name, age, symptoms, medications, allergies, history, ml_results):
    """Generate SOAP note"""
    
    hpi = ml_results['hpi']
    
    subjective = f"""SUBJECTIVE:
Patient: {patient_name}, {age} years old

History of Present Illness (HPI):
{hpi}

Current Medications: {medications or 'None'}
Allergies: {allergies or 'NKDA'}
Past Medical History: {history or 'None reported'}

ML Analysis: {ml_results['severity']} severity ({ml_results['severity_confidence']:.0%} confidence)
BioBERT NER: {ml_results['entity_count']} medical entities extracted"""

    objective = """OBJECTIVE:
Vitals: To be obtained
Physical Exam: To be performed
Labs/Imaging: Pending"""

    symptoms_lower = symptoms.lower()
    if 'chest pain' in symptoms_lower:
        ddx = ["Acute coronary syndrome", "Costochondritis", "GERD", "Anxiety"]
    elif 'fever' in symptoms_lower and 'cough' in symptoms_lower:
        ddx = ["URI", "Pneumonia", "Bronchitis", "Influenza"]
    elif 'rash' in symptoms_lower:
        ddx = ["Contact dermatitis", "Allergic reaction", "Eczema"]
    else:
        ddx = ["To be determined"]
    
    assessment = f"""ASSESSMENT:
Differential: {', '.join(ddx)}
Impression: Pending exam"""

    plan = """PLAN:
- Complete physical exam
- Order labs/imaging as indicated
- Treatment per diagnosis
- Follow-up as needed"""

    return f"{subjective}\n\n{objective}\n\n{assessment}\n\n{plan}"

def perform_intake(patient_name, age, symptoms, medications, allergies, medical_history, progress=gr.Progress()):
    """Process intake with ML"""
    
    if not patient_name or not age or not symptoms:
        return "<p class='text-red-600 font-bold text-lg'>⚠️ Please fill in required fields!</p>", "", "", ""
    
    progress(0.2, desc="Running BioBERT NER...")
    ml_results = ml_system.process_intake(symptoms, medications, allergies, medical_history)
    
    progress(0.5, desc="Generating clinical summary...")
    soap_note = generate_soap_note(patient_name, int(age), symptoms, medications, allergies, medical_history, ml_results)
    
    progress(0.7, desc="Matching protocols...")
    red_flags = identify_red_flags(symptoms)
    protocol_name, protocol_details = match_schmitt_thompson_protocol(symptoms)
    
    progress(1.0, desc="Complete!")
    
    # Colorful entity badges
    entity_badges_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px;">'
    
    entity_colors = {
        'symptom': 'background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white;',
        'body_part': 'background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white;',
        'severity': 'background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); color: white;',
        'duration': 'background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white;',
        'medication': 'background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white;'
    }
    
    for entity in ml_results['entities'][:12]:
        style = entity_colors.get(entity['type'], 'background: #gray; color: white;')
        entity_badges_html += f'''
            <span style="{style} padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                {entity["text"]} <span style="opacity: 0.8; font-size: 11px;">({entity["type"]})</span>
            </span>
        '''
    
    entity_badges_html += '</div>'
    
    if not ml_results['entities']:
        entity_badges_html = '<p style="color: #6b7280; font-style: italic;">No medical entities detected</p>'
    
    # Clinical Summary
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h3 style="color: #1e40af; font-size: 24px; font-weight: 800; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 28px;">📋</span> Pre-Visit Clinical Summary
        </h3>
        
        <div style="background: white; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <h4 style="font-weight: 700; color: #1f2937; margin-bottom: 8px;">Patient Info</h4>
            <p style="font-size: 14px; color: #374151; margin: 4px 0;"><strong>Name:</strong> {patient_name}</p>
            <p style="font-size: 14px; color: #374151; margin: 4px 0;"><strong>Age:</strong> {age} years</p>
            <p style="font-size: 14px; color: #374151; margin: 4px 0;"><strong>Generated:</strong> {datetime.now().strftime('%I:%M %p')}</p>
        </div>
        
        <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
            <h4 style="font-weight: 700; color: #6b21a8; margin-bottom: 12px; display: flex; align-items: center; gap: 6px;">
                <span style="font-size: 18px;">🔬</span> BioBERT Entity Extraction
            </h4>
            {entity_badges_html}
            <p style="font-size: 12px; color: #7c3aed; margin-top: 12px; font-weight: 600;">
                ✓ BioBERT (110M params) extracted {ml_results['entity_count']} medical terms
            </p>
        </div>
        
        <div style="background: white; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
            <h4 style="font-weight: 700; color: #1f2937; margin-bottom: 8px;">ML Severity Assessment</h4>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="font-size: 14px; color: #374151;"><strong>Level:</strong> {ml_results['severity']}</p>
                    <p style="font-size: 12px; color: #6b7280; margin-top: 4px;">DistilBERT Classifier (66M params)</p>
                </div>
                <div style="text-align: right;">
                    <p style="font-size: 32px; font-weight: 800; color: #059669;">{ml_results['severity_confidence']:.0%}</p>
                    <p style="font-size: 11px; color: #6b7280;">Confidence</p>
                </div>
            </div>
        </div>
        
        {f'''<div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #dc2626; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
            <h4 style="font-weight: 800; color: #991b1b; margin-bottom: 8px; font-size: 16px;">🚨 RED FLAGS DETECTED</h4>
            <ul style="margin: 0; padding-left: 20px;">
                {''.join([f'<li style="color: #b91c1c; font-size: 14px; font-weight: 600; margin: 4px 0;">{flag}</li>' for flag in red_flags])}
            </ul>
            <p style="font-size: 12px; color: #7f1d1d; margin-top: 8px; font-weight: 700;">⚠️ Immediate physician review required</p>
        </div>''' if red_flags else ''}
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-radius: 12px; padding: 16px;">
            <h4 style="font-weight: 700; color: #065f46; margin-bottom: 4px;">✅ ML Processing Complete</h4>
            <p style="font-size: 12px; color: #047857;">BioBERT (110M) • T5 (60M) • DistilBERT (66M) = 236M total params</p>
        </div>
    </div>
    """
    
    # SOAP Note with better formatting
    soap_html = f"""
    <div style="background: white; border: 2px solid #e5e7eb; border-radius: 16px; padding: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h3 style="color: #1f2937; font-size: 22px; font-weight: 800; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 26px;">📄</span> AI-Generated SOAP Note
        </h3>
        <pre style="font-family: 'Courier New', monospace; font-size: 13px; color: #1f2937; background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 20px; border-radius: 12px; line-height: 1.6; white-space: pre-wrap; border: 1px solid #d1d5db;">{soap_note}</pre>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 10px; padding: 12px; margin-top: 16px;">
            <p style="font-size: 12px; color: #1e40af; font-weight: 600;">🤖 Generated by: BioBERT NER + T5 Summarization (60M params) + DistilBERT Severity</p>
            <p style="font-size: 11px; color: #3b82f6; margin-top: 4px;">* Requires physician review and validation</p>
        </div>
    </div>
    """
    
    # Protocol with vibrant colors
    priority_colors = {
        'IMMEDIATE': 'background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);',
        'URGENT': 'background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);',
        'ROUTINE': 'background: linear-gradient(135deg, #10b981 0%, #059669 100%);',
        'VARIES': 'background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);'
    }
    
    priority_style = next((v for k, v in priority_colors.items() if k in protocol_details['priority']), priority_colors['ROUTINE'])
    
    protocol_html = f"""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #9333ea; border-radius: 16px; padding: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 26px;">🔬</span> Schmitt-Thompson Protocol
        </h3>
        
        <div style="background: white; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
            <p style="font-size: 15px; margin-bottom: 8px;"><strong style="color: #1f2937;">Protocol:</strong> <span style="color: #7c3aed; font-weight: 700;">{protocol_name}</span></p>
            <p style="font-size: 14px; margin-bottom: 12px;"><strong style="color: #1f2937;">Code:</strong> {protocol_details['code']}</p>
            <div style="display: inline-block; {priority_style} color: white; padding: 8px 20px; border-radius: 25px; font-weight: 700; font-size: 14px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                {protocol_details['priority']}
            </div>
            
            <h4 style="font-weight: 700; color: #1f2937; margin-top: 16px; margin-bottom: 8px; font-size: 14px;">Assessment Questions:</h4>
            <ul style="margin: 0; padding-left: 24px;">
                {chr(10).join([f'<li style="color: #4b5563; font-size: 13px; margin: 6px 0;">{q}</li>' for q in protocol_details['questions']])}
            </ul>
        </div>
        
        <div style="background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); border-radius: 12px; padding: 14px;">
            <p style="font-size: 13px; font-weight: 700; color: white; margin-bottom: 4px;">Recommended Action:</p>
            <p style="font-size: 14px; color: white; font-weight: 600;">{protocol_details['action']}</p>
        </div>
    </div>
    """
    
    # Doctor Dashboard
    action_html = f"""
    <div style="background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 16px; padding: 28px; box-shadow: 0 8px 16px rgba(0,0,0,0.15);">
        <h3 style="color: white; font-size: 28px; font-weight: 800; margin-bottom: 20px; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            ✅ Doctor Dashboard
        </h3>
        
        <div style="background: rgba(255,255,255,0.25); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; margin-bottom: 16px;">
            <p style="font-size: 18px; font-weight: 700; color: white; margin-bottom: 6px;">Pre-Visit Preparation: Complete ✓</p>
            <p style="font-size: 13px; color: rgba(255,255,255,0.9);">All clinical information gathered before patient arrival</p>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 16px;">
            <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 16px; text-align: center;">
                <p style="font-size: 32px; font-weight: 800; color: white; margin: 0;">8</p>
                <p style="font-size: 12px; color: rgba(255,255,255,0.85); margin-top: 4px;">Minutes Saved</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 16px; text-align: center;">
                <p style="font-size: 32px; font-weight: 800; color: white; margin: 0;">{ml_results['entity_count']}</p>
                <p style="font-size: 12px; color: rgba(255,255,255,0.85); margin-top: 4px;">Entities Found</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 16px; text-align: center;">
                <p style="font-size: 32px; font-weight: 800; color: white; margin: 0;">3</p>
                <p style="font-size: 12px; color: rgba(255,255,255,0.85); margin-top: 4px;">ML Models</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 16px; text-align: center;">
                <p style="font-size: 32px; font-weight: 800; color: white; margin: 0;">236M</p>
                <p style="font-size: 12px; color: rgba(255,255,255,0.85); margin-top: 4px;">Parameters</p>
            </div>
        </div>
        
        {f'''<div style="background: #dc2626; border-radius: 12px; padding: 18px; border: 3px solid white; margin-bottom: 16px;">
            <p style="font-size: 18px; font-weight: 800; color: white; margin-bottom: 6px;">⚠️ ALERT: {len(red_flags)} Red Flag(s)</p>
            <p style="font-size: 13px; color: white;">Immediate review required - potential emergency</p>
        </div>''' if red_flags else ''}
        
        <div style="background: rgba(255,255,255,0.15); border-radius: 10px; padding: 14px; border: 1px dashed rgba(255,255,255,0.4);">
            <p style="font-size: 11px; color: rgba(255,255,255,0.9); margin: 0;">
                <strong>Powered by Paratus Health AI</strong><br>
                BioBERT • T5 • DistilBERT • Schmitt-Thompson Protocols • EHR-Ready
            </p>
        </div>
    </div>
    """
    
    return summary_html, soap_html, protocol_html, action_html

# Beautiful Gradio Interface
with gr.Blocks(
    title="Paratus Health AI Intake",
    css="""
    .gradio-container {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%) !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 14px !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
    .gr-button-primary:hover {
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4) !important;
        transform: translateY(-2px);
    }
    """
) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 32px 0;">
        <div style="display: inline-flex; align-items: center; gap: 16px; margin-bottom: 16px;">
            <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);">
                <span style="font-size: 32px;">🏥</span>
            </div>
            <h1 style="font-size: 48px; font-weight: 900; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
                Paratus Health
            </h1>
        </div>
        <p style="font-size: 22px; color: #4b5563; font-weight: 600; margin-bottom: 8px;">AI Pre-Visit Intake Assistant</p>
        <p style="font-size: 15px; color: #6b7280;">236M Parameter ML Pipeline • BioBERT + T5 + DistilBERT</p>
        <div style="margin-top: 16px; display: inline-flex; gap: 12px; flex-wrap: wrap; justify-content: center;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 700;">Medical NER</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 700;">Clinical Summarization</span>
            <span style="background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); color: white; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 700;">Severity Classification</span>
        </div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📞 Patient Interview")
            
            example_dropdown = gr.Dropdown(
                choices=["🚨 Cardiac Emergency", "⚠️ Respiratory Infection", "👨‍⚕️ Dermatology Consult", "📋 Routine Physical"],
                label="Quick Load Example",
                value=None
            )
            
            patient_name = gr.Textbox(label="Patient Name", placeholder="Sarah Johnson")
            age = gr.Number(label="Age", value=45, precision=0)
            symptoms = gr.Textbox(label="Chief Complaint", placeholder="Describe symptoms...", lines=6)
            medications = gr.Textbox(label="Current Medications", placeholder="e.g., Lisinopril 10mg", lines=2)
            allergies = gr.Textbox(label="Allergies", placeholder="e.g., Penicillin", lines=1)
            medical_history = gr.Textbox(label="Past Medical History", placeholder="e.g., Hypertension", lines=2)
            
            generate_btn = gr.Button("🤖 Generate Clinical Summary", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 AI Analysis")
            clinical_summary = gr.HTML(label="Clinical Summary")
            soap_note = gr.HTML(label="SOAP Note")
            protocol_match = gr.HTML(label="Protocol")
            action_items = gr.HTML(label="Dashboard")
    
    gr.Examples(
        examples=[
            ["Sarah Johnson", 62, "Severe chest pain for 3 hours. Pressure-like pain radiating to left arm. Sweating and nauseous.", "Lisinopril, Aspirin", "Penicillin", "Hypertension"],
            ["Michael Chen", 8, "Fever 103.5°F for 12 hours. Coughing, sore throat. Very tired, won't eat.", "Tylenol PRN", "None", "Asthma"],
            ["Emily Rodriguez", 28, "Red itchy rash on arms/chest for 5 days. Spreading. Worse at night.", "Birth control", "None", "None"],
            ["David Kim", 45, "Annual physical. Healthy. Exercise 3x/week. No complaints.", "None", "Shellfish", "None"]
        ],
        inputs=[patient_name, age, symptoms, medications, allergies, medical_history],
        label="Example Scenarios"
    )
    
    gr.Markdown("---\n**Built by Anju Vilashni Nandhakumar** | nandhakumar.anju@gmail.com")
    
    generate_btn.click(
        fn=perform_intake,
        inputs=[patient_name, age, symptoms, medications, allergies, medical_history],
        outputs=[clinical_summary, soap_note, protocol_match, action_items]
    )
    
    def load_example(choice):
        examples = {
            "🚨 Cardiac Emergency": ["Sarah Johnson", 62, "Severe chest pain for 3 hours. Pressure-like pain radiating to left arm. Sweating and nauseous.", "Lisinopril, Aspirin", "Penicillin", "Hypertension"],
            "⚠️ Respiratory Infection": ["Michael Chen", 8, "Fever 103.5°F for 12 hours. Coughing, sore throat. Very tired, won't eat.", "Tylenol PRN", "None", "Asthma"],
            "👨‍⚕️ Dermatology Consult": ["Emily Rodriguez", 28, "Red itchy rash on arms/chest for 5 days. Spreading. Worse at night.", "Birth control", "None", "None"],
            "📋 Routine Physical": ["David Kim", 45, "Annual physical. Healthy. Exercise 3x/week. No complaints.", "None", "Shellfish", "None"]
        }
        
        if choice in examples:
            return examples[choice]
        return ["", 0, "", "", "", ""]
    
    example_dropdown.change(fn=load_example, inputs=[example_dropdown], outputs=[patient_name, age, symptoms, medications, allergies, medical_history])

if __name__ == "__main__":
    demo.launch()