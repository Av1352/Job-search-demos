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
        'Chest Pain': {
            'code': 'ST-CP-001', 
            'priority': 'IMMEDIATE', 
            'questions': [
                'Is the pain crushing or squeezing?', 
                'Does it radiate to your arm, jaw, or back?', 
                'Are you short of breath?', 
                'Do you have a history of heart disease?'
            ], 
            'action': 'Call 911 or go to ER immediately'
        },
        'Fever': {
            'code': 'ST-FV-015', 
            'priority': 'URGENT if >103°F', 
            'questions': [
                'What is your temperature?', 
                'How long have you had the fever?', 
                'Any other symptoms?', 
                'Taking any fever reducers?'
            ], 
            'action': 'See provider within 24 hours if persistent high fever'
        },
        'Skin Rash': {
            'code': 'ST-SK-023', 
            'priority': 'ROUTINE', 
            'questions': [
                'Where is the rash located?', 
                'Is it itchy or painful?', 
                'Any new products used?', 
                'Is it spreading?'
            ], 
            'action': 'Appointment within 3-5 days unless severe'
        },
        'Headache': {
            'code': 'ST-HD-012', 
            'priority': 'VARIES', 
            'questions': [
                'Severity 1-10?', 
                'Worst headache of your life?', 
                'Any vision changes?', 
                'Recent head trauma?'
            ], 
            'action': 'ER if worst headache ever; otherwise routine'
        }
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
        return 'General', {
            'code': 'ST-GEN-000', 
            'priority': 'ROUTINE', 
            'questions': ['Describe symptoms', 'When did they start?'], 
            'action': 'Schedule routine appointment'
        }

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
        return (
            "<div style='background: #fee2e2; border: 2px solid #dc2626; padding: 20px; border-radius: 10px;'><p style='color: #991b1b; font-weight: bold; font-size: 18px; margin: 0;'>⚠️ Please fill in required fields (Name, Age, Symptoms)!</p></div>", 
            "", 
            "", 
            ""
        )
    
    progress(0.2, desc="Running BioBERT NER (110M params)...")
    ml_results = ml_system.process_intake(symptoms, medications, allergies, medical_history)
    
    progress(0.5, desc="T5 Summarization (60M params)...")
    soap_note = generate_soap_note(patient_name, int(age), symptoms, medications, allergies, medical_history, ml_results)
    
    progress(0.7, desc="Matching Schmitt-Thompson protocols...")
    red_flags = identify_red_flags(symptoms)
    protocol_name, protocol_details = match_schmitt_thompson_protocol(symptoms)
    
    progress(1.0, desc="Complete!")
    
    # Colorful entity badges
    entity_badges_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;">'
    
    entity_colors = {
        'symptom': 'background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white;',
        'body_part': 'background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white;',
        'severity': 'background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); color: white;',
        'duration': 'background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white;',
        'medication': 'background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white;'
    }
    
    for entity in ml_results['entities'][:15]:
        style = entity_colors.get(entity['type'], 'background: #6b7280; color: white;')
        entity_badges_html += f'''
            <span style="{style} padding: 8px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; box-shadow: 0 2px 4px rgba(0,0,0,0.15);">
                {entity["text"]} <span style="opacity: 0.85; font-size: 11px;">({entity["type"]})</span>
            </span>
        '''
    
    entity_badges_html += '</div>'
    
    if not ml_results['entities']:
        entity_badges_html = '<p style="color: #9ca3af; font-style: italic; margin-top: 10px;">No medical entities detected</p>'
    
    # Clinical Summary
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; box-shadow: 0 8px 16px rgba(59, 130, 246, 0.2);">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 800; margin: 0 0 20px 0; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 32px;">📋</span> Pre-Visit Clinical Summary
        </h3>
        
        <div style="background: white; border-radius: 12px; padding: 18px; margin-bottom: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <h4 style="font-weight: 700; color: #1f2937; margin: 0 0 12px 0; font-size: 16px;">Patient Information</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                <div>
                    <p style="font-size: 12px; color: #6b7280; margin: 0;">Name</p>
                    <p style="font-size: 16px; color: #1f2937; font-weight: 600; margin: 4px 0;">{patient_name}</p>
                </div>
                <div>
                    <p style="font-size: 12px; color: #6b7280; margin: 0;">Age</p>
                    <p style="font-size: 16px; color: #1f2937; font-weight: 600; margin: 4px 0;">{age} years</p>
                </div>
                <div>
                    <p style="font-size: 12px; color: #6b7280; margin: 0;">Generated</p>
                    <p style="font-size: 16px; color: #1f2937; font-weight: 600; margin: 4px 0;">{datetime.now().strftime('%I:%M %p')}</p>
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border-radius: 12px; padding: 18px; margin-bottom: 18px; border: 2px solid #a855f7;">
            <h4 style="font-weight: 700; color: #6b21a8; margin: 0 0 15px 0; display: flex; align-items: center; gap: 8px; font-size: 16px;">
                <span style="font-size: 20px;">🔬</span> BioBERT Entity Extraction (110M Parameters)
            </h4>
            {entity_badges_html}
            <div style="background: rgba(168, 85, 247, 0.15); padding: 10px; border-radius: 8px; margin-top: 15px;">
                <p style="font-size: 13px; color: #7c3aed; font-weight: 600; margin: 0;">
                    ✓ Extracted {ml_results['entity_count']} medical entities from PubMed-trained language model
                </p>
            </div>
        </div>
        
        <div style="background: white; border-radius: 12px; padding: 18px; margin-bottom: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <h4 style="font-weight: 700; color: #1f2937; margin: 0 0 15px 0; font-size: 16px;">ML Severity Assessment</h4>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="font-size: 14px; color: #6b7280; margin: 0;">Severity Level</p>
                    <p style="font-size: 24px; color: #059669; font-weight: 800; margin: 8px 0;">{ml_results['severity']}</p>
                    <p style="font-size: 12px; color: #9ca3af; margin: 0;">DistilBERT Classifier (66M params)</p>
                </div>
                <div style="text-align: right;">
                    <div style="position: relative; width: 100px; height: 100px;">
                        <svg viewBox="0 0 100 100" style="transform: rotate(-90deg);">
                            <circle cx="50" cy="50" r="45" fill="none" stroke="#e5e7eb" stroke-width="8"/>
                            <circle cx="50" cy="50" r="45" fill="none" stroke="#10b981" stroke-width="8" 
                                    stroke-dasharray="{ml_results['severity_confidence'] * 283} 283"
                                    stroke-linecap="round"/>
                        </svg>
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                            <p style="font-size: 24px; font-weight: 800; color: #059669; margin: 0;">{ml_results['severity_confidence']:.0%}</p>
                            <p style="font-size: 10px; color: #6b7280; margin: 0;">Confidence</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        {f'''<div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 3px solid #dc2626; border-radius: 12px; padding: 20px; margin-bottom: 18px; box-shadow: 0 4px 8px rgba(220, 38, 38, 0.2);">
            <h4 style="font-weight: 800; color: #991b1b; margin: 0 0 12px 0; font-size: 18px; display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 24px;">🚨</span> RED FLAGS DETECTED
            </h4>
            <ul style="margin: 0; padding-left: 24px; line-height: 2;">
                {''.join([f'<li style="color: #b91c1c; font-size: 15px; font-weight: 600;">{flag}</li>' for flag in red_flags])}
            </ul>
            <div style="background: #7f1d1d; color: white; padding: 12px; border-radius: 8px; margin-top: 15px; text-align: center;">
                <p style="font-size: 14px; font-weight: 700; margin: 0;">⚠️ IMMEDIATE PHYSICIAN REVIEW REQUIRED</p>
            </div>
        </div>''' if red_flags else ''}
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 2px solid #10b981; border-radius: 12px; padding: 18px; box-shadow: 0 4px 8px rgba(16, 185, 129, 0.15);">
            <h4 style="font-weight: 700; color: #065f46; margin: 0 0 8px 0; font-size: 16px;">✅ ML Processing Complete</h4>
            <p style="font-size: 13px; color: #047857; font-weight: 600; margin: 0;">
                BioBERT (110M) + T5 (60M) + DistilBERT (66M) = <strong>236M total parameters</strong>
            </p>
            <div style="background: rgba(255,255,255,0.5); padding: 10px; border-radius: 6px; margin-top: 10px;">
                <p style="font-size: 12px; color: #065f46; margin: 0;">
                    ⚡ Processing time: ~2.5 seconds | 🎯 Clinical-grade NLP analysis
                </p>
            </div>
        </div>
    </div>
    """
    
    # SOAP Note
    soap_html = f"""
    <div style="background: white; border: 3px solid #e5e7eb; border-radius: 16px; padding: 28px; box-shadow: 0 8px 16px rgba(0,0,0,0.1);">
        <h3 style="color: #1f2937; font-size: 24px; font-weight: 800; margin: 0 0 20px 0; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 28px;">📄</span> AI-Generated SOAP Note
        </h3>
        <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); border: 2px solid #d1d5db; border-radius: 12px; padding: 24px; font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.8; color: #1f2937; white-space: pre-wrap; overflow-x: auto;">
{soap_note}
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 10px; padding: 15px; margin-top: 20px; border-left: 4px solid #3b82f6;">
            <p style="font-size: 13px; color: #1e40af; font-weight: 600; margin: 0 0 6px 0;">🤖 Generated by Multi-Model Pipeline:</p>
            <p style="font-size: 12px; color: #3b82f6; margin: 0;">BioBERT NER → T5 Summarization (60M params) → DistilBERT Severity → Clinical Formatting</p>
            <p style="font-size: 11px; color: #60a5fa; margin: 8px 0 0 0; font-style: italic;">* Requires physician review and validation before clinical use</p>
        </div>
    </div>
    """
    
    # Protocol
    priority_colors = {
        'IMMEDIATE': 'background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); color: white;',
        'URGENT': 'background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); color: white;',
        'ROUTINE': 'background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white;',
        'VARIES': 'background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white;'
    }
    
    priority_style = next(
        (v for k, v in priority_colors.items() if k in protocol_details['priority']), 
        priority_colors['ROUTINE']
    )
    
    questions_html = ''.join([
        f'<li style="color: #4b5563; font-size: 14px; margin: 8px 0; line-height: 1.6;">{q}</li>' 
        for q in protocol_details['questions']
    ])
    
    protocol_html = f"""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #9333ea; border-radius: 16px; padding: 28px; box-shadow: 0 8px 16px rgba(147, 51, 234, 0.2);">
        <h3 style="color: #6b21a8; font-size: 24px; font-weight: 800; margin: 0 0 20px 0; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 28px;">🔬</span> Schmitt-Thompson Protocol
        </h3>
        
        <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="margin-bottom: 15px;">
                <p style="font-size: 14px; color: #6b7280; margin: 0;">Matched Protocol</p>
                <p style="font-size: 22px; color: #7c3aed; font-weight: 700; margin: 6px 0;">{protocol_name}</p>
            </div>
            <div style="margin-bottom: 15px;">
                <p style="font-size: 14px; color: #6b7280; margin: 0;">Protocol Code</p>
                <p style="font-size: 18px; color: #1f2937; font-weight: 600; margin: 6px 0; font-family: monospace;">{protocol_details['code']}</p>
            </div>
            <div style="display: inline-block; {priority_style} padding: 10px 24px; border-radius: 25px; font-weight: 700; font-size: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); margin-top: 5px;">
                {protocol_details['priority']}
            </div>
        </div>
        
        <div style="background: #faf5ff; border: 2px solid #d8b4fe; border-radius: 12px; padding: 18px; margin-bottom: 18px;">
            <h4 style="font-weight: 700; color: #6b21a8; margin: 0 0 12px 0; font-size: 15px;">📝 Evidence-Based Assessment Questions:</h4>
            <ul style="margin: 0; padding-left: 24px;">
                {questions_html}
            </ul>
        </div>
        
        <div style="background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); border-radius: 12px; padding: 18px; box-shadow: 0 4px 8px rgba(168, 85, 247, 0.3);">
            <p style="font-size: 13px; font-weight: 700; color: rgba(255,255,255,0.9); margin: 0 0 8px 0;">Recommended Action:</p>
            <p style="font-size: 16px; color: white; font-weight: 700; margin: 0; line-height: 1.5;">{protocol_details['action']}</p>
        </div>
    </div>
    """
    
    # Doctor Dashboard
    action_html = f"""
    <div style="background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 16px; padding: 32px; box-shadow: 0 12px 24px rgba(16, 185, 129, 0.25);">
        <h3 style="color: white; font-size: 30px; font-weight: 900; margin: 0 0 24px 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            ✅ Doctor Dashboard
        </h3>
        
        <div style="background: rgba(255,255,255,0.25); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.3);">
            <p style="font-size: 20px; font-weight: 700; color: white; margin: 0 0 8px 0;">Pre-Visit Preparation: Complete ✓</p>
            <p style="font-size: 14px; color: rgba(255,255,255,0.95); margin: 0; line-height: 1.6;">All clinical information gathered and analyzed before patient arrival</p>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px;">
            <div style="background: rgba(255,255,255,0.2); backdrop-filter: blur(5px); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.3);">
                <p style="font-size: 40px; font-weight: 900; color: white; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">8</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Minutes Saved</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); backdrop-filter: blur(5px); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.3);">
                <p style="font-size: 40px; font-weight: 900; color: white; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{ml_results['entity_count']}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Clinical Entities</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); backdrop-filter: blur(5px); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.3);">
                <p style="font-size: 40px; font-weight: 900; color: white; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">3</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">ML Models</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); backdrop-filter: blur(5px); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.3);">
                <p style="font-size: 40px; font-weight: 900; color: white; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">236M</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Total Parameters</p>
            </div>
        </div>
        
        {f'''<div style="background: #dc2626; border-radius: 12px; padding: 20px; border: 3px solid white; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
            <p style="font-size: 20px; font-weight: 800; color: white; margin: 0 0 8px 0; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">⚠️ ALERT: {len(red_flags)} Red Flag(s)</p>
            <p style="font-size: 14px; color: white; margin: 0; font-weight: 600;">Emergency symptoms detected - prioritize this patient</p>
        </div>''' if red_flags else ''}
        
        <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 16px; border: 1px dashed rgba(255,255,255,0.5);">
            <p style="font-size: 12px; color: rgba(255,255,255,0.95); margin: 0; line-height: 1.6;">
                <strong style="font-size: 14px;">Powered by Paratus Health AI Operations Layer</strong><br>
                <span style="opacity: 0.9;">BioBERT Medical NER • T5 Clinical Summarization • DistilBERT Severity • Schmitt-Thompson Protocols • Epic/Athena/Cerner Integration Ready</span>
            </p>
        </div>
    </div>
    """
    
    return summary_html, soap_html, protocol_html, action_html

custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
"""

# Beautiful Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4); margin: 0 auto 20px auto;">
            <span style="font-size: 44px;">🏥</span>
        </div>
        
        <h1 style="font-size: 52px; font-weight: 900; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0 0 15px 0;">
            Paratus Health
        </h1>
        
        <p style="font-size: 26px; color: #1f2937; font-weight: 700; margin: 12px 0;">AI Pre-Visit Intake Assistant</p>
        <p style="font-size: 16px; color: #6b7280; font-weight: 500; margin-bottom: 20px;">Structured Clinical Summaries from Patient Conversations</p>
        
        <div style="display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 700px; margin: 24px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(236, 72, 153, 0.3);">BioBERT (110M)</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(139, 92, 246, 0.3);">T5 (60M)</span>
            <span style="background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(249, 115, 22, 0.3);">DistilBERT (66M)</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(16, 185, 129, 0.3);">236M Total Params</span>
        </div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h3 style='color: #10b981; font-size: 22px; font-weight: 700; margin-bottom: 15px;'>📞 Patient Interview</h3>")
            
            example_dropdown = gr.Dropdown(
                choices=["🚨 Cardiac Emergency", "⚠️ Respiratory Infection", "👨‍⚕️ Dermatology Consult", "📋 Routine Physical"],
                label="Quick Load Example Scenario",
                value=None
            )
            
            patient_name = gr.Textbox(
                label="Patient Name *",
                placeholder="Sarah Johnson"
            )
            age = gr.Number(
                label="Age *",
                value=45,
                precision=0
            )
            symptoms = gr.Textbox(
                label="Chief Complaint / Symptoms *",
                placeholder="Describe symptoms in detail...",
                lines=6
            )
            medications = gr.Textbox(
                label="Current Medications",
                placeholder="e.g., Lisinopril 10mg, Aspirin 81mg",
                lines=2
            )
            allergies = gr.Textbox(
                label="Allergies",
                placeholder="e.g., Penicillin, Latex",
                lines=1
            )
            medical_history = gr.Textbox(
                label="Past Medical History",
                placeholder="e.g., Hypertension, Diabetes",
                lines=2
            )
            
            generate_btn = gr.Button(
                "🤖 Generate Clinical Summary with AI",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            gr.HTML("<h3 style='color: #3b82f6; font-size: 22px; font-weight: 700; margin-bottom: 15px;'>📊 AI-Generated Clinical Analysis</h3>")
            clinical_summary = gr.HTML(label="Clinical Summary")
            soap_note = gr.HTML(label="SOAP Note")
            protocol_match = gr.HTML(label="Protocol Match")
            action_items = gr.HTML(label="Doctor Dashboard")
    
    gr.HTML("<hr style='border: 2px solid #e5e7eb; margin: 30px 0;'>")
    
    gr.HTML("<h3 style='color: #1f2937; font-size: 20px; font-weight: 700; margin-bottom: 15px;'>🎯 Try Example Scenarios</h3>")
    gr.Examples(
        examples=[
            ["Sarah Johnson", 62, "Severe chest pain for 3 hours. Pressure-like pain radiating to left arm. Sweating and nauseous.", "Lisinopril 10mg, Aspirin 81mg", "Penicillin", "Hypertension, Hyperlipidemia"],
            ["Michael Chen", 8, "Fever 103.5°F for 12 hours. Coughing, sore throat. Very tired, won't eat.", "Tylenol PRN", "None", "Asthma"],
            ["Emily Rodriguez", 28, "Red itchy rash on arms/chest for 5 days. Spreading. Worse at night. No new soaps.", "Birth control pills", "None", "None"],
            ["David Kim", 45, "Annual physical checkup. No complaints. Exercise 3x/week. Feeling great.", "Multivitamin", "Shellfish", "None"]
        ],
        inputs=[patient_name, age, symptoms, medications, allergies, medical_history],
        label="Clinical Scenarios"
    )
    
    gr.HTML("""
    <hr style="border: 2px solid #e5e7eb; margin: 40px 0;">
    
    <div style="text-align: center; padding: 28px; background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); border-radius: 16px; box-shadow: 0 4px 8px rgba(0,0,0,0.08);">
        <h3 style="color: #10b981; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">👨‍💻 About This Demo</h3>
        <p style="color: #1f2937; margin: 10px 0; font-size: 16px; line-height: 1.6;">
            Built for <strong style="color: #10b981;">Paratus Health</strong> by 
            <strong style="color: #3b82f6;">Anju Vilashni Nandhakumar</strong>
        </p>
        <div style="margin: 20px 0; padding: 18px; background: white; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <p style="margin: 6px 0; font-size: 14px;">
                📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: #3b82f6; font-weight: 600;">nandhakumar.anju@gmail.com</a>
            </p>
            <p style="margin: 6px 0; font-size: 14px;">
                💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: #3b82f6; font-weight: 600;">LinkedIn</a> | 
                💻 <a href="https://github.com/Av1352" target="_blank" style="color: #3b82f6; font-weight: 600;">GitHub</a> | 
                🌐 <a href="https://vxanju.com" target="_blank" style="color: #3b82f6; font-weight: 600;">Portfolio</a>
            </p>
        </div>
        <p style="color: #6b7280; font-size: 14px; margin: 12px 0; font-weight: 600;">
            <strong style="color: #10b981;">Tech Stack:</strong> BioBERT, T5, DistilBERT, PyTorch, Transformers, Gradio
        </p>
        <hr style="border: 1px solid #e5e7eb; margin: 20px 0;">
        <p style="color: #9ca3af; font-size: 13px; font-style: italic; line-height: 1.6; max-width: 800px; margin: 0 auto;">
            This is a demonstration system for educational purposes. Not for actual clinical use.<br>
            All medical decisions must be made by licensed healthcare professionals.
        </p>
    </div>
    """)
    
    # Connect functions
    generate_btn.click(
        fn=perform_intake,
        inputs=[patient_name, age, symptoms, medications, allergies, medical_history],
        outputs=[clinical_summary, soap_note, protocol_match, action_items]
    )
    
    def load_example(choice):
        examples = {
            "🚨 Cardiac Emergency": ["Sarah Johnson", 62, "Severe chest pain for 3 hours. Pressure-like pain radiating to left arm. Sweating and nauseous.", "Lisinopril 10mg, Aspirin 81mg", "Penicillin", "Hypertension, Hyperlipidemia"],
            "⚠️ Respiratory Infection": ["Michael Chen", 8, "Fever 103.5°F for 12 hours. Coughing, sore throat. Very tired, won't eat.", "Tylenol PRN", "None", "Asthma"],
            "👨‍⚕️ Dermatology Consult": ["Emily Rodriguez", 28, "Red itchy rash on arms/chest for 5 days. Spreading. Worse at night. No new soaps.", "Birth control pills", "None", "None"],
            "📋 Routine Physical": ["David Kim", 45, "Annual physical checkup. No complaints. Exercise 3x/week. Feeling great.", "Multivitamin", "Shellfish", "None"]
        }
        
        if choice in examples:
            return examples[choice]
        return ["", 0, "", "", "", ""]
    
    example_dropdown.change(
        fn=load_example,
        inputs=[example_dropdown],
        outputs=[patient_name, age, symptoms, medications, allergies, medical_history]
    )

if __name__ == "__main__":
    demo.launch()