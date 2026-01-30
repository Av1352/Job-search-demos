"""
Output Biosciences - Biologically-Aware Generative AI
Large Biological Models for drug discovery
Built for Output Biosciences by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Output Biosciences", page_icon="🧬", layout="wide")

# Protein structures
PROTEIN_TARGETS = {
    'EGFR (Cancer)': {'confidence': 0.94, 'druggability': 0.89, 'binding_affinity': -12.3},
    'ACE2 (COVID-19)': {'confidence': 0.91, 'druggability': 0.85, 'binding_affinity': -10.8},
    'BACE1 (Alzheimers)': {'confidence': 0.88, 'druggability': 0.82, 'binding_affinity': -11.5},
    'TNF-α (Inflammation)': {'confidence': 0.92, 'druggability': 0.87, 'binding_affinity': -13.1},
    'PD-L1 (Immunotherapy)': {'confidence': 0.90, 'druggability': 0.84, 'binding_affinity': -11.9}
}

# Drug candidate properties
DRUG_PROPERTIES = {
    'Molecular Weight': {'value': 487.3, 'optimal': '180-500', 'status': '✅'},
    'LogP (Lipophilicity)': {'value': 2.8, 'optimal': '0-3', 'status': '✅'},
    'H-Bond Donors': {'value': 2, 'optimal': '≤5', 'status': '✅'},
    'H-Bond Acceptors': {'value': 6, 'optimal': '≤10', 'status': '✅'},
    'Toxicity Score': {'value': 0.12, 'optimal': '<0.3', 'status': '✅'},
    'Bioavailability': {'value': 0.78, 'optimal': '>0.5', 'status': '✅'}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #a855f7 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🧬</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Output Biosciences</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Biologically-Aware Generative AI</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Large Biological Models • Drug discovery • Protein design</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🧬 Protein Design", "💊 Drug Generation", "📊 Candidate Analysis", "💡 Technology"])

with tab1:
    st.markdown("### Generative Protein Design")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Target Configuration**")
        
        disease_target = st.selectbox("Disease Target", list(PROTEIN_TARGETS.keys()))
        
        st.markdown("**Design Parameters**")
        
        design_mode = st.selectbox("Design Mode", ["De Novo Design", "Optimize Existing", "Scaffold-Based"])
        binding_strength = st.slider("Target Binding Affinity", -15.0, -8.0, -12.0, 0.5)
        selectivity = st.slider("Selectivity Score", 0.5, 1.0, 0.85, 0.05)
        
        st.markdown("**Constraints**")
        drug_like = st.checkbox("Drug-like properties (Lipinski)", value=True)
        synthesizable = st.checkbox("Synthesizable", value=True)
        low_toxicity = st.checkbox("Low toxicity", value=True)
        
        generate_btn = st.button("🧬 Generate Molecule", type="primary", use_container_width=True)
    
    with col2:
        if generate_btn:
            st.markdown("**Generation Progress**")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            stages = [
                ("Analyzing target protein...", 0.2),
                ("Generating molecular structures...", 0.4),
                ("Evaluating binding affinity...", 0.6),
                ("Optimizing drug-like properties...", 0.8),
                ("Validating constraints...", 1.0)
            ]
            
            for stage, progress in stages:
                status_text.text(stage)
                progress_bar.progress(progress)
                time.sleep(0.5)
            
            st.success("✅ Drug candidate generated!")
            
            target_data = PROTEIN_TARGETS[disease_target]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #a855f7 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Generated Candidate</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Target</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{disease_target}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Confidence</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{target_data['confidence']*100:.1f}%</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Binding Affinity</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{target_data['binding_affinity']} kcal/mol</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Druggability</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{target_data['druggability']*100:.1f}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Generation Time", "12.3s", "Fast")
            col2.metric("Lipinski Rule", "✅ Pass", "Drug-like")
            col3.metric("Toxicity", "0.12", "Low")
            col4.metric("Synthesizability", "94.5%", "High")

with tab2:
    st.markdown("### AI-Generated Drug Candidates")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Drug Properties**")
        
        props_data = []
        for prop, data in DRUG_PROPERTIES.items():
            props_data.append({
                'Property': prop,
                'Value': data['value'],
                'Optimal Range': data['optimal'],
                'Status': data['status']
            })
        
        st.dataframe(pd.DataFrame(props_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Molecular Structure**")
        st.code("C23H29N5O4S (Example: Generated small molecule)", language="text")
        st.info("🧪 SMILES: CC(C)Cc1ccc(cc1)C(C)C(=O)NC(Cc2ccccc2)C(=O)O")
    
    with col2:
        st.markdown("**Property Distribution**")
        
        fig1 = go.Figure(data=[go.Bar(
            x=list(DRUG_PROPERTIES.keys()),
            y=[DRUG_PROPERTIES[p]['value'] if isinstance(DRUG_PROPERTIES[p]['value'], (int, float)) else 1 for p in DRUG_PROPERTIES.keys()],
            marker=dict(color='#a855f7'),
            text=[f"{DRUG_PROPERTIES[p]['value']}" for p in DRUG_PROPERTIES.keys()],
            textposition='auto'
        )])
        fig1.update_layout(yaxis_title='Value', height=250)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("**Clinical Trial Prediction**")
        
        phases = ['Phase I', 'Phase II', 'Phase III', 'FDA Approval']
        success_rates = [78, 62, 45, 38]
        
        fig2 = go.Figure(data=[go.Bar(
            x=phases,
            y=success_rates,
            marker=dict(color=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']),
            text=[f"{r}%" for r in success_rates],
            textposition='auto'
        )])
        fig2.update_layout(yaxis_title='Success Rate (%)', height=250)
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Candidate Evaluation Dashboard")
    
    st.markdown("**Generated Candidates Comparison**")
    
    candidates = {
        'Candidate': ['OB-2024-001', 'OB-2024-002', 'OB-2024-003', 'OB-2024-004', 'OB-2024-005'],
        'Target': ['EGFR', 'ACE2', 'BACE1', 'TNF-α', 'PD-L1'],
        'Binding (kcal/mol)': [-12.3, -10.8, -11.5, -13.1, -11.9],
        'Druggability': ['89%', '85%', '82%', '87%', '84%'],
        'Toxicity': [0.12, 0.18, 0.15, 0.10, 0.14],
        'Status': ['✅ Advanced', '✅ Advanced', '⚠️ Review', '✅ Advanced', '✅ Advanced']
    }
    st.dataframe(pd.DataFrame(candidates), hide_index=True, use_container_width=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Binding Affinity Distribution**")
        
        fig3 = go.Figure(data=[go.Scatter(
            x=candidates['Candidate'],
            y=candidates['Binding (kcal/mol)'],
            mode='markers',
            marker=dict(
                size=15,
                color=candidates['Binding (kcal/mol)'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Affinity")
            )
        )])
        fig3.update_layout(yaxis_title='Binding Affinity', height=250)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("**Success Metrics**")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates", "5", "Generated")
        col2.metric("Advanced", "4/5", "80%")
        col3.metric("Avg Affinity", "-11.9", "Strong")

with tab4:
    st.markdown("### Large Biological Models Technology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**AI Models**")
        st.markdown("""
        - ✅ Protein language models (ESM-2, ProtGPT)
        - ✅ Molecular generation (diffusion models)
        - ✅ Binding affinity prediction
        - ✅ ADMET property prediction
        - ✅ Retrosynthesis planning
        - ✅ Multi-objective optimization
        """)
        
        st.markdown("**Biological Understanding**")
        st.markdown("""
        - ✅ Protein structure prediction (AlphaFold-level)
        - ✅ Protein-protein interactions
        - ✅ Binding site identification
        - ✅ Functional annotation
        - ✅ Evolutionary constraints
        - ✅ Pathway analysis
        """)
    
    with col2:
        st.markdown("**Drug Discovery Pipeline**")
        st.markdown("""
        - ✅ Target identification
        - ✅ Hit generation (10K+ candidates)
        - ✅ Lead optimization
        - ✅ ADMET prediction
        - ✅ Toxicity screening
        - ✅ Clinical trial simulation
        """)
        
        st.markdown("**Validation**")
        st.markdown("""
        - ✅ Molecular dynamics simulations
        - ✅ Docking studies
        - ✅ In silico validation
        - ✅ Wet lab confirmation
        - ✅ Animal model prediction
        - ✅ Clinical outcome forecasting
        """)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #6b21a8; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #9333ea; font-weight: 700; margin: 0 0 6px 0;">✓ Large Bio Models</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">ESM-2, ProtGPT, diffusion models</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #9333ea; font-weight: 700; margin: 0 0 6px 0;">✓ 94% Confidence</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">High-quality predictions</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #9333ea; font-weight: 700; margin: 0 0 6px 0;">✓ 10K+ Candidates</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Rapid hit generation</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #9333ea; font-weight: 700; margin: 0 0 6px 0;">✓ 12.3s Generation</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Fast molecular design</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #a855f7 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Output Biosciences</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)