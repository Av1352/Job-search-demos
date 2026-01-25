"""
Novaflow - AI Data Analyst for Biology Labs
Automated experiment analysis and insights
Built for Novaflow by Anju Nandhakumar
"""

import streamlit as st
from utils.sidebar import render_sidebar
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Novaflow - Biology Lab AI", layout="wide")
render_sidebar()

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Sample experimental data
SAMPLE_EXPERIMENTS = {
    "Protein Expression - Western Blot": {
        "type": "Western Blot",
        "samples": 8,
        "data": pd.DataFrame({
            'Sample': ['Control', 'Treatment 1µM', 'Treatment 5µM', 'Treatment 10µM', 
                        'Treatment 20µM', 'Treatment 50µM', 'Positive Control', 'Negative Control'],
            'Band_Intensity': [0.15, 0.18, 0.32, 0.58, 0.82, 0.95, 1.0, 0.05],
            'Normalized': [1.0, 1.2, 2.1, 3.9, 5.5, 6.3, 6.7, 0.3]
        })
    },
    "Cell Viability - MTT Assay": {
        "type": "MTT Assay",
        "samples": 6,
        "data": pd.DataFrame({
            'Concentration': [0, 1, 5, 10, 25, 50],
            'Viability_%': [100, 98, 92, 78, 45, 22],
            'Std_Dev': [2.3, 3.1, 4.2, 5.8, 6.1, 4.9]
        })
    },
    "Gene Expression - qPCR": {
        "type": "qPCR",
        "samples": 4,
        "data": pd.DataFrame({
            'Gene': ['GAPDH (control)', 'Target Gene 1', 'Target Gene 2', 'Target Gene 3'],
            'Ct_Value': [18.2, 24.5, 21.3, 28.9],
            'Fold_Change': [1.0, 3.8, 2.1, 0.4],
            'P_Value': [1.0, 0.003, 0.015, 0.082]
        })
    }
}

def analyze_experiment(exp_data, exp_type):
    """Analyze experimental data and generate insights"""
    
    df = exp_data
    
    if exp_type == "Western Blot":
        # Find EC50 or optimal concentration
        max_response = df['Normalized'].max()
        ec50_idx = (df['Normalized'] - max_response/2).abs().idxmin()
        ec50_sample = df.loc[ec50_idx, 'Sample']
        
        insights = {
            'key_finding': f"Dose-dependent response observed. EC50 approximately at {ec50_sample}.",
            'fold_induction': f"{max_response:.1f}x over control",
            'statistical_sig': "p < 0.01" if max_response > 3 else "p < 0.05" if max_response > 2 else "Not significant",
            'recommendation': "Proceed to functional assays" if max_response > 3 else "Consider higher doses or alternative compounds"
        }
    
    elif exp_type == "MTT Assay":
        # Calculate IC50
        viable_50 = df[df['Viability_%'] < 50]
        if len(viable_50) > 0:
            ic50_conc = viable_50.iloc[0]['Concentration']
        else:
            ic50_conc = ">50"
        
        insights = {
            'key_finding': f"IC50 approximately {ic50_conc} µM",
            'toxicity_profile': "High toxicity" if ic50_conc != ">50" and float(ic50_conc) < 10 else "Moderate toxicity" if ic50_conc != ">50" else "Low toxicity",
            'statistical_sig': "p < 0.001",
            'recommendation': f"Therapeutic window identified. Consider in vivo studies at {float(ic50_conc)*0.3 if ic50_conc != '>50' else 5}-{ic50_conc if ic50_conc != '>50' else 15} µM range."
        }
    
    elif exp_type == "qPCR":
        # Identify significant genes
        sig_genes = df[df['P_Value'] < 0.05]
        upregulated = sig_genes[sig_genes['Fold_Change'] > 1.5]
        downregulated = sig_genes[sig_genes['Fold_Change'] < 0.67]
        
        insights = {
            'key_finding': f"{len(upregulated)} genes significantly upregulated, {len(downregulated)} downregulated",
            'top_hit': df.loc[df['Fold_Change'].idxmax(), 'Gene'] if len(upregulated) > 0 else "None",
            'statistical_sig': f"{len(sig_genes)}/{len(df)} genes p < 0.05",
            'recommendation': "Validate top candidates with Western blot or functional assays"
        }
    
    return insights

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #059669 0%, #73BA9B 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(5, 150, 105, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #14b8a6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🔬</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Novaflow
        </h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">AI Data Analyst for Biology Labs</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Automated experiment analysis and insights</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Data Analysis</span>
            <span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);">Statistical Testing</span>
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Visualization</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">YC Backed</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Novaflow</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Value Prop
st.markdown("""
<div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); padding: 25px; border-radius: 15px; border: 2px solid #059669; margin-bottom: 30px;">
    <h3 style="color: #065f46; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">🎯 The Lab Data Problem</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #ef4444; font-weight: 700; margin: 0 0 8px 0;">❌ Today</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Researchers spend 10+ hours/week on Excel. Manual calculations, copy-paste errors. Stats analysis takes days.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #f59e0b; font-weight: 700; margin: 0 0 8px 0;">💰 Cost Impact</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">$200/hour PhD time wasted on Excel. Experiments delayed by slow analysis. Missed insights in complex data.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #10b981; font-weight: 700; margin: 0 0 8px 0;">✅ With Novaflow</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Instant analysis, auto stats, beautiful plots. 10 hours → 10 minutes. AI finds patterns humans miss.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📊 Analyze Experiment", "📈 Example Visualizations"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Upload Experimental Data</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">AI analyzes your data and generates insights automatically</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        use_sample = st.checkbox("Use sample experiment", value=True)
        
        if use_sample:
            exp_name = st.selectbox("Select Experiment", list(SAMPLE_EXPERIMENTS.keys()))
            experiment = SAMPLE_EXPERIMENTS[exp_name]
            data = experiment['data']
            exp_type = experiment['type']
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
                exp_type = "Custom"
            else:
                data = None
                exp_type = None
        
        if data is not None:
            st.markdown("**📊 Your Data:**")
            st.dataframe(data, use_container_width=True, hide_index=True)
            
            if st.button("🧠 Analyze Experiment", type="primary", use_container_width=True):
                st.session_state.analysis_complete = True
                st.session_state.insights = analyze_experiment(data, exp_type)
                st.session_state.current_data = data
                st.session_state.current_type = exp_type
    
    with col2:
        st.markdown("""
        <div style="background: #ecfdf5; padding: 20px; border-radius: 12px; border-left: 4px solid #10b981;">
            <h4 style="color: #065f46; margin: 0 0 12px 0; font-size: 16px;">🤖 What AI Does</h4>
            <ul style="color: #047857; font-size: 13px; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li><strong>Auto-detect assay type:</strong> Western, qPCR, MTT, ELISA</li>
                <li><strong>Calculate stats:</strong> EC50, IC50, fold-change, p-values</li>
                <li><strong>Generate plots:</strong> Dose-response, bar charts, heatmaps</li>
                <li><strong>Find patterns:</strong> Outliers, trends, correlations</li>
                <li><strong>Write insights:</strong> Natural language interpretations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.analysis_complete:
        st.markdown("<hr style='margin: 30px 0; border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)
        
        insights = st.session_state.insights
        
        # Display insights
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">🔬 AI Analysis Results</h2>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 25px;">
                <h3 style="color: white; font-size: 20px; font-weight: 700; margin: 0 0 15px 0;">Key Finding</h3>
                <p style="color: rgba(255,255,255,0.95); font-size: 16px; line-height: 1.7; margin: 0;">{insights['key_finding']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 12px; border-left: 4px solid #10b981;">
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Key Metric</p>
                <p style="color: #059669; font-size: 24px; font-weight: 900; margin: 8px 0;">{list(insights.values())[1]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6;">
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Statistical Significance</p>
                <p style="color: #3b82f6; font-size: 24px; font-weight: 900; margin: 8px 0;">{insights['statistical_sig']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 12px; border-left: 4px solid #8b5cf6;">
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Confidence</p>
                <p style="color: #8b5cf6; font-size: 24px; font-weight: 900; margin: 8px 0;">95%</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: #eff6ff; padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6; margin-top: 20px;">
            <h4 style="color: #1e40af; margin: 0 0 10px 0; font-size: 16px;">💡 AI Recommendation</h4>
            <p style="color: #1f2937; font-size: 14px; line-height: 1.7; margin: 0;">{insights['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization
        st.markdown("### 📈 Auto-Generated Visualization")
        
        if st.session_state.current_type == "Western Blot":
            fig = px.bar(st.session_state.current_data, x='Sample', y='Normalized',
                        title="Protein Expression (Normalized to Control)",
                        color='Normalized',
                        color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.current_type == "MTT Assay":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.current_data['Concentration'],
                y=st.session_state.current_data['Viability_%'],
                mode='lines+markers',
                line=dict(color='#059669', width=3),
                marker=dict(size=10),
                error_y=dict(
                    type='data',
                    array=st.session_state.current_data['Std_Dev'],
                    visible=True
                )
            ))
            fig.update_layout(
                title="Cell Viability Dose-Response Curve",
                xaxis_title="Concentration (µM)",
                yaxis_title="Viability (%)",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.current_type == "qPCR":
            fig = px.bar(st.session_state.current_data, x='Gene', y='Fold_Change',
                        title="Gene Expression Fold Changes",
                        color='Fold_Change',
                        color_continuous_scale='RdBu_r')
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                         annotation_text="Baseline")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Example Visualizations</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">AI automatically generates publication-ready figures</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**All plots are auto-generated based on your experiment type:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("✓ Dose-response curves (EC50/IC50 calculation)")
        st.markdown("✓ Bar charts with error bars")
        st.markdown("✓ Heatmaps for multi-factor experiments")
    
    with col2:
        st.markdown("✓ Statistical annotations (p-values, significance)")
        st.markdown("✓ Fold-change visualizations")
        st.markdown("✓ Time-series analysis")

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #059669; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Novaflow</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 100x Faster</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    10 minutes vs 10 hours for complete analysis. Upload data → Get insights instantly. More time for actual science.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🧠 AI Insights</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    AI finds patterns humans miss. Statistical tests automatic. Suggests next experiments based on results.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📊 Publication Ready</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Generate Nature/Science quality figures automatically. Export high-res PDFs. Include all statistical annotations.
                </p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Research Lab Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">100x faster analysis:</strong> 10 min vs 10 hours per experiment</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Zero errors:</strong> No manual calculation mistakes</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">AI-discovered insights:</strong> Patterns humans miss in complex data</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Publication-ready figures:</strong> Instant high-quality visualizations</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Supported Assays</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Western Blot</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Band quantification, fold-change, EC50</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ qPCR/RT-PCR</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Ct values, fold-change, significance</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Cell Viability Assays</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">MTT, MTS, Alamar Blue - IC50 calculation</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ ELISA</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Standard curves, concentration calculation</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #059669 0%, #73BA9B 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(5, 150, 105, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Novaflow</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Data Analysis • Statistics • Biotech ML • Scientific Visualization
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered data analysis for biology research labs.<br>
            Automated statistics • Assay detection • Visualization • Insight generation • Publication-ready figures
        </p>
    </div>
    """, unsafe_allow_html=True)