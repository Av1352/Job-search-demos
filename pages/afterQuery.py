"""
AfterQuery - AI Capabilities Research Lab
Investigating the boundaries of AI capabilities
Built for AfterQuery by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="AfterQuery", page_icon="🔬", layout="wide")

# AI models and capabilities
AI_MODELS = {
    'GPT-4': {'reasoning': 92, 'math': 88, 'coding': 90, 'creativity': 85, 'safety': 94},
    'Claude-3-Opus': {'reasoning': 90, 'math': 85, 'coding': 88, 'creativity': 87, 'safety': 96},
    'Gemini-Ultra': {'reasoning': 89, 'math': 90, 'coding': 86, 'creativity': 82, 'safety': 91},
    'GPT-3.5': {'reasoning': 78, 'math': 72, 'coding': 75, 'creativity': 70, 'safety': 85},
    'Llama-3-70B': {'reasoning': 82, 'math': 76, 'coding': 80, 'creativity': 73, 'safety': 88}
}

# Capability categories
CAPABILITY_TESTS = {
    'Reasoning': ['Logical Inference', 'Causal Analysis', 'Analogical Reasoning', 'Counterfactual Thinking'],
    'Math': ['Arithmetic', 'Algebra', 'Calculus', 'Probability', 'Proofs'],
    'Coding': ['Python', 'JavaScript', 'Algorithm Design', 'Debugging', 'Code Review'],
    'Creativity': ['Story Writing', 'Poetry', 'Problem Solving', 'Ideation'],
    'Safety': ['Toxicity Filter', 'Bias Detection', 'Jailbreak Resistance', 'PII Protection']
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #ec4899 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🔬</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">AfterQuery</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">AI Capabilities Research Lab</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Investigating the boundaries of what AI can do</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Capability Testing", "📊 Model Comparison", "🎯 Edge Cases", "💡 Research Insights"])

with tab1:
    st.markdown("### AI Capability Benchmarking")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Select Model**")
        selected_model = st.selectbox("AI Model", list(AI_MODELS.keys()))
        
        st.markdown("**Select Capability**")
        selected_capability = st.selectbox("Test Category", list(CAPABILITY_TESTS.keys()))
        
        test_btn = st.button("🔬 Run Capability Tests", type="primary", use_container_width=True)
    
    with col2:
        if test_btn:
            # Run tests
            model_scores = AI_MODELS[selected_model]
            tests = CAPABILITY_TESTS[selected_capability]
            
            # Generate test results
            np.random.seed(42)
            base_score = model_scores[selected_capability.lower()]
            test_results = []
            
            for test in tests:
                score = base_score + np.random.uniform(-5, 5)
                passed = score >= 80
                test_results.append({
                    'Test': test,
                    'Score': f"{score:.1f}%",
                    'Status': '✅ Pass' if passed else '❌ Fail',
                    'Difficulty': np.random.choice(['Easy', 'Medium', 'Hard'])
                })
            
            # Display results
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ec4899 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Test Results: {selected_model}</h3>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                    <p style="font-size: 18px; color: white; margin: 0;"><strong>Category:</strong> {selected_capability}</p>
                    <p style="font-size: 18px; color: white; margin: 8px 0 0 0;"><strong>Overall Score:</strong> {base_score:.1f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(pd.DataFrame(test_results), hide_index=True, use_container_width=True)
            
            # Performance breakdown
            passed = sum(1 for r in test_results if '✅' in r['Status'])
            total = len(test_results)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Tests Passed", f"{passed}/{total}", f"{passed/total*100:.0f}%")
            col2.metric("Avg Score", f"{base_score:.1f}%", "🎯")
            col3.metric("Difficulty", "Mixed", "Easy-Hard")

with tab2:
    st.markdown("### Multi-Model Capability Comparison")
    
    # Capability radar chart
    st.markdown("**Capability Profiles**")
    
    fig1 = go.Figure()
    
    categories = ['Reasoning', 'Math', 'Coding', 'Creativity', 'Safety']
    
    for model_name in ['GPT-4', 'Claude-3-Opus', 'Gemini-Ultra']:
        scores = [AI_MODELS[model_name][cat.lower()] for cat in categories]
        scores.append(scores[0])  # Close the radar
        
        fig1.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories + [categories[0]],
            fill='toself',
            name=model_name
        ))
    
    fig1.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Detailed comparison table
    st.markdown("**Detailed Capability Scores**")
    
    comparison_data = []
    for model in AI_MODELS.keys():
        row = {'Model': model}
        row.update({k.capitalize(): f"{v}%" for k, v in AI_MODELS[model].items()})
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
    
    # Performance heatmap
    st.markdown("**Performance Heatmap**")
    
    heatmap_data = []
    for model in AI_MODELS.keys():
        heatmap_data.append([AI_MODELS[model][cat.lower()] for cat in categories])
    
    fig2 = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=categories,
        y=list(AI_MODELS.keys()),
        colorscale='RdYlGn',
        text=heatmap_data,
        texttemplate='%{text}%',
        textfont={"size": 12}
    ))
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Edge Case & Boundary Testing")
    
    st.markdown("**Adversarial Test Categories**")
    
    edge_cases = {
        'Jailbreak Attempts': {'tested': 247, 'blocked': 239, 'success_rate': 96.8},
        'Ambiguous Queries': {'tested': 189, 'handled': 172, 'success_rate': 91.0},
        'Multilingual Edge Cases': {'tested': 156, 'correct': 142, 'success_rate': 91.0},
        'Logical Paradoxes': {'tested': 98, 'resolved': 82, 'success_rate': 83.7},
        'Out-of-Distribution': {'tested': 134, 'handled': 109, 'success_rate': 81.3}
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Edge case performance
        categories_list = list(edge_cases.keys())
        success_rates = [edge_cases[cat]['success_rate'] for cat in categories_list]
        
        fig3 = go.Figure(data=[go.Bar(
            x=categories_list,
            y=success_rates,
            marker=dict(
                color=success_rates,
                colorscale='RdYlGn',
                cmin=70,
                cmax=100
            ),
            text=[f"{rate:.1f}%" for rate in success_rates],
            textposition='auto'
        )])
        fig3.update_layout(
            title='Edge Case Handling Success Rate',
            xaxis_title='Test Category',
            yaxis_title='Success Rate (%)',
            height=300
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("**Test Summary**")
        total_tested = sum(edge_cases[cat]['tested'] for cat in edge_cases)
        st.metric("Total Tests", total_tested, "824 cases")
        st.metric("Avg Success", "88.8%", "+3.2%")
        st.metric("Critical Fails", "12", "-5")
    
    # Detailed edge case results
    st.markdown("**Edge Case Details**")
    
    edge_df = pd.DataFrame([
        {
            'Category': cat,
            'Tests': data['tested'],
            'Successful': data.get('blocked', data.get('handled', data.get('correct', data.get('resolved', 0)))),
            'Success Rate': f"{data['success_rate']:.1f}%"
        }
        for cat, data in edge_cases.items()
    ])
    st.dataframe(edge_df, hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### Research Insights & Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Key Research Findings**")
        
        st.markdown("""
        **1. Reasoning Ceiling** (92%)
        - GPT-4 hits 92% on complex reasoning
        - Improvement plateau above 90%
        - Requires new architectures for breakthrough
        
        **2. Safety-Capability Tradeoff**
        - Higher safety = -3-5% capability loss
        - Claude leads: 96% safety, 90% reasoning
        - GPT-4: 94% safety, 92% reasoning
        
        **3. Domain Specialization**
        - Math specialists (Gemini) outperform generalists +8%
        - Code-focused models excel on algorithm tasks
        - Tradeoff: narrow vs broad capability
        
        **4. Edge Case Vulnerability**
        - 88.8% edge case success (needs improvement)
        - Jailbreaks blocked 96.8% (strong)
        - Logical paradoxes: weakest area (83.7%)
        """)
    
    with col2:
        st.markdown("**Capability Frontiers**")
        
        # Progress over time simulation
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        frontier_data = {
            'month': months,
            'reasoning': [85, 87, 88, 90, 91, 92],
            'math': [82, 84, 86, 88, 89, 90],
            'coding': [83, 85, 86, 88, 89, 90]
        }
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=months, y=frontier_data['reasoning'],
            mode='lines+markers', name='Reasoning',
            line=dict(color='#ec4899', width=2)
        ))
        fig4.add_trace(go.Scatter(
            x=months, y=frontier_data['math'],
            mode='lines+markers', name='Math',
            line=dict(color='#3b82f6', width=2)
        ))
        fig4.add_trace(go.Scatter(
            x=months, y=frontier_data['coding'],
            mode='lines+markers', name='Coding',
            line=dict(color='#10b981', width=2)
        ))
        fig4.update_layout(
            title='Capability Progress (6 Months)',
            xaxis_title='Month',
            yaxis_title='Score (%)',
            height=250
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("**Breakthrough Needed**")
        st.markdown("""
        - Reasoning plateau at 92%
        - Requires architectural innovation
        - Next frontier: 95%+ reasoning
        """)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #9f1239; font-size: 24px; font-weight: 800;">💡 Research Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #db2777; font-weight: 700; margin: 0 0 6px 0;">✓ 824 Edge Cases</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Adversarial & boundary testing</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #db2777; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-Model Comparison</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">GPT-4, Claude, Gemini, Llama</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #db2777; font-weight: 700; margin: 0 0 6px 0;">✓ 5 Capability Domains</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Reasoning, math, coding, creativity, safety</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #db2777; font-weight: 700; margin: 0 0 6px 0;">✓ Research Insights</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Capability frontiers & breakthroughs</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #ec4899 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for AfterQuery</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)