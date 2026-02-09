"""
Superpower - Preventative Health Super App
100+ biomarker testing with AI-powered insights
Built for Superpower by Anju Vilashni Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Superpower - Health Super App", page_icon="⚡", layout="wide")

# Biomarker categories
BIOMARKER_CATEGORIES = {
    'Hormones': {'count': 15, 'optimal': 92, 'tests': ['Testosterone', 'Estrogen', 'Thyroid panel', 'Cortisol', 'DHEA']},
    'Metabolic': {'count': 18, 'optimal': 85, 'tests': ['Glucose', 'Insulin', 'HbA1c', 'Lipid panel', 'Uric acid']},
    'Cardiovascular': {'count': 12, 'optimal': 88, 'tests': ['ApoB', 'Lp(a)', 'hs-CRP', 'Homocysteine', 'BP']},
    'Immune': {'count': 10, 'optimal': 90, 'tests': ['WBC', 'Lymphocytes', 'Antibodies', 'Cytokines']},
    'Inflammation': {'count': 8, 'optimal': 78, 'tests': ['CRP', 'IL-6', 'TNF-alpha', 'ESR']},
    'Liver': {'count': 14, 'optimal': 94, 'tests': ['ALT', 'AST', 'GGT', 'Bilirubin', 'Albumin']},
    'Kidney': {'count': 9, 'optimal': 91, 'tests': ['Creatinine', 'eGFR', 'BUN', 'Electrolytes']},
    'Nutrients': {'count': 16, 'optimal': 82, 'tests': ['Vitamin D', 'B12', 'Folate', 'Iron', 'Magnesium']}
}

# Health insights
HEALTH_INSIGHTS = {
    'Strengths': [
        'Excellent cardiovascular health (ApoB optimal)',
        'Strong immune function (all markers normal)',
        'Optimal liver health (ALT/AST perfect)',
        'Good metabolic control (glucose 88 mg/dL)'
    ],
    'Action Items': [
        'Vitamin D low (22 ng/mL) - supplement 2000 IU daily',
        'Inflammation markers elevated - reduce processed foods',
        'Sleep quality suboptimal - target 8 hours consistently',
        'Omega-3 index low - increase fatty fish or supplement'
    ]
}

# Platform features
PLATFORM_FEATURES = {
    'Testing': '100+ biomarkers, twice yearly, $499/year',
    'AI Analysis': 'Superpower Score + Biological Age calculation',
    'Data Integration': 'Medical records + wearables + genetics',
    'Concierge Care': '24/7 AI + human hybrid support team',
    'Marketplace': 'Curated health products & services',
    'Tracking': 'Longitudinal health trends over time'
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">⚡</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Superpower</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Your Personal Health OS</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">100+ biomarkers • AI insights • $499/year</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["⚡ Health Dashboard", "🧪 Lab Results", "💡 AI Insights", "📊 Trends"])

with tab1:
    st.markdown("### Your Health Command Center")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">87</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Superpower Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">38.5</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Biological Age</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">102</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Biomarkers Tracked</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">3</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Action Items</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Quick Stats")
        
        quick_stats = pd.DataFrame({
            'Category': ['Last Test Date', 'Next Test Due', 'Tests Completed', 'Membership', 'Partner Labs'],
            'Value': ['Jan 15, 2026', 'Jul 15, 2026', '2', 'Active ($499/yr)', '2,000 nationwide']
        })
        
        st.dataframe(quick_stats, hide_index=True, use_container_width=True)
        
        st.markdown("### Recent Activity")
        
        st.markdown("""
        <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #6366f1;">
            <p style="margin: 0; font-weight: 600; color: #4338ca;">Lab Results Received</p>
            <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">Jan 18, 2026 • 102 biomarkers analyzed</p>
        </div>
        <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #22c55e;">
            <p style="margin: 0; font-weight: 600; color: #166534;">Action Plan Updated</p>
            <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">Jan 19, 2026 • 3 new recommendations</p>
        </div>
        <div style="background: white; padding: 15px; border-radius: 10px; border-left: 4px solid #f59e0b;">
            <p style="margin: 0; font-weight: 600; color: #92400e;">Concierge Message</p>
            <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">Jan 20, 2026 • Check your Vitamin D levels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Health Score Breakdown")
        
        categories = ['Hormones', 'Metabolic', 'Cardio', 'Immune', 'Liver', 'Kidney', 'Nutrients', 'Inflammation']
        scores = [92, 85, 88, 90, 94, 91, 82, 78]
        
        fig1 = go.Figure(data=go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            marker=dict(color='#6366f1'),
            line=dict(color='#6366f1', width=2)
        ))
        
        fig1.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.markdown("### Comprehensive Biomarker Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Test Categories")
        
        biomarker_data = []
        for category, data in BIOMARKER_CATEGORIES.items():
            status = "🟢 Optimal" if data['optimal'] >= 85 else "🟡 Monitor" if data['optimal'] >= 70 else "🔴 Attention"
            biomarker_data.append({
                'Category': category,
                'Biomarkers': data['count'],
                'Score': f"{data['optimal']}%",
                'Status': status
            })
        
        st.dataframe(pd.DataFrame(biomarker_data), hide_index=True, use_container_width=True)
        
        st.markdown("### Key Biomarkers")
        
        key_biomarkers = pd.DataFrame({
            'Test': ['Vitamin D', 'hs-CRP', 'HbA1c', 'ApoB', 'Testosterone', 'TSH'],
            'Result': ['22 ng/mL', '1.8 mg/L', '5.2%', '75 mg/dL', '520 ng/dL', '1.8 mIU/L'],
            'Optimal Range': ['30-50', '<1.0', '<5.7%', '<80', '400-700', '0.5-4.5'],
            'Status': ['🟡 Low', '🟡 Elevated', '🟢 Good', '🟢 Good', '🟢 Good', '🟢 Good']
        })
        
        st.dataframe(key_biomarkers, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### Biomarker Distribution")
        
        categories_list = list(BIOMARKER_CATEGORIES.keys())
        biomarker_counts = [BIOMARKER_CATEGORIES[cat]['count'] for cat in categories_list]
        
        fig2 = px.pie(
            values=biomarker_counts,
            names=categories_list,
            color_discrete_sequence=['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#22c55e', '#3b82f6', '#14b8a6', '#f97316']
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=300, showlegend=False)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### vs Standard Physical")
        
        comparison = pd.DataFrame({
            'Test Coverage': ['Standard Annual Physical', 'Superpower Testing'],
            'Biomarkers': [10, 102],
            'Frequency': ['1x/year', '2x/year'],
            'Cost': ['Copay ($25-50)', '$499/year']
        })
        
        st.dataframe(comparison, hide_index=True, use_container_width=True)

with tab3:
    st.markdown("### AI-Powered Health Recommendations")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Your Strengths")
        
        for strength in HEALTH_INSIGHTS['Strengths']:
            st.markdown(f"""
            <div style="background: #dcfce7; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #22c55e;">
                <p style="margin: 0; color: #15803d; font-size: 14px;">✓ {strength}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Personalized Action Plan")
        
        for i, action in enumerate(HEALTH_INSIGHTS['Action Items'], 1):
            priority_color = '#f59e0b' if i <= 2 else '#3b82f6'
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid {priority_color};">
                <p style="margin: 0; font-weight: 600; color: #1f2937;">{i}. {action}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### AI Health Assistant")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); padding: 20px; border-radius: 12px;">
            <h4 style="margin: 0 0 15px 0; color: #3730a3;">💬 Ask Your Concierge</h4>
            <p style="margin: 0 0 10px 0; color: #4338ca; font-size: 14px;">24/7 AI + human hybrid support available</p>
        </div>
        """, unsafe_allow_html=True)
        
        question = st.text_area("Ask a health question", 
                               "Why is my Vitamin D low and what should I do?",
                               height=80)
        
        if st.button("💬 Get Answer", use_container_width=True):
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #6366f1; margin-top: 15px;">
                <p style="margin: 0 0 10px 0; font-weight: 700; color: #4338ca;">AI Concierge Response:</p>
                <p style="margin: 0; color: #666; font-size: 14px; line-height: 1.6;">
                Based on your results (22 ng/mL), your Vitamin D is below the optimal range (30-50 ng/mL). This is common, especially in winter months or with limited sun exposure.
                <br><br>
                <strong>Recommended:</strong><br>
                • Supplement with 2,000 IU Vitamin D3 daily<br>
                • Get 15-20 min sun exposure when possible<br>
                • Retest in 3 months to monitor improvement<br>
                • Consider adding Vitamin K2 for synergy
                <br><br>
                Our marketplace has curated Vitamin D supplements. Would you like product recommendations?
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Recommended Products")
        
        products = pd.DataFrame({
            'Product': ['Thorne Vitamin D3', 'Life Extension D3+K2', 'Nordic Naturals D3'],
            'Dosage': ['2,000 IU', '2,000 IU + 45mcg K2', '1,000 IU'],
            'Price': ['$18/mo', '$24/mo', '$16/mo']
        })
        
        st.dataframe(products, hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### Longitudinal Health Tracking")
    
    # Key impact metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">+5.2</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Score Improvement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">-3.5yr</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Biological Age Change</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">6mo</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Tracking Period</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Superpower Score Trend")
        
        months = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        scores = [81.5, 82.3, 83.8, 84.5, 85.2, 86.1, 87.0]
        
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=months,
            y=scores,
            mode='lines+markers',
            line=dict(color='#6366f1', width=3),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)'
        ))
        
        fig3.update_layout(
            yaxis_title='Superpower Score',
            height=250,
            yaxis_range=[75, 95]
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("### Key Biomarker Improvements")
        
        improvements = pd.DataFrame({
            'Biomarker': ['Vitamin D', 'hs-CRP', 'HbA1c', 'Omega-3 Index'],
            'Before': ['22 ng/mL', '2.8 mg/L', '5.4%', '4.2%'],
            'Current': ['38 ng/mL', '1.8 mg/L', '5.2%', '6.8%'],
            'Change': ['↑ 73%', '↓ 36%', '↓ 4%', '↑ 62%']
        })
        
        st.dataframe(improvements, hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; margin-top: 15px;">
            <h4 style="margin: 0 0 10px 0; color: #92400e;">🎯 6-Month Progress</h4>
            <p style="margin: 8px 0; color: #78350f; font-size: 14px;">
            <strong>Score:</strong> 81.5 → 87.0 (+5.5 points)<br>
            <strong>Bio Age:</strong> 42 → 38.5 (-3.5 years)<br>
            <strong>Action Items Completed:</strong> 7/10<br>
            <strong>Overall:</strong> Significant improvement ✓
            </p>
        </div>
        """, unsafe_allow_html=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #3730a3; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #6366f1; font-weight: 700; margin: 0 0 6px 0;">✓ 100+ Biomarkers</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Twice yearly testing</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #6366f1; font-weight: 700; margin: 0 0 6px 0;">✓ $30M Series A</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Forerunner-led</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #6366f1; font-weight: 700; margin: 0 0 6px 0;">✓ 24/7 AI Concierge</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Hybrid AI + human</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #6366f1; font-weight: 700; margin: 0 0 6px 0;">✓ 150K Waitlist</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Now live</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Superpower</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)