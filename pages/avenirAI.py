"""
Avenir AI - Employee Benefits Intelligence Platform
AI agents for HR benefits automation and cost optimization
Built for Avenir AI by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Avenir AI - Benefits Intelligence", page_icon="💼", layout="wide")

# Benefits categories and costs
BENEFITS_CATEGORIES = {
    'Health Insurance': {'current_cost': 850000, 'employees': 500, 'utilization': 78, 'potential_savings': 127500},
    'Dental & Vision': {'current_cost': 120000, 'employees': 500, 'utilization': 65, 'potential_savings': 18000},
    'Life Insurance': {'current_cost': 45000, 'employees': 500, 'utilization': 100, 'potential_savings': 4500},
    '401(k) Match': {'current_cost': 380000, 'employees': 500, 'utilization': 72, 'potential_savings': 0},
    'Wellness Programs': {'current_cost': 95000, 'employees': 500, 'utilization': 42, 'potential_savings': 28500},
    'Mental Health': {'current_cost': 72000, 'employees': 500, 'utilization': 38, 'potential_savings': 0},
    'Commuter Benefits': {'current_cost': 48000, 'employees': 500, 'utilization': 55, 'potential_savings': 9600}
}

# AI agent capabilities
AI_AGENTS = {
    'Benefits Optimization Agent': 'Analyzes plans, finds cost savings, recommends alternatives',
    'Employee Insights Agent': 'Tracks utilization, engagement, satisfaction patterns',
    'Compliance Agent': 'Monitors ACA, ERISA, COBRA requirements automatically',
    'Vendor Management Agent': 'Negotiates rates, manages renewals, benchmarks pricing',
    'Enrollment Agent': 'Automates open enrollment, answers employee questions',
    'Analytics Agent': 'Generates reports, forecasts costs, identifies trends'
}

# Platform metrics
PLATFORM_METRICS = {
    'Cost Reduction': {'value': 15.8, 'baseline': '$1.61M', 'optimized': '$1.36M'},
    'Time Saved': {'value': 420, 'baseline': '520 hrs/month', 'optimized': '100 hrs/month'},
    'Employee Satisfaction': {'value': 4.7, 'baseline': '3.2/5', 'optimized': '4.7/5'},
    'Plan Utilization': {'value': 68, 'baseline': '52%', 'optimized': '68%'}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">💼</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Avenir AI</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Employee Benefits Intelligence</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">AI agents • Cost optimization • Data intelligence</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["💼 Benefits Dashboard", "🤖 AI Optimization", "📊 Cost Analytics", "💡 Platform Intelligence"])

with tab1:
    st.markdown("### Benefits Intelligence Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">$1.61M</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Annual Benefits Cost</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ec4899 0%, #f97316 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">500</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Employees Covered</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f97316 0%, #eab308 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">$254K</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Potential Savings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #22c55e 0%, #10b981 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">15.8%</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Cost Reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Benefits Breakdown by Category")
        
        benefits_data = []
        for category, data in BENEFITS_CATEGORIES.items():
            per_employee = data['current_cost'] / data['employees']
            benefits_data.append({
                'Category': category,
                'Annual Cost': f"${data['current_cost']:,}",
                'Per Employee': f"${per_employee:.0f}",
                'Utilization': f"{data['utilization']}%"
            })
        
        st.dataframe(pd.DataFrame(benefits_data), hide_index=True, use_container_width=True)
        
        st.markdown("### Utilization vs Industry Benchmark")
        
        categories = list(BENEFITS_CATEGORIES.keys())
        utilization = [BENEFITS_CATEGORIES[c]['utilization'] for c in categories]
        benchmark = [85, 75, 100, 80, 68, 55, 72]
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Bar(
            x=categories,
            y=utilization,
            name='Your Company',
            marker_color='#7c3aed'
        ))
        
        fig1.add_trace(go.Bar(
            x=categories,
            y=benchmark,
            name='Industry Avg',
            marker_color='#94a3b8'
        ))
        
        fig1.update_layout(
            barmode='group',
            yaxis_title='Utilization (%)',
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### Cost Distribution")
        
        categories_list = list(BENEFITS_CATEGORIES.keys())
        costs = [BENEFITS_CATEGORIES[c]['current_cost'] for c in categories_list]
        
        fig2 = px.pie(
            values=costs,
            names=categories_list,
            color_discrete_sequence=['#7c3aed', '#ec4899', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6']
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=300, showlegend=False)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### AI Insights")
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; border-left: 5px solid #f59e0b;">
            <h4 style="margin: 0 0 12px 0; color: #92400e;">⚠️ Action Required</h4>
            <ul style="margin: 0; padding-left: 20px; color: #78350f;">
                <li>Health insurance renewal in 45 days - AI found 3 better options</li>
                <li>Wellness program utilization 26% below target - engagement needed</li>
                <li>Dental plan overfunded by $18K - recommend plan adjustment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #dcfce7; padding: 20px; border-radius: 12px; border-left: 5px solid #22c55e; margin-top: 15px;">
            <h4 style="margin: 0 0 12px 0; color: #166534;">✓ Opportunities</h4>
            <ul style="margin: 0; padding-left: 20px; color: #15803d;">
                <li>Switch to high-deductible plan + HSA: Save $127K annually</li>
                <li>Consolidate vision/dental vendors: Save $18K annually</li>
                <li>Virtual mental health upgrade: Improve utilization by 22%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### AI-Powered Benefits Optimization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Company Profile**")
        
        company_size = st.selectbox("Company Size", ["50-100", "100-250", "250-500", "500-1000", "1000+"], index=2)
        industry = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail"])
        region = st.selectbox("Primary Location", ["San Francisco Bay Area", "New York", "Austin", "Seattle", "Boston"])
        
        st.markdown("**Current Challenges**")
        
        challenges = st.multiselect(
            "Select pain points",
            ["High costs", "Low utilization", "Poor employee satisfaction", "Admin burden", "Compliance issues"],
            ["High costs", "Admin burden"]
        )
        
        st.markdown("**Optimization Goals**")
        
        primary_goal = st.radio(
            "Primary objective",
            ["Reduce costs", "Improve employee satisfaction", "Increase utilization", "Simplify admin"]
        )
        
        optimize_btn = st.button("🤖 Run AI Optimization", type="primary", use_container_width=True)
    
    with col2:
        if optimize_btn:
            st.markdown("**AI Analysis in Progress**")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            stages = [
                ("Analyzing current benefits structure...", 0.2),
                ("Benchmarking against 10,000+ similar companies...", 0.4),
                ("Evaluating 50+ alternative plan configurations...", 0.6),
                ("Calculating ROI for each optimization...", 0.8),
                ("Generating actionable recommendations...", 1.0)
            ]
            
            for stage, progress in stages:
                status_text.text(stage)
                progress_bar.progress(progress)
                time.sleep(0.5)
            
            st.success("✅ AI optimization complete!")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%); padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Optimization Summary</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Annual Savings</p>
                        <p style="font-size: 28px; color: white; font-weight: 900; margin: 0;">$254K</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Cost Reduction</p>
                        <p style="font-size: 28px; color: white; font-weight: 900; margin: 0;">15.8%</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Implementation</p>
                        <p style="font-size: 28px; color: white; font-weight: 900; margin: 0;">60 days</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Top 5 Recommendations")
            
            recommendations = [
                {
                    'recommendation': 'Switch to High-Deductible Health Plan + HSA',
                    'savings': '$127,500',
                    'impact': 'High',
                    'difficulty': 'Medium',
                    'timeline': '90 days'
                },
                {
                    'recommendation': 'Consolidate Dental/Vision Vendors',
                    'savings': '$18,000',
                    'impact': 'Medium',
                    'difficulty': 'Low',
                    'timeline': '30 days'
                },
                {
                    'recommendation': 'Redesign Wellness Program (Virtual-First)',
                    'savings': '$28,500',
                    'impact': 'High',
                    'difficulty': 'Medium',
                    'timeline': '60 days'
                },
                {
                    'recommendation': 'Optimize 401(k) Matching Structure',
                    'savings': '$45,000',
                    'impact': 'Medium',
                    'difficulty': 'Low',
                    'timeline': '45 days'
                },
                {
                    'recommendation': 'Implement Tiered Mental Health Benefits',
                    'savings': '$15,000',
                    'impact': 'High',
                    'difficulty': 'Medium',
                    'timeline': '60 days'
                }
            ]
            
            st.dataframe(pd.DataFrame(recommendations), hide_index=True, use_container_width=True)
            
            st.markdown("### Implementation Roadmap")
            
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #7c3aed;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">Phase 1 (Days 1-30): Quick Wins</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">• Consolidate dental/vision vendors<br>• Optimize commuter benefits structure</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #ec4899; margin-top: 10px;">
                <p style="margin: 0; font-weight: 700; color: #be185d;">Phase 2 (Days 31-60): Medium Impact</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">• Redesign wellness program<br>• Implement tiered mental health benefits</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #f97316; margin-top: 10px;">
                <p style="margin: 0; font-weight: 700; color: #c2410c;">Phase 3 (Days 61-90): Major Transformation</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">• Switch to HDHP + HSA<br>• Renegotiate health insurance contract</p>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Cost Analytics & ROI")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">420</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Hours Saved Monthly</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ec4899 0%, #f97316 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">+47%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Satisfaction Increase</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f97316 0%, #eab308 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">+31%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Utilization Improvement</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cost Trend (12 Months)")
        
        months = ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        baseline = [135, 134, 136, 138, 137, 139, 138, 137, 136, 135, 134, 135]
        with_avenir = [135, 134, 132, 128, 125, 122, 118, 115, 113, 113, 113, 113]
        
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=months,
            y=baseline,
            name='Without Avenir',
            line=dict(color='#94a3b8', width=3, dash='dash')
        ))
        
        fig3.add_trace(go.Scatter(
            x=months,
            y=with_avenir,
            name='With Avenir AI',
            line=dict(color='#7c3aed', width=3),
            fill='tonexty',
            fillcolor='rgba(124, 58, 237, 0.1)'
        ))
        
        fig3.update_layout(
            yaxis_title='Monthly Cost ($K)',
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("### Savings Breakdown")
        
        savings_data = {
            'Category': ['Health Insurance', 'Wellness Programs', 'Dental/Vision', '401(k) Optimization', 'Mental Health', 'Other'],
            'Annual Savings': ['$127.5K', '$28.5K', '$18.0K', '$45.0K', '$15.0K', '$20.1K']
        }
        
        st.dataframe(pd.DataFrame(savings_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### ROI Analysis")
        
        roi_data = pd.DataFrame({
            'Metric': ['Before Avenir', 'After Avenir', 'Improvement'],
            'Annual Cost': ['$1.61M', '$1.36M', '↓ $254K'],
            'Per Employee': ['$3,220', '$2,711', '↓ $509'],
            'Admin Hours': ['520/mo', '100/mo', '↓ 420/mo']
        })
        
        st.dataframe(roi_data, hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; margin-top: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #92400e;">💰 Total Value Created</h4>
            <p style="font-size: 32px; font-weight: 900; color: #92400e; margin: 0;">$412K</p>
            <p style="margin: 8px 0 0 0; color: #78350f; font-size: 14px;">Annually (cost savings + time value)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Payback Period")
        
        months_list = list(range(1, 13))
        cumulative_savings = [21, 42, 64, 87, 110, 134, 159, 184, 210, 236, 263, 290]
        
        fig4 = go.Figure()
        
        fig4.add_trace(go.Scatter(
            x=months_list,
            y=cumulative_savings,
            mode='lines+markers',
            line=dict(color='#22c55e', width=3),
            fill='tozeroy',
            fillcolor='rgba(34, 197, 94, 0.1)'
        ))
        
        fig4.add_hline(y=50, line_dash="dash", line_color="red", 
                      annotation_text="Avenir AI Cost")
        
        fig4.update_layout(
            xaxis_title='Months',
            yaxis_title='Cumulative Savings ($K)',
            height=250
        )
        
        st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.markdown("### Platform Intelligence & Automation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**AI Agent Capabilities**")
        
        for agent, description in AI_AGENTS.items():
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #7c3aed;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">{agent}</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**Data Intelligence**")
        st.markdown("""
        - ✅ 10,000+ company benchmark database
        - ✅ Real-time benefits pricing data
        - ✅ Employee utilization patterns
        - ✅ Industry-specific insights
        - ✅ Compliance monitoring (ACA, ERISA, COBRA)
        - ✅ Vendor performance tracking
        - ✅ Predictive cost modeling
        """)
    
    with col2:
        st.markdown("**Automation Workflows**")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ede9fe 0%, #fce7f3 100%); padding: 20px; border-radius: 12px;">
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">1. Continuous Monitoring</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">AI tracks costs, utilization, satisfaction 24/7</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">2. Opportunity Detection</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Identifies savings, flags issues proactively</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">3. Automated Analysis</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Benchmarks plans, calculates ROI, models scenarios</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">4. Vendor Management</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Negotiates renewals, manages contracts</p>
            </div>
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">5. Employee Support</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Answers questions, handles enrollment automation</p>
            </div>
            <div>
                <p style="margin: 0; font-weight: 700; color: #5b21b6;">6. Reporting & Compliance</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #6b21a8;">Auto-generates reports, tracks compliance</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Integration Points**")
        st.markdown("""
        - 🏢 HRIS (Workday, ADP, BambooHR)
        - 💳 Payroll systems
        - 🏥 Insurance carrier portals
        - 📊 Benefits administration platforms
        - 📧 Email & Slack for notifications
        - 📱 Employee mobile apps
        """)
        
        st.markdown("**Team Background**")
        
        team_data = {
            'Background': ['MIT Research Labs', 'Google DeepMind', 'Microsoft Copilot', 'Apple B2B AI'],
            'Experience': ['AI for workplace wellbeing', 'Large language models', 'Enterprise AI products', 'B2B product development']
        }
        
        st.dataframe(pd.DataFrame(team_data), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #ede9fe 0%, #fce7f3 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #5b21b6; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ 15.8% Cost Reduction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">$254K annual savings</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ 420 Hours Saved</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Monthly admin reduction</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ 6 AI Agents</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">End-to-end automation</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ 47% Satisfaction Boost</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Employee experience</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Avenir AI</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)