"""
Dryft - Agentic Operating System for Manufacturing
AI agents + mathematical optimization for production planning
Built for Dryft by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Dryft - Manufacturing AI", page_icon="🏭", layout="wide")

# AI decision types
AI_DECISIONS = {
    'Order Quantity Optimization': {'frequency': '580/month', 'accuracy': 94.2, 'savings': '$245K/year'},
    'Safety Stock Adjustment': {'frequency': '420/month', 'accuracy': 96.8, 'savings': '$180K/year'},
    'Supplier Follow-up': {'frequency': '850/month', 'accuracy': 98.1, 'savings': '$95K/year'},
    'Production Rescheduling': {'frequency': '340/month', 'accuracy': 91.5, 'savings': '$320K/year'},
    'Inventory Rebalancing': {'frequency': '290/month', 'accuracy': 93.7, 'savings': '$155K/year'}
}

# Platform features
PLATFORM_FEATURES = {
    'Context-Aware Agents': 'Understands factory constraints, supplier relationships, production dependencies',
    'Mathematical Optimization': 'Linear programming, constraint solving for optimal decisions',
    'Predictive Modeling': 'Forecasts delays, demand changes, supply disruptions',
    'Auto-Execution': 'Takes action automatically (orders, schedules, communications)',
    'Continuous Learning': 'Improves from outcomes, adapts to factory patterns',
    'ERP Integration': 'Replaces or augments legacy systems'
}

# Customer results
CUSTOMER_RESULTS = {
    'Inventory Carrying Costs': {'reduction': 28, 'annual_savings': 1.2},
    'Lead Time Reduction': {'reduction': 42, 'annual_savings': 0.8},
    'Stock-out Prevention': {'reduction': 65, 'annual_savings': 1.5},
    'Administrative Time': {'reduction': 75, 'annual_savings': 0.9}
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🏭</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Dryft</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Agentic Operating System for Manufacturing</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">$5M General Catalyst • Enterprise Resource Automation • Ex-Porsche</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏭 AI Decision Engine", "📊 Operations Dashboard", "📈 Customer Results", "💡 Technology"])

with tab1:
    st.markdown("### AI-Powered Manufacturing Decisions")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Scenario Simulation**")
        
        st.markdown("**Trigger Event**")
        
        event_type = st.selectbox("What happened?",
                                 ["Supplier delayed shipment",
                                  "Design change from engineering",
                                  "Demand spike from sales",
                                  "Machine breakdown",
                                  "Material quality issue"])
        
        st.markdown("**Current State**")
        
        current_inventory = st.number_input("Current Inventory (units)", 100, 10000, 2450)
        production_schedule = st.text_input("Current Schedule", "2,000 units by Feb 15")
        suppliers = st.multiselect("Active Suppliers", 
                                   ["Supplier A (primary)", "Supplier B (backup)", "Supplier C (premium)"],
                                   ["Supplier A (primary)"])
        
        st.markdown("**Constraints**")
        
        lead_time = st.number_input("Lead Time (days)", 1, 60, 14)
        budget_limit = st.number_input("Budget Limit ($K)", 10, 1000, 250)
        
        analyze_btn = st.button("🏭 Run AI Analysis", type="primary", use_container_width=True)
    
    with col2:
        if analyze_btn:
            st.markdown("**Dryft AI Decision**")
            
            import time
            with st.spinner("AI analyzing production impact..."):
                time.sleep(1.0)
            
            st.success("✅ Optimal decision calculated and executed!")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">AI Recommendation & Execution</h3>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                    <p style="font-size: 16px; color: white; margin: 0; line-height: 1.8;">
                    <strong>Decision:</strong> Switch to Supplier B for 40% of order, increase safety stock by 15%
                    <br><br>
                    <strong>Actions Taken:</strong><br>
                    ✓ Contacted Supplier B (auto-email sent)<br>
                    ✓ Adjusted order quantity to 800 units<br>
                    ✓ Increased safety stock from 300 → 345 units<br>
                    ✓ Rescheduled production line 2 by 3 days<br>
                    ✓ Updated ERP system automatically<br>
                    ✓ Notified production manager
                    <br><br>
                    <strong>Impact:</strong> Delivery still on-time, risk mitigated, $12K savings vs expedited shipping
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Mathematical Optimization Results")
            
            optimization_results = pd.DataFrame({
                'Option': ['Expedite primary supplier', 'Switch to backup supplier', 'Delay production', 'AI Optimal (hybrid)'],
                'Cost': ['$42K', '$28K', '$0', '$18K'],
                'Risk': ['Medium', 'Low', 'High', 'Low'],
                'Timeline': ['On-time', '+2 days', '+14 days', 'On-time']
            })
            
            st.dataframe(optimization_results, hide_index=True, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Decision Time", "4.2 min", "vs 2 days")
            col2.metric("Cost Savings", "$24K", "vs expedite")
            col3.metric("On-Time", "✓", "Maintained")
            col4.metric("Confidence", "96.8%", "High")

with tab2:
    st.markdown("### Manufacturing Operations Intelligence")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Decisions/Month", "2,480", "Automated")
    col2.metric("Avg Decision Time", "3.8 min", "vs 2 days")
    col3.metric("Accuracy", "94.5%", "Validated")
    col4.metric("Annual Savings", "$1.2M+", "Per customer")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### AI Decision Types & Volume")
        
        decision_data = []
        for decision, data in AI_DECISIONS.items():
            decision_data.append({
                'Decision Type': decision,
                'Monthly Frequency': data['frequency'],
                'Accuracy': f"{data['accuracy']}%",
                'Annual Savings': data['savings']
            })
        
        st.dataframe(pd.DataFrame(decision_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### Decision Distribution")
        
        decisions_list = list(AI_DECISIONS.keys())
        frequencies = [int(AI_DECISIONS[d]['frequency'].split('/')[0]) for d in decisions_list]
        
        fig1 = px.pie(
            values=frequencies,
            names=decisions_list,
            color_discrete_sequence=['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#22c55e']
        )
        fig1.update_traces(textposition='inside', textinfo='percent')
        fig1.update_layout(height=300, showlegend=True)
        
        st.plotly_chart(fig1, use_container_width=True)

with tab3:
    st.markdown("### Customer Impact")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">7-Fig</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Annual Savings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">94.5%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Decision Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%); padding: 30px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 42px; font-weight: 800;">75%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 600;">Admin Time Saved</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Results Breakdown")
    
    results_data = []
    for category, data in CUSTOMER_RESULTS.items():
        results_data.append({
            'Category': category,
            'Reduction': f"{data['reduction']}%",
            'Annual Savings': f"${data['annual_savings']}M"
        })
    
    st.dataframe(pd.DataFrame(results_data), hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### Platform Technology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Core Capabilities**")
        
        for feature, description in PLATFORM_FEATURES.items():
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #6366f1;">
                <p style="margin: 0; font-weight: 700; color: #4338ca;">{feature}</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Founding Team**")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); padding: 20px; border-radius: 12px;">
            <h4 style="margin: 0 0 15px 0; color: #3730a3;">👩‍💻 Ex-Porsche + Stanford</h4>
            <p style="margin: 8px 0; color: #4338ca; font-size: 14px;">
            <strong>Anna-Julia Storch (CEO):</strong><br>
            • Automotive data science (supplier training)<br>
            • Stanford graduate<br>
            • MIT research (workplace wellbeing)
            </p>
            <p style="margin: 15px 0 0 0; color: #4338ca; font-size: 14px;">
            <strong>Leonie Freisinger (CTO):</strong><br>
            • Porsche Taycan safety systems engineer<br>
            • Stanford graduate<br>
            • Europe's largest student AI initiative leader
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Investors**")
        
        investors_data = {
            'Investor': ['General Catalyst', 'Neo', 'Sandberg Bernthal Ventures', 'Angels'],
            'Type': ['Lead', 'Participant', 'Participant', 'Jeff Wilke (ex-Amazon), Claire Hughes Johnson (ex-Stripe)']
        }
        
        st.dataframe(pd.DataFrame(investors_data), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #3730a3; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #6366f1; font-weight: 700; margin: 0 0 6px 0;">✓ $5M Funded</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">General Catalyst-led</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #6366f1; font-weight: 700; margin: 0 0 6px 0;">✓ AI + Math Optimization</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Context-aware agents</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #6366f1; font-weight: 700; margin: 0 0 6px 0;">✓ 7-Figure Savings</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Proven customer results</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #6366f1; font-weight: 700; margin: 0 0 6px 0;">✓ Ex-Porsche Founders</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Stanford + automotive</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Dryft</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)