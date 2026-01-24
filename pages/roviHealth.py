"""
Rovi Health - AI Healthcare Concierge
Employee wellness navigation and care coordination
Built for Rovi Health by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils.sidebar import render_sidebar
render_sidebar()

st.set_page_config(page_title="Rovi Health - AI Healthcare Concierge", layout="wide")

# Initialize session state
if 'symptoms_analyzed' not in st.session_state:
    st.session_state.symptoms_analyzed = False
if 'care_plan_generated' not in st.session_state:
    st.session_state.care_plan_generated = False

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #059669 0%, #73BA9B 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(5, 150, 105, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #14b8a6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🏥</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Rovi Concierge
        </h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">AI Healthcare Navigation for Employees</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Get the right care, at the right place, at the right time</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">24/7 Access</span>
            <span style="background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(20, 184, 166, 0.4);">Care Navigation</span>
            <span style="background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(6, 182, 212, 0.4);">Benefits Guide</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">YC Backed</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Rovi Health</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Value Proposition
st.markdown("""
<div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); padding: 25px; border-radius: 15px; border: 2px solid #059669; margin-bottom: 30px;">
    <h3 style="color: #065f46; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">🎯 The Problem We Solve</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #ef4444; font-weight: 700; margin: 0 0 8px 0;">❌ Today</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Employees spend 2+ hours navigating healthcare, often going to wrong place (ER vs urgent care)</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #f59e0b; font-weight: 700; margin: 0 0 8px 0;">💰 Cost Impact</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Unnecessary ER visits cost employers $1,800 vs $150 urgent care. 40% of ER visits are non-emergency.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #10b981; font-weight: 700; margin: 0 0 8px 0;">✅ With Rovi</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Get right care in 2 minutes. Save $500K+ per 1,000 employees annually. Reduce ER visits by 35%.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🩺 Symptom Checker", "🏥 Find Care", "💳 My Benefits", "📊 Impact Dashboard"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">How are you feeling?</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Tell me what's bothering you, and I'll help figure out next steps</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Symptom input
        symptoms = st.text_area(
            "Describe your symptoms",
            placeholder="Example: I have a sore throat, slight fever (100.5°F), and feel tired. Started yesterday.",
            height=100
        )
        
        # Quick questions
        st.markdown("**Quick questions to help me assess:**")
        col_a, col_b = st.columns(2)
        with col_a:
            duration = st.selectbox("How long have you had symptoms?", 
                ["Less than 24 hours", "1-3 days", "4-7 days", "More than a week"])
            pain_level = st.slider("Pain level (if any)", 0, 10, 3)
        with col_b:
            fever = st.selectbox("Do you have a fever?", 
                ["No fever", "Low (99-100°F)", "Moderate (100-102°F)", "High (102°F+)"])
            breathing = st.selectbox("Any breathing difficulty?", 
                ["No difficulty", "Mild discomfort", "Moderate difficulty", "Severe difficulty"])
        
        if st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True):
            st.session_state.symptoms_analyzed = True
            st.session_state.care_plan_generated = True
    
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #92400e; margin: 0 0 10px 0; font-size: 16px;">🚨 Call 911 if you have:</h4>
            <ul style="color: #78350f; font-size: 13px; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li>Chest pain or pressure</li>
                <li>Severe difficulty breathing</li>
                <li>Sudden confusion</li>
                <li>Severe bleeding</li>
                <li>Sudden severe headache</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.symptoms_analyzed:
        # Analysis results
        st.markdown("<hr style='margin: 30px 0; border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)
        
        # Determine urgency based on inputs
        if breathing in ["Severe difficulty"] or pain_level >= 8:
            urgency = "high"
            urgency_color = "#ef4444"
            urgency_icon = "🚨"
            urgency_text = "High Urgency"
            recommendation = "Emergency Room"
            recommendation_icon = "🏥"
        elif fever in ["High (102°F+)"] or pain_level >= 6:
            urgency = "moderate"
            urgency_color = "#f59e0b"
            urgency_icon = "⚠️"
            urgency_text = "Moderate Urgency"
            recommendation = "Urgent Care"
            recommendation_icon = "⚡"
        else:
            urgency = "low"
            urgency_color = "#10b981"
            urgency_icon = "✅"
            urgency_text = "Low Urgency"
            recommendation = "Primary Care Doctor"
            recommendation_icon = "👨‍⚕️"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">📋 Your Care Plan</h2>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Urgency Level</p>
                    <p style="font-size: 36px; margin: 0;">{urgency_icon}</p>
                    <p style="font-size: 18px; color: white; font-weight: 700; margin: 8px 0 0 0;">{urgency_text}</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Recommended Care</p>
                    <p style="font-size: 36px; margin: 0;">{recommendation_icon}</p>
                    <p style="font-size: 18px; color: white; font-weight: 700; margin: 8px 0 0 0;">{recommendation}</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Est. Out-of-Pocket</p>
                    <p style="font-size: 36px; color: white; font-weight: 900; margin: 0;">$30</p>
                    <p style="font-size: 13px; color: rgba(255,255,255,0.8); margin: 8px 0 0 0;">With your plan</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed recommendations
        col_x, col_y = st.columns(2)
        
        with col_x:
            st.markdown("""
            <div style="background: #ecfdf5; padding: 25px; border-radius: 15px; border: 2px solid #059669;">
                <h3 style="color: #065f46; margin: 0 0 15px 0; font-size: 20px;">💡 What You Should Do</h3>
                <ol style="color: #047857; font-size: 15px; line-height: 1.8; margin: 0; padding-left: 20px;">
                    <li><strong>Rest and stay hydrated</strong> - Drink plenty of fluids</li>
                    <li><strong>Monitor your temperature</strong> - Take it every 4 hours</li>
                    <li><strong>Over-the-counter relief</strong> - Acetaminophen for fever/pain</li>
                    <li><strong>See doctor if worsening</strong> - Schedule visit if no improvement in 48 hours</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col_y:
            st.markdown("""
            <div style="background: #fef3c7; padding: 25px; border-radius: 15px; border: 2px solid #f59e0b;">
                <h3 style="color: #92400e; margin: 0 0 15px 0; font-size: 20px;">🏥 Nearby Options</h3>
                <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 12px;">
                    <p style="font-weight: 700; color: #1f2937; margin: 0 0 5px 0;">Boston Medical Center - Urgent Care</p>
                    <p style="color: #6b7280; font-size: 13px; margin: 0;">📍 0.8 miles · Open now · $30 copay</p>
                    <p style="color: #059669; font-size: 13px; margin: 5px 0 0 0; font-weight: 600;">⏱️ 15 min wait time</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 10px;">
                    <p style="font-weight: 700; color: #1f2937; margin: 0 0 5px 0;">Mass General - Primary Care</p>
                    <p style="color: #6b7280; font-size: 13px; margin: 0;">📍 1.2 miles · Next available: Tomorrow 2pm</p>
                    <p style="color: #059669; font-size: 13px; margin: 5px 0 0 0; font-weight: 600;">💳 $20 copay</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Find the right care, fast</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Search by specialty, location, or what you need</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        care_type = st.selectbox("Type of care", 
            ["Primary Care", "Urgent Care", "Specialist", "Mental Health", "Physical Therapy"])
    with col2:
        location = st.text_input("Location", value="Boston, MA")
    with col3:
        insurance = st.selectbox("Insurance", ["Blue Cross Blue Shield", "Aetna", "UnitedHealth"])
    
    if st.button("🔍 Search Providers", type="primary"):
        st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
        
        # Mock provider results
        providers = [
            {
                "name": "Dr. Sarah Chen, MD",
                "specialty": "Primary Care",
                "rating": 4.8,
                "distance": "0.5 miles",
                "next_available": "Today 3:00 PM",
                "copay": "$20",
                "in_network": True
            },
            {
                "name": "Boston Medical Group",
                "specialty": "Primary Care",
                "rating": 4.6,
                "distance": "0.8 miles",
                "next_available": "Tomorrow 9:00 AM",
                "copay": "$20",
                "in_network": True
            },
            {
                "name": "Mass General - Downtown",
                "specialty": "Primary Care",
                "rating": 4.9,
                "distance": "1.2 miles",
                "next_available": "Tomorrow 2:00 PM",
                "copay": "$20",
                "in_network": True
            }
        ]
        
        for provider in providers:
            st.markdown(f"""
            <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <h4 style="color: #1f2937; font-size: 18px; font-weight: 700; margin: 0 0 8px 0;">{provider['name']}</h4>
                        <p style="color: #6b7280; font-size: 14px; margin: 0 0 8px 0;">{provider['specialty']}</p>
                        <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                            <span style="color: #059669; font-size: 13px;">⭐ {provider['rating']} rating</span>
                            <span style="color: #6b7280; font-size: 13px;">📍 {provider['distance']}</span>
                            <span style="color: #3b82f6; font-size: 13px;">⏰ {provider['next_available']}</span>
                            <span style="color: #10b981; font-size: 13px; font-weight: 600;">✅ In-network</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <p style="font-size: 24px; color: #059669; font-weight: 700; margin: 0;">{provider['copay']}</p>
                        <p style="font-size: 12px; color: #6b7280; margin: 5px 0 0 0;">copay</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Your Benefits Explained</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Simple breakdown of what you pay and what's covered</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Benefits summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background: #ecfdf5; padding: 20px; border-radius: 12px; text-align: center;">
            <p style="font-size: 14px; color: #047857; margin: 0;">Annual Deductible</p>
            <p style="font-size: 32px; color: #059669; font-weight: 900; margin: 8px 0;">$1,500</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">$800 met this year</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #eff6ff; padding: 20px; border-radius: 12px; text-align: center;">
            <p style="font-size: 14px; color: #1e40af; margin: 0;">Out-of-Pocket Max</p>
            <p style="font-size: 32px; color: #3b82f6; font-weight: 900; margin: 8px 0;">$5,000</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">$1,200 used</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; text-align: center;">
            <p style="font-size: 14px; color: #92400e; margin: 0;">Remaining Savings</p>
            <p style="font-size: 32px; color: #f59e0b; font-weight: 900; margin: 8px 0;">$3,800</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Until max reached</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Cost breakdown
    st.markdown("""
    <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
        <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px; font-weight: 700;">💰 What You'll Pay</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 2px solid #e5e7eb;">
                <th style="text-align: left; padding: 15px; color: #6b7280; font-size: 13px; font-weight: 600;">Service</th>
                <th style="text-align: right; padding: 15px; color: #6b7280; font-size: 13px; font-weight: 600;">Your Cost</th>
                <th style="text-align: right; padding: 15px; color: #6b7280; font-size: 13px; font-weight: 600;">Typical Cost</th>
            </tr>
            <tr style="border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 15px; color: #1f2937; font-weight: 600;">Primary Care Visit</td>
                <td style="padding: 15px; text-align: right; color: #059669; font-weight: 700; font-size: 18px;">$20</td>
                <td style="padding: 15px; text-align: right; color: #6b7280;">$150-200</td>
            </tr>
            <tr style="border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 15px; color: #1f2937; font-weight: 600;">Urgent Care</td>
                <td style="padding: 15px; text-align: right; color: #059669; font-weight: 700; font-size: 18px;">$30</td>
                <td style="padding: 15px; text-align: right; color: #6b7280;">$100-150</td>
            </tr>
            <tr style="border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 15px; color: #1f2937; font-weight: 600;">Emergency Room</td>
                <td style="padding: 15px; text-align: right; color: #f59e0b; font-weight: 700; font-size: 18px;">$250</td>
                <td style="padding: 15px; text-align: right; color: #6b7280;">$1,500-3,000</td>
            </tr>
            <tr style="border-bottom: 1px solid #f3f4f6;">
                <td style="padding: 15px; color: #1f2937; font-weight: 600;">Specialist Visit</td>
                <td style="padding: 15px; text-align: right; color: #059669; font-weight: 700; font-size: 18px;">$40</td>
                <td style="padding: 15px; text-align: right; color: #6b7280;">$200-300</td>
            </tr>
            <tr>
                <td style="padding: 15px; color: #1f2937; font-weight: 600;">Mental Health Visit</td>
                <td style="padding: 15px; text-align: right; color: #059669; font-weight: 700; font-size: 18px;">$20</td>
                <td style="padding: 15px; text-align: right; color: #6b7280;">$150-250</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Company Impact Dashboard</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Real-time metrics showing Rovi's impact on your workforce</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 32px; border-radius: 20px; margin-bottom: 25px;">
        <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">📊 Annual Impact (1,000 Employees)</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Cost Savings</p>
                <p style="font-size: 36px; color: white; font-weight: 900; margin: 8px 0;">$580K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">vs traditional care</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">ER Reduction</p>
                <p style="font-size: 36px; color: #86efac; font-weight: 900; margin: 8px 0;">35%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">fewer unnecessary visits</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Time Saved</p>
                <p style="font-size: 36px; color: white; font-weight: 900; margin: 8px 0;">5,200</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">employee hours/year</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Satisfaction</p>
                <p style="font-size: 36px; color: #fbbf24; font-weight: 900; margin: 8px 0;">92%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">employee NPS</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #ecfdf5; padding: 25px; border-radius: 15px; border: 2px solid #059669;">
            <h3 style="color: #065f46; margin: 0 0 20px 0; font-size: 20px;">💰 Where Savings Come From</h3>
            <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #1f2937; font-weight: 600;">Reduced ER visits</span>
                    <span style="color: #059669; font-weight: 700;">$320K</span>
                </div>
                <div style="background: #e5e7eb; height: 8px; border-radius: 4px;">
                    <div style="background: #059669; width: 55%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #1f2937; font-weight: 600;">Better care navigation</span>
                    <span style="color: #059669; font-weight: 700;">$180K</span>
                </div>
                <div style="background: #e5e7eb; height: 8px; border-radius: 4px;">
                    <div style="background: #10b981; width: 31%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 10px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #1f2937; font-weight: 600;">Preventive care adoption</span>
                    <span style="color: #059669; font-weight: 700;">$80K</span>
                </div>
                <div style="background: #e5e7eb; height: 8px; border-radius: 4px;">
                    <div style="background: #14b8a6; width: 14%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #eff6ff; padding: 25px; border-radius: 15px; border: 2px solid #3b82f6;">
            <h3 style="color: #1e40af; margin: 0 0 20px 0; font-size: 20px;">📈 Employee Engagement</h3>
            <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 12px;">
                <p style="color: #6b7280; font-size: 13px; margin: 0 0 8px 0;">Monthly Active Users</p>
                <p style="color: #3b82f6; font-size: 28px; font-weight: 700; margin: 0;">68%</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 12px;">
                <p style="color: #6b7280; font-size: 13px; margin: 0 0 8px 0;">Avg. Response Time</p>
                <p style="color: #3b82f6; font-size: 28px; font-weight: 700; margin: 0;">2 min</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 10px;">
                <p style="color: #6b7280; font-size: 13px; margin: 0 0 8px 0;">Issues Resolved Without Escalation</p>
                <p style="color: #3b82f6; font-size: 28px; font-weight: 700; margin: 0;">82%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #059669; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Rovi Health</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Massive ROI</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    $580K annual savings per 1,000 employees. 35% reduction in unnecessary ER visits. Pays for itself in 3 months.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ Instant Value</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Employees get help in 2 minutes, 24/7. Right care, right place, right time. No more guessing or wasted hours.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📈 Measurable Impact</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Real-time dashboard shows cost savings, utilization patterns, employee satisfaction. Data-driven decisions.
                </p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Enterprise Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">$580K savings</strong> per 1,000 employees annually</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">35% ER reduction</strong> through proper care navigation</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">5,200 hours saved</strong> employee time per year</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">92% satisfaction</strong> employee NPS score</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Clinical NLP</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Symptom analysis, triage logic, care recommendations</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Provider Network</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Real-time availability, in-network search, cost estimation</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Benefits Engine</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Plan integration, cost transparency, deductible tracking</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Analytics Dashboard</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Real-time impact metrics, cost savings, utilization</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #059669 0%, #73BA9B 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(5, 150, 105, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Rovi Health</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Clinical NLP • Healthcare Navigation • Benefits Integration
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered healthcare navigation and employee wellness coordination.<br>
            Symptom analysis • Care recommendations • Provider search • Benefits transparency • Impact analytics
        </p>
    </div>
    """, unsafe_allow_html=True)