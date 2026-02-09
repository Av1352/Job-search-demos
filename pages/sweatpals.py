"""
Sweatpals - Social Fitness Discovery Platform
Connect through movement with AI-powered event matching
Built for Sweatpals by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Sweatpals - Social Fitness", page_icon="🏃", layout="wide")

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🏃</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Sweatpals</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Social Fitness Discovery</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">1M+ users • AI matching • Daylife movement</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🏃 Discover Events", "📊 Platform Analytics", "💡 AI Matching"])

with tab1:
    st.markdown("### Find Your Fitness Community")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Your Preferences**")
        
        location = st.text_input("Location", "Austin, TX")
        
        activities = st.multiselect("Interested in",
                                    ["Run/Walk", "Yoga", "Strength", "Cycling", "Hiking", "Pickleball", "Dance", "Mindfulness", "Swimming"],
                                    ["Run/Walk", "Yoga"])
        
        day_time = st.selectbox("Preferred Time", ["Morning (6-10am)", "Midday (10am-2pm)", "Evening (5-8pm)", "Anytime"])
        
        budget = st.slider("Budget per session", 0, 50, 15)
        
        vibe = st.multiselect("Vibe",
                                ["Social/Fun", "Competitive", "Beginner-friendly", "LGBTQ+", "Women only", "Dog-friendly"],
                                ["Social/Fun", "Beginner-friendly"])
        
        discover_btn = st.button("🏃 Discover Events", type="primary", use_container_width=True)
    
    with col2:
        if discover_btn:
            st.markdown("**AI-Matched Events Near You**")
            
            import time
            with st.spinner("Finding your perfect fitness community..."):
                time.sleep(1.0)
            
            st.success(f"✅ Found 47 events matching your preferences in {location}!")
            
            # Sample events
            events = [
                {
                    'name': '☀️ Morning Run Club + Coffee',
                    'host': '@austinrunnersco',
                    'time': 'Wed 7:00 AM',
                    'location': 'Lady Bird Lake Trail',
                    'price': 'Free',
                    'attendees': 34,
                    'match': 98
                },
                {
                    'name': '🧘 Rooftop Yoga & Social',
                    'host': '@zenflowATX',
                    'time': 'Sat 9:30 AM',
                    'location': 'Downtown Rooftop',
                    'price': '$12',
                    'attendees': 18,
                    'match': 95
                },
                {
                    'name': '🏃 5K Social Run + Brunch',
                    'host': '@runandbrunch',
                    'time': 'Sun 8:00 AM',
                    'location': 'Zilker Park',
                    'price': '$8',
                    'attendees': 42,
                    'match': 92
                },
                {
                    'name': '💪 Beginner HIIT Class',
                    'host': '@fitfam_austin',
                    'time': 'Thu 6:30 PM',
                    'location': 'Auditorium Shores',
                    'price': '$15',
                    'attendees': 22,
                    'match': 88
                }
            ]
            
            for event in events:
                st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 12px; margin-bottom: 15px; border-left: 5px solid #f97316;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 1;">
                            <h4 style="margin: 0 0 8px 0; color: #1f2937;">{event['name']}</h4>
                            <p style="margin: 4px 0; color: #666; font-size: 14px;">👤 {event['host']} • 📍 {event['location']}</p>
                            <p style="margin: 4px 0; color: #666; font-size: 14px;">🕐 {event['time']} • 👥 {event['attendees']} going</p>
                        </div>
                        <div style="text-align: right;">
                            <span style="background: #22c55e; color: white; padding: 4px 10px; border-radius: 15px; font-size: 12px; font-weight: 600;">{event['match']}% match</span>
                            <p style="margin: 8px 0 0 0; font-weight: 700; color: #f97316; font-size: 16px;">{event['price']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Platform Growth & Engagement")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Monthly Users", "1M+", "+170K WAU")
    col2.metric("Weekly Active", "170K", "16% engagement")
    col3.metric("Events Created", "20K+", "Monthly")
    col4.metric("Hosts", "4,500", "+30% revenue")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### User Growth (18 months)")
        
        months = ['Jun 24', 'Aug 24', 'Oct 24', 'Dec 24', 'Feb 25', 'Apr 25', 'Jun 25', 'Aug 25', 'Oct 25', 'Dec 25']
        users = [5000, 28000, 85000, 180000, 320000, 485000, 650000, 820000, 925000, 1000000]
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=months,
            y=users,
            mode='lines+markers',
            line=dict(color='#f97316', width=3),
            fill='tozeroy',
            fillcolor='rgba(249, 115, 22, 0.1)'
        ))
        
        fig1.update_layout(
            yaxis_title='Monthly Active Users',
            height=300
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### Host Economics")
        
        host_data = {
            'Metric': ['Avg host revenue', 'Customer increase', 'Platform fee', 'Avg payout time'],
            'Value': ['$70K/year', '+30%', '10-15%', '48 hours']
        }
        
        st.dataframe(pd.DataFrame(host_data), hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; margin-top: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #92400e;">🌎 Expansion</h4>
            <p style="margin: 8px 0; color: #78350f; font-size: 14px;">
            <strong>Currently:</strong> 24 cities across US<br>
            <strong>Q1 2026:</strong> Expanding to 36 cities<br>
            <strong>Focus:</strong> Austin, Denver, Chicago, Tampa, SF, LA
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### AI-Powered Discovery & Matching")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Personalization Engine**")
        st.markdown("""
        - ✅ Activity preference matching
        - ✅ Schedule optimization
        - ✅ Location-based recommendations
        - ✅ Social vibe matching
        - ✅ Skill level pairing
        - ✅ Budget filtering
        - ✅ Friend network integration
        - ✅ Behavior-based suggestions
        """)
    
    with col2:
        st.markdown("**Platform Features**")
        st.markdown("""
        - 🎫 Digital ticketing & payments
        - 📝 Waiver signing
        - 👥 Participant management
        - 💬 Group chat per event
        - 📸 Event photo sharing
        - ⭐ Reviews & ratings
        - 🔔 Smart notifications
        - 💰 Creator monetization tools
        """)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #9a3412; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #f97316; font-weight: 700; margin: 0 0 6px 0;">✓ 1M+ Monthly Users</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">18-month growth</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #f97316; font-weight: 700; margin: 0 0 6px 0;">✓ AI Event Matching</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Personalized discovery</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #f97316; font-weight: 700; margin: 0 0 6px 0;">✓ +30% Host Revenue</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">$70K avg annual</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #f97316; font-weight: 700; margin: 0 0 6px 0;">✓ $12M Funded</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">a16z Speedrun, Pear VC</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Sweatpals</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)