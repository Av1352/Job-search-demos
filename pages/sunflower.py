"""
Sunflower - AI-Powered Sobriety Platform
AI sponsor "Sam" + tele-therapy for addiction recovery
Built for Sunflower by Anju Vilashni Nandhakumar
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
st.set_page_config(page_title="Sunflower - AI Sobriety", page_icon="🌻", layout="wide")

# Addiction types
ADDICTION_TYPES = {
    'Alcohol': {'users': 42000, 'avg_sober_days': 87, 'success_rate': 68},
    'Marijuana': {'users': 28000, 'avg_sober_days': 124, 'success_rate': 72},
    'Nicotine': {'users': 15000, 'avg_sober_days': 156, 'success_rate': 65},
    'Opioids': {'users': 8500, 'avg_sober_days': 63, 'success_rate': 58},
    'Cocaine': {'users': 4200, 'avg_sober_days': 71, 'success_rate': 61},
    'Other': {'users': 2300, 'avg_sober_days': 94, 'success_rate': 64}
}

# Platform features
PLATFORM_FEATURES = {
    'AI Sponsor (Sam)': 'Cartoon bee AI that provides 24/7 support and accountability',
    'Sobriety Tracking': 'Beautiful visual progression tracking',
    'Social Support': 'Twitter-style social media for people in recovery',
    'DIY Learning': 'Masterclass-style content for all addiction types',
    'CBT Journaling': 'Cognitive behavioral therapy exercises',
    'Tele-Therapy': 'Licensed therapists in CA, TX (expanding to 50 states)',
    'MAT Services': 'Medication-assisted treatment for substance use disorders'
}

# User engagement
ENGAGEMENT_METRICS = {
    'Daily Active Users': 45000,
    'Monthly Active Users': 100000,
    'Avg Session Time': 18.5,
    'Check-ins per Day': 3.2,
    'Community Posts': 8500,
    'Therapy Sessions': 2400
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🌻</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Sunflower</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">AI-Powered Sobriety Platform</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Meet Sam, your AI sponsor • 100K users • One trillion days sober</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌻 AI Sponsor Sam", "📊 Sobriety Journey", "👥 Community", "💡 Platform Features"])

with tab1:
    st.markdown("### Meet Sam - Your AI Sobriety Sponsor")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Start Your Recovery Journey**")
        
        user_name = st.text_input("Name", "Alex")
        
        st.markdown("**What brings you here today?**")
        
        addiction_type = st.selectbox("Primary Struggle", list(ADDICTION_TYPES.keys()))
        
        duration = st.selectbox("How long have you been struggling?",
                               ["Less than 1 year", "1-3 years", "3-5 years", "5-10 years", "10+ years"])
        
        sobriety_goal = st.selectbox("Your goal",
                                    ["Complete abstinence", "Harm reduction", "Moderation", "Exploring options"])
        
        st.markdown("**Current Status**")
        
        last_use = st.selectbox("Last use",
                               ["Today", "1-3 days ago", "1 week ago", "2 weeks ago", "1 month+", "Never used"])
        
        support_system = st.multiselect("Current support",
                                       ["Family", "Friends", "AA/NA", "Therapist", "None"],
                                       ["Friends"])
        
        crisis = st.checkbox("I'm in crisis and need immediate help")
        
        if crisis:
            st.error("🚨 **Crisis Resources Available 24/7:**\n\n988 Suicide & Crisis Lifeline\n\nSAMHSA National Helpline: 1-800-662-4357")
        
        connect_btn = st.button("🌻 Connect with Sam", type="primary", use_container_width=True)
    
    with col2:
        if connect_btn and not crisis:
            st.markdown("**Your AI Sponsor - Sam the Bee 🐝**")
            
            import time
            with st.spinner("Sam is personalizing your support plan..."):
                time.sleep(1.2)
            
            st.success("✅ Sam is ready to support you!")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">🐝 Sam's Welcome Message</h3>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                    <p style="font-size: 16px; color: white; margin: 0; line-height: 1.6;">
                    "Hey Alex! 🌻 I'm Sam, your AI sponsor. I'm here for you 24/7 - no judgment, just support. 
                    <br><br>
                    I see you're working on {addiction_type.lower()}. That takes real courage to acknowledge. 
                    <br><br>
                    Based on your profile, I've created a personalized recovery plan. We'll take this one day at a time, together. I'll check in with you throughout the day, celebrate your wins, and help you through the tough moments.
                    <br><br>
                    Remember: You're not alone in this. There are 100,000+ people in our community fighting the same fight. Let's do this! 💪"
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Your Personalized Support Plan")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #fbbf24;">
                    <h4 style="margin: 0 0 12px 0; color: #92400e;">Daily Check-ins with Sam</h4>
                    <ul style="margin: 0; padding-left: 20px; color: #78350f;">
                        <li>Morning motivation (8 AM)</li>
                        <li>Midday check-in (12 PM)</li>
                        <li>Evening reflection (8 PM)</li>
                        <li>On-demand support anytime</li>
                        <li>Trigger management coaching</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #f59e0b;">
                    <h4 style="margin: 0 0 12px 0; color: #78350f;">Your Resources</h4>
                    <ul style="margin: 0; padding-left: 20px; color: #92400e;">
                        <li>CBT journaling exercises</li>
                        <li>Masterclass-style learning content</li>
                        <li>Community support groups</li>
                        <li>Crisis hotline access</li>
                        <li>Optional tele-therapy sessions</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### Sample Conversation with Sam")
            
            st.markdown("""
            <div style="background: #fef3c7; padding: 20px; border-radius: 12px;">
                <p style="margin: 0 0 10px 0;"><strong>You:</strong> I'm really craving right now. This is hard.</p>
                <p style="margin: 10px 0; background: white; padding: 12px; border-radius: 8px;"><strong>🐝 Sam:</strong> I hear you, Alex. Cravings are tough, but they're temporary. Let's work through this together. What's triggering this craving right now? Is it a person, place, or emotion?</p>
                <p style="margin: 10px 0 0 0;"><strong>You:</strong> Stressed about work. Used to drink to relax.</p>
                <p style="margin: 10px 0; background: white; padding: 12px; border-radius: 8px;"><strong>🐝 Sam:</strong> Work stress is a common trigger. Here's what I want you to try: Take 5 deep breaths with me right now. Then, go for a 10-minute walk. When you get back, we'll talk about healthier coping strategies. You've got 12 days sober - don't throw that away for a temporary feeling. You're stronger than you think! 💪</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Track Your Sobriety Journey")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">100K</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Monthly Active Users</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">87</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Avg Days Sober</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ea580c 0%, #dc2626 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">68%</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #22c55e 0%, #10b981 100%); padding: 25px; border-radius: 12px; text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">$1M</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-weight: 600;">Annualized Revenue</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Sobriety Progress Tracker")
        
        days_sober = st.number_input("Days Sober", 0, 1000, 12, help="Track your progress")
        
        # Visual progress
        milestones = [1, 7, 30, 90, 180, 365]
        achieved = [m for m in milestones if m <= days_sober]
        
        st.markdown("**Milestones Achieved:**")
        
        for milestone in milestones:
            status = "✅" if milestone in achieved else "⭕"
            color = "#22c55e" if milestone in achieved else "#d1d5db"
            st.markdown(f"""
            <div style="background: white; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid {color};">
                <span style="font-size: 18px;">{status}</span>
                <strong style="margin-left: 10px;">{milestone} Day{'s' if milestone > 1 else ''} Sober</strong>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Daily Check-in")
        
        mood = st.select_slider("How are you feeling today?",
                               options=["😰 Struggling", "😔 Difficult", "😐 Okay", "🙂 Good", "😊 Great"],
                               value="🙂 Good")
        
        cravings = st.slider("Craving intensity (0-10)", 0, 10, 3)
        
        triggers = st.multiselect("Today's triggers",
                                 ["Stress", "Social pressure", "Boredom", "Loneliness", "Celebration", "None today"],
                                 ["Stress"])
        
        if st.button("📝 Submit Check-in", use_container_width=True):
            st.success("✅ Check-in recorded! Sam will follow up with personalized support.")
    
    with col2:
        st.markdown("### Your Recovery Journey")
        
        # Generate sample progress data
        days = list(range(0, days_sober + 1))
        craving_intensity = [8 - (d * 0.05) + np.random.randn() * 1.5 for d in days]
        craving_intensity = [max(0, min(10, c)) for c in craving_intensity]
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=days,
            y=craving_intensity,
            mode='lines',
            line=dict(color='#fbbf24', width=3),
            fill='tozeroy',
            fillcolor='rgba(251, 191, 36, 0.1)'
        ))
        
        fig1.update_layout(
            title="Craving Intensity Over Time",
            xaxis_title="Days Sober",
            yaxis_title="Craving Intensity (0-10)",
            height=250
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("### Community Support")
        
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #fbbf24;">
            <h4 style="margin: 0 0 15px 0; color: #92400e;">Recent Posts</h4>
            <div style="margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #e5e7eb;">
                <p style="margin: 0; font-weight: 600; color: #78350f;">@sarah_90days</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">Just hit 90 days! 🎉 Sam helped me through so many tough moments. This community is everything. Keep going everyone! 💪</p>
                <p style="margin: 8px 0 0 0; color: #999; font-size: 12px;">❤️ 247 • 💬 18 • 2 hours ago</p>
            </div>
            <div style="margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #e5e7eb;">
                <p style="margin: 0; font-weight: 600; color: #78350f;">@mike_journey</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">Day 5 and the cravings are intense. Sam reminded me why I started. Reading your stories helps. We got this! 🌻</p>
                <p style="margin: 8px 0 0 0; color: #999; font-size: 12px;">❤️ 156 • 💬 34 • 4 hours ago</p>
            </div>
            <div>
                <p style="margin: 0; font-weight: 600; color: #78350f;">@recovery_warrior</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">1 year sober today! 365 days of choosing life. Sam's daily check-ins kept me accountable when I had no one else. Forever grateful 🙏</p>
                <p style="margin: 8px 0 0 0; color: #999; font-size: 12px;">❤️ 892 • 💬 67 • 6 hours ago</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Community & Social Support")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Community Members", "100,000", "+15K this month")
    col2.metric("Daily Posts", "8,500", "Active")
    col3.metric("Support Groups", "342", "Live now")
    col4.metric("Avg Engagement", "18.5 min", "Per session")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Users by Addiction Type")
        
        types_data = []
        for addiction, data in ADDICTION_TYPES.items():
            types_data.append({
                'Type': addiction,
                'Users': f"{data['users']:,}",
                'Avg Days Sober': data['avg_sober_days'],
                'Success Rate': f"{data['success_rate']}%"
            })
        
        st.dataframe(pd.DataFrame(types_data), hide_index=True, use_container_width=True)
        
        st.markdown("### User Distribution")
        
        types_list = list(ADDICTION_TYPES.keys())
        user_counts = [ADDICTION_TYPES[t]['users'] for t in types_list]
        
        fig2 = px.pie(
            values=user_counts,
            names=types_list,
            color_discrete_sequence=['#fbbf24', '#f59e0b', '#ea580c', '#dc2626', '#b91c1c', '#991b1b']
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=300, showlegend=False)
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("### Growth & Traction")
        
        months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        mau = [200, 8500, 22000, 41000, 62000, 78000, 92000, 100000]
        
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=months,
            y=mau,
            mode='lines+markers',
            line=dict(color='#fbbf24', width=4),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor='rgba(251, 191, 36, 0.1)'
        ))
        
        fig3.update_layout(
            title="Monthly Active Users (6 months)",
            yaxis_title="Users",
            height=300
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("### Impact Metrics")
        
        impact_data = {
            'Metric': ['Total days sober', 'Lives saved (est)', 'Relapses prevented', 'Therapy sessions'],
            'Value': ['8.7M days', '1,200+', '24,500+', '2,400/month']
        }
        
        st.dataframe(pd.DataFrame(impact_data), hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### Comprehensive Support Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Core Features**")
        
        for feature, description in PLATFORM_FEATURES.items():
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #fbbf24;">
                <p style="margin: 0; font-weight: 700; color: #92400e;">{feature}</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**Technology**")
        st.markdown("""
        - 🤖 Advanced conversational AI (GPT-based)
        - 📱 Native iOS/Android apps
        - 🌐 Web platform
        - 🔔 Smart push notifications
        - 📊 Progress tracking & analytics
        - 🎓 Evidence-based CBT content
        - 🔒 HIPAA-compliant infrastructure
        - 👥 Peer-to-peer social features
        """)
    
    with col2:
        st.markdown("**Clinical Services**")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 20px; border-radius: 12px;">
            <h4 style="margin: 0 0 15px 0; color: #92400e;">Tele-Therapy Clinic</h4>
            <div style="margin-bottom: 10px;">
                <p style="margin: 0; font-weight: 600; color: #78350f;">Talk Therapy</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #92400e;">Licensed therapists in CA, TX (expanding to 50 states)</p>
            </div>
            <div style="margin-bottom: 10px;">
                <p style="margin: 0; font-weight: 600; color: #78350f;">MAT (Medication-Assisted Treatment)</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #92400e;">Buprenorphine, naltrexone for opioid/alcohol use disorders</p>
            </div>
            <div>
                <p style="margin: 0; font-weight: 600; color: #78350f;">Medical Supervision</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #92400e;">Doctors trained in addiction medicine</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Pricing**")
        
        pricing_data = {
            'Tier': ['Free', 'Premium', 'Therapy'],
            'Price': ['$0/month', '$8.95/month', '$120-200/session'],
            'Includes': [
                'Basic Sam AI, community access',
                'Unlimited Sam, all content, priority support',
                'Licensed therapists, MAT prescriptions'
            ]
        }
        
        st.dataframe(pd.DataFrame(pricing_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Addressable Market**")
        
        market_data = {
            'Segment': ['Alcohol use disorder', 'Drug addiction', 'Total global market', 'US market'],
            'Size': ['107M people (US)', '22M people (US)', '464M people', '129M people']
        }
        
        st.dataframe(pd.DataFrame(market_data), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #92400e; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #fbbf24; font-weight: 700; margin: 0 0 6px 0;">✓ 100K Monthly Users</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">6-month growth</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #fbbf24; font-weight: 700; margin: 0 0 6px 0;">✓ 24/7 AI Sponsor</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Sam the bee</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #fbbf24; font-weight: 700; margin: 0 0 6px 0;">✓ 68% Success Rate</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Sustained sobriety</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #fbbf24; font-weight: 700; margin: 0 0 6px 0;">✓ 8.7M Days Sober</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Community total</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Sunflower</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)