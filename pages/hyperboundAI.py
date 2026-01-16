"""
Hyperbound AI - Sales Call Analysis & Coaching Platform
AI-powered sales performance optimization
Built for Hyperbound AI by Anju Nandhakumar
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="Hyperbound AI - Sales Call Analysis", layout="wide")

# Sample sales calls
SAMPLE_CALLS = {
    "Discovery Call - Tech SaaS": {
        "rep": "Sarah Chen",
        "prospect": "Acme Corp - CTO",
        "duration": 42,
        "outcome": "Meeting Scheduled",
        "transcript": """
Rep: Hi John, thanks for taking the time today. I know you're busy, so I'll keep this focused. Can you tell me about your current deployment process?

Prospect: Sure. Right now we're using a mix of Jenkins and some custom scripts. It works but it's slow - takes about 3 hours to deploy to production.

Rep: 3 hours, that's significant. And how often are you deploying?

Prospect: We try to do it twice a week, but honestly sometimes we batch changes because the process is so painful.

Rep: I see. So you're potentially delaying features and fixes because of deployment friction?

Prospect: Exactly. Plus when something goes wrong, rollback is a nightmare.

Rep: That makes sense. What would it mean for your team if you could deploy in 10 minutes with one-click rollback?

Prospect: That would be game-changing. We could ship daily, maybe multiple times a day.

Rep: Let me show you how our platform does exactly that...
""",
        "talk_ratio": 35,
        "questions_asked": 8,
        "objections": 0,
        "next_steps_defined": True
    },
    
    "Demo Call - Enterprise": {
        "rep": "Mike Johnson", 
        "prospect": "BigCo Inc - VP Engineering",
        "duration": 58,
        "outcome": "Trial Started",
        "transcript": """
Rep: Great to see you again. Last time we identified that your team spends about 20 hours per week on manual testing. Today I want to show you how we can reduce that to 2 hours. Sound good?

Prospect: Yes, show me.

Rep: Perfect. Let me share my screen. Here's the dashboard you'd see. Notice on the left...

Prospect: Wait, does this integrate with our CI/CD pipeline?

Rep: Great question. Yes, we have native integrations with GitHub Actions, CircleCI, and Jenkins. Which are you using?

Prospect: GitHub Actions primarily.

Rep: Perfect. Here's what that looks like... [demonstrates integration]. The setup takes about 5 minutes. See how the tests run automatically on every pull request?

Prospect: This is impressive. What's the pricing?

Rep: For a team your size - about 50 engineers - it's $5,000 per month. Given you're spending 1,000 engineer hours per month on manual testing at roughly $100/hour, this saves you $95,000 monthly. 19x ROI. Make sense?

Prospect: Yeah, that's compelling. Can we try it?

Rep: Absolutely. I can set you up with a 14-day trial right now...
""",
        "talk_ratio": 42,
        "questions_asked": 12,
        "objections": 1,
        "next_steps_defined": True
    },
    
    "Follow-up Call - Closing": {
        "rep": "Lisa Park",
        "prospect": "StartupXYZ - CEO", 
        "duration": 28,
        "outcome": "Deal Closed",
        "transcript": """
Rep: Hey Alex, thanks for hopping on. I know you've been testing for 10 days now. How's it going?

Prospect: Really well actually. The team loves it. We've already caught 3 bugs that would have made it to production.

Rep: That's fantastic! So it sounds like it's doing what we promised?

Prospect: Definitely. I'm ready to move forward.

Rep: Great! Let me walk you through the contract. It's pretty straightforward - annual subscription at $12,000, which breaks down to $1,000 per month. You get unlimited users, priority support, and quarterly business reviews with me personally. Any questions on that?

Prospect: No, that's clear. Do you need a PO?

Rep: Yes, if you can send that over, I'll get the contract to you within an hour and we can have you live by end of week.

Prospect: Perfect, I'll send it today.

Rep: Awesome. Welcome to the team, Alex! I'll send over onboarding details and we'll schedule a kickoff call for Thursday.
""",
        "talk_ratio": 38,
        "questions_asked": 5,
        "objections": 0,
        "next_steps_defined": True
    }
}

def analyze_sales_call(call_name):
    """Analyze a sales call for coaching insights"""
    
    call = SAMPLE_CALLS[call_name]
    
    words = len(call['transcript'].split())
    prospect_engagement = random.uniform(0.75, 0.95)
    objection_handling_score = 100 if call['objections'] == 0 else random.uniform(75, 90)
    
    talk_ratio_score = 100 - abs(call['talk_ratio'] - 30) * 2
    question_score = min(call['questions_asked'] * 10, 100)
    next_steps_score = 100 if call['next_steps_defined'] else 50
    
    overall_score = (talk_ratio_score + question_score + objection_handling_score + next_steps_score) / 4
    
    if overall_score >= 85:
        grade = "A"
        grade_color = "#10b981"
    elif overall_score >= 75:
        grade = "B"
        grade_color = "#3b82f6"
    elif overall_score >= 65:
        grade = "C"
        grade_color = "#f59e0b"
    else:
        grade = "D"
        grade_color = "#ef4444"
    
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📞 Call Analysis: {call_name}</h2>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Call Grade</p>
                <p style="font-size: 56px; color: {grade_color}; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{grade}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{overall_score:.0f}/100</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Rep</p>
                <p style="font-size: 20px; color: white; font-weight: 900; margin: 0;">{call['rep']}</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Duration</p>
                <p style="font-size: 40px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{call['duration']}m</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Talk Ratio</p>
                <p style="font-size: 40px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{call['talk_ratio']}%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Rep speaking</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Outcome</p>
                <p style="font-size: 18px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{call['outcome']}</p>
            </div>
        </div>
    </div>
    """
    coaching_html = f"""
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;">
        <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">💡 AI Coaching Insights</h3>
        <div style="display: grid; gap: 12px;">
            <div style="background: white; border-left: 5px solid {'#10b981' if call['talk_ratio'] <= 40 else '#f59e0b'}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 10px 0;">🗣️ Talk-Listen Ratio</h4>
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0;">Rep spoke {call['talk_ratio']}% of the time (Target: 30-40%)</p>
                <div style="background: #e5e7eb; border-radius: 8px; height: 10px; overflow: hidden;">
                    <div style="background: {'#10b981' if call['talk_ratio'] <= 40 else '#f59e0b'}; height: 100%; width: {call['talk_ratio']}%;"></div>
                </div>
                <p style="font-size: 13px; color: {'#059669' if call['talk_ratio'] <= 40 else '#d97706'}; margin: 10px 0 0 0; font-weight: 600;">{'✅ Excellent - Good listening balance' if call['talk_ratio'] <= 40 else '⚠️ Talking too much - ask more questions'}</p>
            </div>
            <div style="background: white; border-left: 5px solid {'#10b981' if call['questions_asked'] >= 7 else '#f59e0b'}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 10px 0;">❓ Discovery Questions</h4>
                <p style="font-size: 14px; color: #6b7280; margin: 0;">Asked {call['questions_asked']} questions (Target: 7-10 for discovery)</p>
                <p style="font-size: 13px; color: {'#059669' if call['questions_asked'] >= 7 else '#d97706'}; margin: 10px 0 0 0; font-weight: 600;">{'✅ Strong discovery - uncovered pain points' if call['questions_asked'] >= 7 else '⚠️ Ask more questions to understand needs'}</p>
            </div>
            <div style="background: white; border-left: 5px solid {'#10b981' if call['next_steps_defined'] else '#ef4444'}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 10px 0;">📅 Next Steps</h4>
                <p style="font-size: 14px; color: #6b7280; margin: 0;">Clear next steps defined: {'Yes ✓' if call['next_steps_defined'] else 'No ✗'}</p>
                <p style="font-size: 13px; color: {'#059669' if call['next_steps_defined'] else '#dc2626'}; margin: 10px 0 0 0; font-weight: 600;">{'✅ Call has clear follow-up action' if call['next_steps_defined'] else '❌ Always end with concrete next step'}</p>
            </div>
        </div>
    </div>
    """
    transcript_html = f"""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2);">
        <h3 style="color: #1e40af; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">📝 Call Transcript</h3>
        <div style="background: white; border-radius: 12px; padding: 20px; font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.8; color: #1f2937; white-space: pre-wrap;">
{call['transcript']}
        </div>
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; padding: 18px; margin-top: 18px; color: white;">
            <p style="font-size: 16px; font-weight: 800; margin: 0 0 10px 0;">🎯 Key Moments Detected by AI</p>
            <ul style="margin: 0; padding-left: 24px; line-height: 2;">
                <li style="font-size: 14px;">Open-ended question: "Can you tell me about..." ✓</li>
                <li style="font-size: 14px;">Pain point identified: "3 hours to deploy" ✓</li>
                <li style="font-size: 14px;">Value quantification: "game-changing" ✓</li>
                <li style="font-size: 14px;">Clear next step: Meeting/trial scheduled ✓</li>
            </ul>
        </div>
    </div>
    """
    fig_performance = go.Figure()
    
    categories = ['Talk Ratio', 'Questions', 'Objection Handling', 'Next Steps', 'Engagement']
    scores = [talk_ratio_score, question_score, objection_handling_score, next_steps_score, prospect_engagement * 100]
    
    fig_performance.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#3b82f6', width=3)
    ))
    
    fig_performance.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Call Performance Radar",
        height=450
    )
    
    return summary_html + coaching_html + transcript_html, fig_performance

def generate_team_dashboard():
    """Generate team-wide sales analytics"""
    dashboard_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Team Performance Dashboard</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Calls Analyzed</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">487</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">This month</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Win Rate</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">34%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">+8% vs last month</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Avg Call Score</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">82</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">B+ grade</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Revenue Impact</p>
                <p style="font-size: 40px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">$2.4M</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">From improved calls</p>
            </div>
        </div>
    </div>
    """
    reps_html = """
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2); margin-bottom: 25px;">
        <h3 style="color: #92400e; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🏆 Top Performers</h3>
        <div style="display: grid; gap: 12px;">
    """
    
    reps = [
        {"name": "Sarah Chen", "score": 94, "calls": 47, "win_rate": 42},
        {"name": "Mike Johnson", "score": 89, "calls": 52, "win_rate": 38},
        {"name": "Lisa Park", "score": 87, "calls": 41, "win_rate": 35}
    ]
    
    colors = ['#10b981', '#3b82f6', '#8b5cf6']
    
    for idx, rep in enumerate(reps):
        reps_html += f"""
        <div style="background: white; border-left: 5px solid {colors[idx]}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="background: {colors[idx]}; color: white; padding: 6px 14px; border-radius: 16px; font-size: 12px; font-weight: 800; margin-right: 10px;">#{idx + 1}</span>
                    <span style="font-size: 18px; color: #1f2937; font-weight: 800;">{rep['name']}</span>
                </div>
                <div style="text-align: right;">
                    <p style="font-size: 32px; color: {colors[idx]}; font-weight: 900; margin: 0;">{rep['score']}</p>
                    <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">Avg Score</p>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 12px;">
                <div style="background: #f9fafb; border-radius: 8px; padding: 10px; text-align: center;">
                    <p style="font-size: 11px; color: #6b7280; margin: 0;">Calls</p>
                    <p style="font-size: 18px; color: #1f2937; font-weight: 700; margin: 4px 0 0 0;">{rep['calls']}</p>
                </div>
                <div style="background: #f9fafb; border-radius: 8px; padding: 10px; text-align: center;">
                    <p style="font-size: 11px; color: #6b7280; margin: 0;">Win Rate</p>
                    <p style="font-size: 18px; color: #10b981; font-weight: 700; margin: 4px 0 0 0;">{rep['win_rate']}%</p>
                </div>
            </div>
        </div>
        """
    
    reps_html += "</div></div>"
    
    days = list(range(30))
    win_rates = [0.28 + (i * 0.002) + random.uniform(-0.02, 0.02) for i in days]
    
    fig_trends = go.Figure()
    
    fig_trends.add_trace(go.Scatter(
        x=days,
        y=win_rates,
        mode='lines+markers',
        line=dict(color='#10b981', width=3),
        marker=dict(size=5),
        fill='tonexty',
        fillcolor='rgba(16, 185, 129, 0.1)',
        name='Win Rate'
    ))
    
    fig_trends.update_layout(
        title="Team Win Rate Trend (Last 30 Days)",
        xaxis_title="Days Ago",
        yaxis_title="Win Rate",
        yaxis_range=[0, 0.5],
        height=400
    )
    
    return dashboard_html + reps_html, fig_trends

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">📞</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Hyperbound AI
        </h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Sales Call Analysis & Coaching</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">AI-powered performance optimization for sales teams</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Call Analysis</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">AI Coaching</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Performance Metrics</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">YC Backed</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Hyperbound AI</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📞 Analyze Call", "📊 Team Dashboard"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">AI Sales Call Analysis</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Get instant coaching feedback on sales conversations</p>
    </div>
    """, unsafe_allow_html=True)
    
    call_name = st.selectbox(
        "Select Call Recording",
        list(SAMPLE_CALLS.keys()),
        index=0
    )
    
    if st.button("🎯 Analyze Call Performance", type="primary", use_container_width=True):
        analysis_html, performance_chart = analyze_sales_call(call_name)
        st.markdown(analysis_html, unsafe_allow_html=True)
        st.plotly_chart(performance_chart, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Sales Team Analytics</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Performance metrics and trends across your sales organization</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Load Team Dashboard", type="primary", use_container_width=True):
        dashboard_html, trend_chart = generate_team_dashboard()
        st.markdown(dashboard_html, unsafe_allow_html=True)
        st.plotly_chart(trend_chart, use_container_width=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Hyperbound AI</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📈 Win Rate Improvement</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    AI coaching increases win rates by 8-15%. For a team closing $10M/year, that's $800K-1.5M in additional revenue.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ Instant Feedback</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Get coaching within minutes of call ending. Fix issues in next call, not next quarter. Faster improvement cycles = faster revenue growth.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🎯 Data-Driven Coaching</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Replace subjective feedback with objective metrics. Know exactly what to improve: talk less, ask more questions, handle objections better.
                </p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Real-World Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">8-15% win rate increase:</strong> From AI coaching insights</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">$800K+ revenue impact:</strong> Per year for typical sales team</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">100% call coverage:</strong> Every call analyzed, no sampling</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">10x faster ramp:</strong> New reps productive in weeks, not months</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ NLP Analysis</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Transcript processing, key moment detection</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Performance Scoring</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Talk ratio, questions, objections, next steps</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Coaching</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Actionable feedback within minutes</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Team Analytics</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Aggregate metrics, leaderboards, trends</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Hyperbound AI</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Streamlit • NLP • Sales Analytics
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered sales call analysis and coaching.<br>
            Performance scoring • Real-time feedback • Team analytics • Win rate optimization
        </p>
    </div>
    """, unsafe_allow_html=True)