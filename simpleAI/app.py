"""
Simple AI - Enterprise Phone Agent Platform
Automated phone call handling with AI
Built for Simple AI by Anju Nandhakumar
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import random

# Call scenarios and intents
CALL_INTENTS = {
    "Account Support": {
        "keywords": ["account", "password", "login", "access", "reset"],
        "avg_duration": 3.2,
        "resolution_rate": 0.89,
        "sentiment": "neutral"
    },
    "Billing Question": {
        "keywords": ["bill", "charge", "payment", "invoice", "refund"],
        "avg_duration": 4.5,
        "resolution_rate": 0.92,
        "sentiment": "negative"
    },
    "Technical Support": {
        "keywords": ["not working", "error", "broken", "help", "fix"],
        "avg_duration": 6.8,
        "resolution_rate": 0.78,
        "sentiment": "negative"
    },
    "Product Inquiry": {
        "keywords": ["features", "pricing", "demo", "information", "how does"],
        "avg_duration": 2.1,
        "resolution_rate": 0.95,
        "sentiment": "positive"
    },
    "Appointment Scheduling": {
        "keywords": ["appointment", "schedule", "meeting", "calendar", "book"],
        "avg_duration": 1.8,
        "resolution_rate": 0.97,
        "sentiment": "neutral"
    }
}

SAMPLE_CONVERSATIONS = {
    "Account Support": {
        "caller": "Hi, I can't log into my account. I keep getting an error message.",
        "ai": "I'd be happy to help you regain access to your account. Let me verify your identity first. Can you provide your email address associated with the account?",
        "caller_2": "Sure, it's john.smith@email.com",
        "ai_2": "Thank you, John. I've found your account. I'm going to send a password reset link to that email address. You should receive it within 2 minutes. Is there anything else I can help you with today?",
        "outcome": "Resolved"
    },
    "Billing Question": {
        "caller": "I was charged twice this month. I need a refund immediately!",
        "ai": "I understand your frustration, and I apologize for the inconvenience. Let me look into this right away. Can you provide the charge amounts and dates you're seeing?",
        "caller_2": "Two charges of $49.99, both on December 15th.",
        "ai_2": "I see the duplicate charge in our system. This was a processing error on our end. I've initiated a full refund of $49.99, which will appear in your account within 3-5 business days. I've also added a $10 credit to your account for the inconvenience. Is there anything else I can help with?",
        "outcome": "Resolved"
    },
    "Technical Support": {
        "caller": "The app keeps crashing whenever I try to upload a file. This has been happening for 3 days!",
        "ai": "I'm sorry you're experiencing this issue. Let me help you troubleshoot. First, can you tell me what type of file you're trying to upload and approximately how large it is?",
        "caller_2": "It's a PDF document, about 15 megabytes.",
        "ai_2": "Thank you. Files over 10MB can sometimes cause issues with our current upload system. I'm going to escalate this to our technical team for a permanent fix, but in the meantime, I can help you compress the PDF or split it into smaller files. Would you like me to walk you through that process?",
        "outcome": "Escalated"
    },
    "Product Inquiry": {
        "caller": "I'm interested in your premium plan. What features does it include?",
        "ai": "Great question! Our Premium plan includes unlimited storage, priority support, advanced analytics, and API access. It's $99 per month or $990 annually, which saves you 2 months. Would you like me to start a free 14-day trial for you?",
        "caller_2": "Yes, that would be great!",
        "ai_2": "Perfect! I'm setting up your trial now. You'll receive an email with login credentials within the next minute. The trial includes full Premium features, and you won't be charged until the 14 days are up. Is there anything specific you'd like to try during your trial?",
        "outcome": "Resolved"
    }
}

def analyze_call(call_type):
    """Analyze a simulated phone call"""
    
    intent_data = CALL_INTENTS[call_type]
    conversation = SAMPLE_CONVERSATIONS[call_type]
    
    # Generate call metadata
    duration = intent_data["avg_duration"] + random.uniform(-0.5, 0.5)
    resolution = random.random() < intent_data["resolution_rate"]
    
    # Sentiment analysis
    sentiments = {
        "positive": {"score": random.uniform(0.7, 0.95), "emoji": "😊", "color": "#10b981"},
        "neutral": {"score": random.uniform(0.4, 0.6), "emoji": "😐", "color": "#f59e0b"},
        "negative": {"score": random.uniform(0.1, 0.3), "emoji": "😞", "color": "#ef4444"}
    }
    
    sentiment_type = intent_data["sentiment"]
    sentiment_info = sentiments[sentiment_type]
    
    # Call summary HTML
    call_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📞 Call Analysis</h2>
        
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Intent</p>
                <p style="font-size: 18px; color: white; font-weight: 900; margin: 0;">{call_type}</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Duration</p>
                <p style="font-size: 32px; color: white; font-weight: 900; margin: 0;">{duration:.1f}m</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Sentiment</p>
                <p style="font-size: 36px; color: white; margin: 0;">{sentiment_info['emoji']}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0; text-transform: capitalize;">{sentiment_type}</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Resolution</p>
                <p style="font-size: 32px; color: {'#86efac' if resolution else '#fca5a5'}; font-weight: 900; margin: 0;">{'✓' if resolution else '✗'}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{conversation['outcome']}</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">CSAT Score</p>
                <p style="font-size: 32px; color: #fbbf24; font-weight: 900; margin: 0;">{random.randint(4, 5)}/5</p>
            </div>
        </div>
    </div>
    """
    
    # Conversation transcript
    transcript_html = f"""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🗣️ Call Transcript</h3>
        
        <div style="background: white; border-radius: 14px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-left: 5px solid #3b82f6;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <div style="background: #3b82f6; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 20px;">👤</span>
                </div>
                <div>
                    <p style="font-size: 14px; color: #1e40af; font-weight: 700; margin: 0;">Caller</p>
                    <p style="font-size: 12px; color: #6b7280; margin: 0;">0:00</p>
                </div>
            </div>
            <p style="font-size: 15px; color: #1f2937; margin: 0; line-height: 1.6;">"{conversation['caller']}"</p>
        </div>
        
        <div style="background: white; border-radius: 14px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-left: 5px solid #10b981;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <div style="background: #10b981; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 20px;">🤖</span>
                </div>
                <div>
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0;">AI Agent</p>
                    <p style="font-size: 12px; color: #6b7280; margin: 0;">0:05</p>
                </div>
            </div>
            <p style="font-size: 15px; color: #1f2937; margin: 0; line-height: 1.6;">"{conversation['ai']}"</p>
        </div>
        
        <div style="background: white; border-radius: 14px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-left: 5px solid #3b82f6;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <div style="background: #3b82f6; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 20px;">👤</span>
                </div>
                <div>
                    <p style="font-size: 14px; color: #1e40af; font-weight: 700; margin: 0;">Caller</p>
                    <p style="font-size: 12px; color: #6b7280; margin: 0;">0:{duration*0.3:.0f}</p>
                </div>
            </div>
            <p style="font-size: 15px; color: #1f2937; margin: 0; line-height: 1.6;">"{conversation['caller_2']}"</p>
        </div>
        
        <div style="background: white; border-radius: 14px; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-left: 5px solid #10b981;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <div style="background: #10b981; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 20px;">🤖</span>
                </div>
                <div>
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0;">AI Agent</p>
                    <p style="font-size: 12px; color: #6b7280; margin: 0;">0:{duration*0.6:.0f}</p>
                </div>
            </div>
            <p style="font-size: 15px; color: #1f2937; margin: 0; line-height: 1.6;">"{conversation['ai_2']}"</p>
        </div>
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 2px solid #10b981; border-radius: 12px; padding: 18px; margin-top: 20px; text-align: center;">
            <p style="font-size: 16px; color: #065f46; font-weight: 800; margin: 0;">✅ Call Ended • {conversation['outcome']} • Duration: {duration:.1f} minutes</p>
        </div>
    </div>
    """
    
    # Call insights
    insights_html = f"""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(168, 85, 247, 0.2); margin-bottom: 25px;">
        <h3 style="color: #6b21a8; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🎯 Call Insights</h3>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: white; border-radius: 14px; padding: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 15px 0;">🎭 Sentiment Analysis</h4>
                <div style="margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 14px; color: #6b7280; font-weight: 600;">Overall Sentiment</span>
                        <span style="font-size: 24px;">{sentiment_info['emoji']}</span>
                    </div>
                    <div style="background: #e5e7eb; border-radius: 12px; height: 12px; overflow: hidden;">
                        <div style="background: {sentiment_info['color']}; height: 100%; width: {sentiment_info['score']*100}%; transition: width 0.3s;"></div>
                    </div>
                    <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0; text-align: right;">{sentiment_info['score']:.0%} {sentiment_type}</p>
                </div>
                
                <div style="background: #f9fafb; border-radius: 10px; padding: 12px;">
                    <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Caller Tone</p>
                    <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0; text-transform: capitalize;">{sentiment_type}</p>
                </div>
            </div>
            
            <div style="background: white; border-radius: 14px; padding: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 15px 0;">📊 Performance Metrics</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                    <div style="background: #f0f9ff; border-radius: 10px; padding: 14px; text-align: center;">
                        <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">First Response</p>
                        <p style="font-size: 24px; color: #3b82f6; font-weight: 900; margin: 0;">5s</p>
                    </div>
                    <div style="background: #fef3c7; border-radius: 10px; padding: 14px; text-align: center;">
                        <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Avg Response</p>
                        <p style="font-size: 24px; color: #f59e0b; font-weight: 900; margin: 0;">8s</p>
                    </div>
                    <div style="background: #f3e8ff; border-radius: 10px; padding: 14px; text-align: center;">
                        <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Turns</p>
                        <p style="font-size: 24px; color: #a855f7; font-weight: 900; margin: 0;">4</p>
                    </div>
                    <div style="background: #d1fae5; border-radius: 10px; padding: 14px; text-align: center;">
                        <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Resolution</p>
                        <p style="font-size: 24px; color: #10b981; font-weight: 900; margin: 0;">{'✓' if resolution else '✗'}</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    # AI performance
    ai_perf_html = f"""
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2);">
        <h3 style="color: #065f46; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">🤖 AI Agent Performance</h3>
        
        <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
            <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 12px 0;">Capabilities Demonstrated</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                <span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 8px 18px; border-radius: 20px; font-size: 13px; font-weight: 700; box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);">Intent Classification</span>
                <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 8px 18px; border-radius: 20px; font-size: 13px; font-weight: 700; box-shadow: 0 2px 6px rgba(139, 92, 246, 0.3);">Empathy Detection</span>
                <span style="background: linear-gradient(135deg, #ec4899 0%, #db2777 100%); color: white; padding: 8px 18px; border-radius: 20px; font-size: 13px; font-weight: 700; box-shadow: 0 2px 6px rgba(236, 72, 153, 0.3);">Account Lookup</span>
                <span style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 8px 18px; border-radius: 20px; font-size: 13px; font-weight: 700; box-shadow: 0 2px 6px rgba(245, 158, 11, 0.3);">Problem Solving</span>
                <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 8px 18px; border-radius: 20px; font-size: 13px; font-weight: 700; box-shadow: 0 2px 6px rgba(16, 185, 129, 0.3);">Natural Language</span>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div style="background: white; border-radius: 12px; padding: 18px; text-align: center;">
                <p style="font-size: 13px; color: #6b7280; margin: 0 0 8px 0;">Response Quality</p>
                <p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">A+</p>
                <p style="font-size: 12px; color: #9ca3af; margin: 8px 0 0 0;">Professional & Clear</p>
            </div>
            
            <div style="background: white; border-radius: 12px; padding: 18px; text-align: center;">
                <p style="font-size: 13px; color: #6b7280; margin: 0 0 8px 0;">Accuracy</p>
                <p style="font-size: 32px; color: #3b82f6; font-weight: 900; margin: 0;">98%</p>
                <p style="font-size: 12px; color: #9ca3af; margin: 8px 0 0 0;">Correct Information</p>
            </div>
            
            <div style="background: white; border-radius: 12px; padding: 18px; text-align: center;">
                <p style="font-size: 13px; color: #6b7280; margin: 0 0 8px 0;">Empathy Score</p>
                <p style="font-size: 32px; color: #ec4899; font-weight: 900; margin: 0;">9.2/10</p>
                <p style="font-size: 12px; color: #9ca3af; margin: 8px 0 0 0;">Highly Empathetic</p>
            </div>
        </div>
    </div>
    """
    
    # Create sentiment timeline
    timeline_points = 8
    sentiment_scores = []
    base_sentiment = sentiment_info['score']
    
    for i in range(timeline_points):
        score = base_sentiment + random.uniform(-0.15, 0.15)
        sentiment_scores.append(max(0, min(1, score)))
    
    fig_sentiment = go.Figure()
    
    fig_sentiment.add_trace(go.Scatter(
        x=list(range(timeline_points)),
        y=sentiment_scores,
        mode='lines+markers',
        line=dict(color=sentiment_info['color'], width=3),
        marker=dict(size=8),
        fill='tonexty',
        fillcolor=f"rgba({int(sentiment_info['color'][1:3], 16)}, {int(sentiment_info['color'][3:5], 16)}, {int(sentiment_info['color'][5:7], 16)}, 0.2)",
        name='Sentiment'
    ))
    
    fig_sentiment.update_layout(
        title="Sentiment Throughout Call",
        xaxis_title="Call Progress",
        yaxis_title="Sentiment Score",
        yaxis_range=[0, 1],
        height=350
    )
    
    return call_html + transcript_html + insights_html, fig_sentiment

def generate_call_center_analytics():
    """Generate call center analytics dashboard"""
    
    dashboard_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 Call Center Performance</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Calls</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">8,547</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Last 24 hours</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Resolution Rate</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">91.5%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">7,821 resolved</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Avg Handle Time</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">3.8m</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">vs 6.2m human</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">CSAT Score</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">4.6/5</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">92% satisfied</p>
            </div>
        </div>
    </div>
    """
    
    # Intent distribution
    intent_html = """
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2); margin-bottom: 25px;">
        <h3 style="color: #92400e; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🎯 Call Intent Distribution</h3>
        <div style="display: grid; gap: 12px;">
    """
    
    total_calls = 8547
    intent_colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b']
    
    for idx, (intent, data) in enumerate(CALL_INTENTS.items()):
        count = random.randint(1000, 2500)
        pct = (count / total_calls) * 100
        
        intent_html += f"""
        <div style="background: white; border-left: 5px solid {intent_colors[idx]}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <span style="font-size: 18px; color: #1f2937; font-weight: 800;">{intent}</span>
                    <p style="font-size: 13px; color: #6b7280; margin: 4px 0 0 0;">Avg: {data['avg_duration']:.1f}m • Resolution: {data['resolution_rate']:.0%}</p>
                </div>
                <div style="text-align: right;">
                    <p style="font-size: 28px; color: {intent_colors[idx]}; font-weight: 900; margin: 0;">{count:,}</p>
                    <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">{pct:.1f}%</p>
                </div>
            </div>
            <div style="background: #e5e7eb; border-radius: 8px; height: 8px; overflow: hidden;">
                <div style="background: {intent_colors[idx]}; height: 100%; width: {pct}%; transition: width 0.3s;"></div>
            </div>
        </div>
        """
    
    intent_html += "</div></div>"
    
    # Create charts
    
    # 1. Calls over time
    hours = list(range(24))
    call_volume = [random.randint(250, 450) for _ in hours]
    
    fig_volume = go.Figure()
    
    fig_volume.add_trace(go.Scatter(
        x=hours,
        y=call_volume,
        mode='lines+markers',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=6),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)',
        name='Call Volume'
    ))
    
    fig_volume.update_layout(
        title="Call Volume (Last 24 Hours)",
        xaxis_title="Hour",
        yaxis_title="Number of Calls",
        height=400
    )
    
    # 2. Resolution rate by intent
    intents = list(CALL_INTENTS.keys())
    resolution_rates = [CALL_INTENTS[i]["resolution_rate"] + random.uniform(-0.05, 0.05) for i in intents]
    
    fig_resolution = go.Figure(data=[
        go.Bar(
            x=intents,
            y=resolution_rates,
            marker_color=intent_colors,
            text=[f'{r:.1%}' for r in resolution_rates],
            textposition='outside'
        )
    ])
    
    fig_resolution.add_hline(y=0.90, line_dash="dash", line_color="#059669", 
                             annotation_text="Target: 90%", annotation_position="right")
    
    fig_resolution.update_layout(
        title="Resolution Rate by Intent Type",
        yaxis_title="Resolution Rate",
        yaxis_range=[0, 1.1],
        height=400
    )
    
    # 3. Intent distribution pie
    intent_counts = {intent: random.randint(1000, 2500) for intent in intents}
    
    fig_intent = go.Figure(data=[go.Pie(
        labels=list(intent_counts.keys()),
        values=list(intent_counts.values()),
        marker=dict(colors=intent_colors),
        hole=0.4,
        textinfo='label+percent',
        textfont=dict(size=13, color='white', family='Arial Black')
    )])
    
    fig_intent.update_layout(
        title="Call Distribution by Intent (24h)",
        height=450
    )
    
    return dashboard_html + intent_html, fig_volume, fig_resolution, fig_intent

custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">📞</span>
        </div>
        
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Simple AI Phone Agent
        </h1>
        
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Enterprise Voice AI for Customer Service</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Automated phone calls • Natural conversations • 24/7 availability</p>
        
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 800px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Voice AI</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Intent Classification</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Sentiment Analysis</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">YC Backed</span>
        </div>
        
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Simple AI</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("📞 Call Simulation"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Simulate AI Phone Agent Conversations</h3>
                <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Select a call type to see how the AI agent handles different customer scenarios</p>
            </div>
            """)
            
            call_type_dropdown = gr.Dropdown(
                choices=list(CALL_INTENTS.keys()),
                value="Account Support",
                label="Select Call Type"
            )
            
            simulate_btn = gr.Button("📞 Simulate Call", variant="primary", size="lg")
            
            call_output = gr.HTML(label="Call Transcript & Analysis")
            sentiment_chart = gr.Plot(label="Sentiment Timeline")
            
            simulate_btn.click(
                fn=analyze_call,
                inputs=[call_type_dropdown],
                outputs=[call_output, sentiment_chart]
            )
        
        with gr.Tab("📊 Analytics Dashboard"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Call Center Analytics</h3>
                <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Real-time performance metrics across all AI phone agents</p>
            </div>
            """)
            
            analytics_btn = gr.Button("📊 Load Analytics", variant="primary", size="lg")
            
            analytics_output = gr.HTML(label="Performance Dashboard")
            volume_chart = gr.Plot(label="Call Volume")
            resolution_chart = gr.Plot(label="Resolution by Intent")
            intent_chart = gr.Plot(label="Intent Distribution")
            
            analytics_btn.click(
                fn=generate_call_center_analytics,
                inputs=[],
                outputs=[analytics_output, volume_chart, resolution_chart, intent_chart]
            )
    
    gr.HTML("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Simple AI</h2>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Massive Cost Savings</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Human call center agent costs $15-25/hour. AI agent costs $0.50-2.00/hour. Handle 8,500 calls/day = $200K-400K saved annually per enterprise customer.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 24/7 Availability</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    No shifts, no breaks, no holidays. AI agents handle after-hours calls, weekend spikes, holiday volume. Never miss a customer call, ever.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📈 Scalability</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Instant scaling. Go from 100 calls/hour to 10,000 calls/hour with zero additional hiring, training, or infrastructure. Handle Black Friday spikes effortlessly.
                </p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Real-World Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">40% faster resolution:</strong> 3.8min vs 6.2min human average handle time</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">91.5% resolution rate:</strong> Most calls handled without human escalation</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">4.6/5 CSAT:</strong> Matches or beats human agent satisfaction scores</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">$300K+ saved/year:</strong> Per enterprise customer (vs human call center)</li>
            </ul>
        </div>
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time STT/TTS</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Sub-second speech recognition and synthesis</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Intent Classification</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">95%+ accuracy routing to right workflow</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Sentiment Analysis</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Detect frustration, escalate to human if needed</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ CRM Integration</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Salesforce, HubSpot, Zendesk compatible</p>
                </div>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Simple AI</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Gradio • Plotly • NLP • Voice AI
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing enterprise AI phone agents for customer service automation.<br>
            Natural conversations • Intent routing • Sentiment tracking • Performance analytics
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()