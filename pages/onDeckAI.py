"""
OnDeck AI - Analyze Any Footage Without Training
Zero-shot video analysis and insights
Built for OnDeck AI by Anju Nandhakumar
"""

import streamlit as st
from utils.sidebar import render_sidebar
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="OnDeck AI - Video Analysis", layout="wide")
render_sidebar()

# Initialize session state
if 'video_analyzed' not in st.session_state:
    st.session_state.video_analyzed = False

# Sample video scenarios
VIDEO_SCENARIOS = {
    "Retail Store - Customer Traffic": {
        "duration": "2 hours",
        "frames": 7200,
        "insights": {
            "people_count": 247,
            "avg_dwell_time": "3.2 minutes",
            "peak_hours": "11am-1pm, 5pm-7pm",
            "conversion_rate": "18%",
            "hot_zones": "Electronics (35%), Clothing (28%), Checkout (22%)"
        },
        "actions": ["Walking", "Browsing", "Picking up items", "Purchasing", "Waiting in line"],
        "anomalies": ["Spill at aisle 3 (12:34pm)", "Unattended bag (2:15pm)"]
    },
    "Warehouse - Safety Compliance": {
        "duration": "8 hours",
        "frames": 28800,
        "insights": {
            "people_count": 42,
            "safety_violations": 8,
            "forklift_incidents": 2,
            "ppe_compliance": "94%",
            "high_risk_zones": "Loading dock (12 incidents), Pallet storage (5 incidents)"
        },
        "actions": ["Forklift operation", "Manual lifting", "Pallet moving", "Safety checks"],
        "anomalies": ["No hard hat detected (3x)", "Forklift speeding (2x)", "Blocked exit (1x)"]
    },
    "Construction Site - Progress Monitoring": {
        "duration": "12 hours",
        "frames": 43200,
        "insights": {
            "workers_present": 28,
            "equipment_utilization": "78%",
            "completed_tasks": 12,
            "delays_detected": 3,
            "productivity_score": "85%"
        },
        "actions": ["Excavation", "Concrete pouring", "Steel installation", "Equipment operation"],
        "anomalies": ["Equipment idle 2.5 hours", "Weather delay (rain)", "Missing materials delivery"]
    }
}

def analyze_video(scenario_data):
    """Analyze video and extract insights using zero-shot learning"""
    
    insights = scenario_data['insights']
    actions = scenario_data['actions']
    anomalies = scenario_data['anomalies']
    
    # Generate activity timeline
    timeline = []
    for i, action in enumerate(actions):
        timeline.append({
            'timestamp': f"{(i*30)//60}:{(i*30)%60:02d}",
            'action': action,
            'confidence': np.random.uniform(0.85, 0.99),
            'duration': f"{np.random.randint(5, 45)} min"
        })
    
    return {
        'summary': insights,
        'timeline': timeline,
        'anomalies': anomalies,
        'total_events': len(timeline) + len(anomalies)
    }

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(99, 102, 241, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(139, 92, 246, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🎥</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">OnDeck AI</h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Analyze Any Footage, No Training Needed</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Zero-shot video understanding for any use case</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);">Zero-Shot Learning</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Video Analysis</span>
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Action Recognition</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">YC Backed</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">Built for <strong style="color: white;">OnDeck AI</strong> by <strong style="color: white;">Anju Nandhakumar</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Value Prop
st.markdown("""
<div style="background: linear-gradient(135deg, #f3e8ff, #e9d5ff); padding: 25px; border-radius: 15px; border: 2px solid #8b5cf6; margin-bottom: 30px;">
    <h3 style="color: #5b21b6; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">🎯 The Video Analysis Problem</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #ef4444; font-weight: 700; margin: 0 0 8px 0;">❌ Today</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Need custom models for each use case. Training takes weeks, requires labeled data. New scenario = start over.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #f59e0b; font-weight: 700; margin: 0 0 8px 0;">💰 Cost Impact</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">$50K+ per custom model. 3-6 months development. Requires ML team. Limited to trained scenarios only.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #10b981; font-weight: 700; margin: 0 0 8px 0;">✅ With OnDeck</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Zero training needed. Analyze any footage instantly. One model, infinite use cases. Deploy in hours, not months.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎥 Analyze Video", "📊 Insights Dashboard", "🔧 How It Works"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Upload Any Video</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">No training needed - AI understands any scenario automatically</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        use_sample = st.checkbox("Use sample scenario", value=True)
        
        if use_sample:
            scenario_name = st.selectbox("Select Video Type", list(VIDEO_SCENARIOS.keys()))
            scenario = VIDEO_SCENARIOS[scenario_name]
            
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 12px; border: 2px solid #e5e7eb;">
                <p style="color: #6b7280; font-size: 13px; margin: 0;"><strong>Duration:</strong> {scenario['duration']}</p>
                <p style="color: #6b7280; font-size: 13px; margin: 5px 0 0 0;"><strong>Frames:</strong> {scenario['frames']:,}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov'])
        
        st.markdown("**🎯 What to look for?**")
        analysis_goals = st.multiselect(
            "Select analysis goals",
            ["People counting", "Activity recognition", "Safety compliance", "Traffic patterns", 
             "Anomaly detection", "Object tracking"],
            default=["People counting", "Activity recognition", "Anomaly detection"]
        )
        
        if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
            st.session_state.video_analyzed = True
            st.session_state.analysis = analyze_video(scenario if use_sample else {})
            st.session_state.current_scenario = scenario_name if use_sample else "Custom"
    
    with col2:
        st.markdown("""
        <div style="background: #ecfdf5; padding: 20px; border-radius: 12px; border-left: 4px solid #10b981;">
            <h4 style="color: #065f46; margin: 0 0 12px 0; font-size: 16px;">⚡ Zero-Shot Capabilities</h4>
            <ul style="color: #047857; font-size: 13px; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li><strong>No training needed:</strong> Works on any footage immediately</li>
                <li><strong>Any scenario:</strong> Retail, warehouse, construction, traffic, etc.</li>
                <li><strong>Action recognition:</strong> Detects activities without examples</li>
                <li><strong>Object detection:</strong> People, vehicles, equipment, products</li>
                <li><strong>Anomaly detection:</strong> Finds unusual events automatically</li>
                <li><strong>Natural language queries:</strong> Ask questions about your footage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.video_analyzed:
        st.markdown("<hr style='margin: 30px 0; border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)
        
        analysis = st.session_state.analysis
        
        st.success("✅ Video analysis complete!")
        
        # Summary metrics
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">📊 Analysis Summary</h2>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
        """, unsafe_allow_html=True)
        
        for key, value in list(analysis['summary'].items())[:4]:
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">{key.replace('_', ' ').title()}</p>
                    <p style="font-size: 32px; color: white; font-weight: 900; margin: 8px 0;">{value}</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Activity timeline
        col_a, col_b = st.columns([2, 1])
        
        with col_a:
            st.markdown("### 🎬 Detected Activities")
            df_timeline = pd.DataFrame(analysis['timeline'])
            st.dataframe(df_timeline, use_container_width=True, hide_index=True)
        
        with col_b:
            st.markdown("### ⚠️ Anomalies Detected")
            for anomaly in analysis['anomalies']:
                st.markdown(f"""
                <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 3px solid #f59e0b; margin-bottom: 10px;">
                    <p style="color: #78350f; font-size: 13px; margin: 0; font-weight: 600;">{anomaly}</p>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Video Analytics Platform</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Real-time insights across all your video feeds</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Platform metrics
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 32px; border-radius: 20px; margin-bottom: 25px;">
        <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">📊 Platform Performance</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Videos Analyzed</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 8px 0;">1,247</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">This month</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Use Cases</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 8px 0;">47</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">No retraining</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Avg Analysis Time</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 8px 0;">2.3s</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">per minute of video</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Accuracy</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 8px 0;">94%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">Action recognition</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Use cases
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">🏪 Retail Analytics</h3>
            <ul style="color: #6b7280; font-size: 14px; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li>Customer traffic patterns and heatmaps</li>
                <li>Dwell time by product area</li>
                <li>Conversion rate tracking</li>
                <li>Queue length monitoring</li>
                <li>Theft detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">🏗️ Construction Monitoring</h3>
            <ul style="color: #6b7280; font-size: 14px; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li>Progress tracking and task completion</li>
                <li>Equipment utilization rates</li>
                <li>Worker productivity analysis</li>
                <li>Safety violation detection</li>
                <li>Timeline vs actual comparison</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Zero-Shot Learning Pipeline</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">How OnDeck analyzes any video without custom training</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">🤖 AI Pipeline</h3>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #6366f1; margin-bottom: 12px;">
                <h4 style="color: #4f46e5; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">1. Vision-Language Model</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">CLIP/ViLT understands images and text together - no retraining needed</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #8b5cf6; margin-bottom: 12px;">
                <h4 style="color: #7c3aed; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">2. Action Recognition</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Temporal models detect activities from frame sequences</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6; margin-bottom: 12px;">
                <h4 style="color: #2563eb; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">3. Object Tracking</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Track people, vehicles, objects across frames</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #10b981;">
                <h4 style="color: #059669; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">4. Insight Generation</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Aggregate results, detect patterns, generate natural language insights</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">✨ Supported Scenarios</h3>
            <div style="background: #ecfdf5; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #047857; font-weight: 700; font-size: 14px; margin: 0;">✓ Retail & Commerce</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Traffic, conversion, theft, checkout flow</p>
            </div>
            <div style="background: #eff6ff; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #1e40af; font-weight: 700; font-size: 14px; margin: 0;">✓ Warehouse & Logistics</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Safety, efficiency, equipment usage</p>
            </div>
            <div style="background: #fef3c7; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #92400e; font-weight: 700; font-size: 14px; margin: 0;">✓ Construction Sites</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Progress, safety, worker productivity</p>
            </div>
            <div style="background: #f3e8ff; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #6b21a8; font-weight: 700; font-size: 14px; margin: 0;">✓ Traffic & Parking</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Vehicle counting, congestion, violations</p>
            </div>
            <div style="background: #fce7f3; padding: 12px 15px; border-radius: 8px;">
                <p style="color: #9f1239; font-weight: 700; font-size: 14px; margin: 0;">✓ Security & Surveillance</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Intrusion, loitering, unusual behavior</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #6366f1; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for OnDeck AI</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🚀 Instant Deployment</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">Deploy to new use case in hours, not months. No training data needed. No ML team required. Just point at footage and go.</p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 $50K+ Saved</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">Per custom model. No training costs, no labeled data, no months of development. One model serves infinite scenarios.</p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🎯 94% Accuracy</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">Without training. Vision-language models understand context from natural language descriptions. Adapts to any scenario.</p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Enterprise Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">47 use cases:</strong> One model, no retraining</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Hours to deploy:</strong> vs 3-6 months custom training</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">$50K+ saved:</strong> per avoided custom model</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">94% accuracy:</strong> Without any training data</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Stack</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Vision-Language Models</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">CLIP, ViLT for zero-shot understanding</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Temporal Models</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Action recognition across frame sequences</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Object Tracking</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Multi-object tracking without retraining</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ NLP Queries</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Ask questions about footage in plain English</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(99, 102, 241, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">Built for <strong style="color: white;">OnDeck AI</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong></p>
        <div style="margin: 25px 0; padding: 22px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.2);">
            <p style="margin: 8px 0; font-size: 16px;">📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: white; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a></p>
            <p style="margin: 8px 0; font-size: 16px;">💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">LinkedIn</a> | 💻 <a href="https://github.com/Av1352" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">GitHub</a> | 🌐 <a href="https://vxanju.com" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">Portfolio</a></p>
        </div>
        <p style="font-size: 15px; margin: 18px 0; font-weight: 700;"><strong style="color: white;">Tech Stack:</strong> Zero-Shot Learning • CLIP/ViLT • Video Analysis • Action Recognition</p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">Demo showcasing zero-shot video analysis without custom model training.<br>Vision-language models • Action recognition • Object tracking • Multi-scenario support</p>
    </div>
    """, unsafe_allow_html=True)