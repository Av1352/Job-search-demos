"""
Tempo - Live Home Fitness with Computer Vision
AI-powered form correction and real-time feedback
Built for Tempo by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Tempo", page_icon="💪", layout="wide")

# Exercises
EXERCISES = {
    'Squat': {'difficulty': 'Beginner', 'muscle_groups': ['Quads', 'Glutes', 'Core'], 'form_checks': 8},
    'Deadlift': {'difficulty': 'Intermediate', 'muscle_groups': ['Back', 'Glutes', 'Hamstrings'], 'form_checks': 12},
    'Bench Press': {'difficulty': 'Intermediate', 'muscle_groups': ['Chest', 'Triceps', 'Shoulders'], 'form_checks': 10},
    'Plank': {'difficulty': 'Beginner', 'muscle_groups': ['Core', 'Shoulders', 'Back'], 'form_checks': 6},
    'Bicep Curl': {'difficulty': 'Beginner', 'muscle_groups': ['Biceps', 'Forearms'], 'form_checks': 5}
}

# Form analysis
FORM_ERRORS = {
    'Knee Cave': 'Knees collapsing inward during squat',
    'Rounded Back': 'Lumbar spine rounding on deadlift',
    'Elbow Flare': 'Elbows too wide on bench press',
    'Hip Drop': 'Hips sagging below shoulder line on plank',
    'Shoulder Swing': 'Using momentum instead of bicep on curls'
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #f43f5e 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">💪</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Tempo</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Live Home Fitness with Computer Vision</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Real-time form correction • 3D pose tracking • AI coaching</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["💪 Live Training", "📊 Form Analysis", "📈 Performance Tracking", "💡 Technology"])

with tab1:
    st.markdown("### Real-Time Form Correction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Workout Configuration**")
        
        exercise = st.selectbox("Select Exercise", list(EXERCISES.keys()))
        
        st.markdown("**Session Settings**")
        
        reps = st.slider("Target Reps", 5, 20, 10)
        sets = st.slider("Sets", 1, 5, 3)
        
        st.markdown("**AI Coaching**")
        
        enable_realtime = st.checkbox("Real-time feedback", value=True)
        enable_voice = st.checkbox("Voice cues", value=True)
        enable_counting = st.checkbox("Auto rep counting", value=True)
        
        st.markdown("**3D Tracking**")
        st.info("📹 Camera captures 3D pose data at 60 FPS")
        
        start_btn = st.button("💪 Start Workout", type="primary", use_container_width=True)
    
    with col2:
        if start_btn:
            st.markdown("**Live Form Analysis**")
            
            with st.spinner("Analyzing form..."):
                import time
                time.sleep(1.5)
            
            # Simulate rep counting
            current_rep = np.random.randint(1, reps+1)
            form_score = np.random.uniform(85, 98)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f43f5e 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Current Rep: {current_rep}/{reps}</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Form Score</p>
                        <p style="font-size: 36px; color: white; font-weight: 900; margin: 0;">{form_score:.1f}%</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Status</p>
                        <p style="font-size: 36px; color: white; font-weight: 900; margin: 0;">{'✅' if form_score > 90 else '⚠️'}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Real-Time Feedback**")
            
            if form_score < 90:
                st.warning("⚠️ **Correction:** Knees tracking forward - keep them aligned with toes")
            else:
                st.success("✅ **Excellent form!** Keep it up!")
            
            # Joint tracking visualization
            st.markdown("**3D Pose Tracking**")
            
            # Simulate skeleton tracking
            fig1 = go.Figure()
            
            # Skeleton points (simplified)
            joints = {
                'head': (0, 1.7), 'neck': (0, 1.5),
                'l_shoulder': (-0.2, 1.4), 'r_shoulder': (0.2, 1.4),
                'l_elbow': (-0.3, 1.0), 'r_elbow': (0.3, 1.0),
                'l_wrist': (-0.35, 0.6), 'r_wrist': (0.35, 0.6),
                'hip': (0, 0.9),
                'l_knee': (-0.15, 0.5), 'r_knee': (0.15, 0.5),
                'l_ankle': (-0.15, 0.1), 'r_ankle': (0.15, 0.1)
            }
            
            # Draw connections
            connections = [
                ('head', 'neck'), ('neck', 'l_shoulder'), ('neck', 'r_shoulder'),
                ('l_shoulder', 'l_elbow'), ('r_shoulder', 'r_elbow'),
                ('l_elbow', 'l_wrist'), ('r_elbow', 'r_wrist'),
                ('neck', 'hip'), ('hip', 'l_knee'), ('hip', 'r_knee'),
                ('l_knee', 'l_ankle'), ('r_knee', 'r_ankle')
            ]
            
            for conn in connections:
                x_vals = [joints[conn[0]][0], joints[conn[1]][0]]
                y_vals = [joints[conn[0]][1], joints[conn[1]][1]]
                fig1.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines',
                    line=dict(color='#f43f5e', width=3),
                    showlegend=False
                ))
            
            # Draw joints
            fig1.add_trace(go.Scatter(
                x=[j[0] for j in joints.values()],
                y=[j[1] for j in joints.values()],
                mode='markers',
                marker=dict(size=10, color='#10b981'),
                showlegend=False
            ))
            
            fig1.update_layout(
                xaxis=dict(range=[-0.5, 0.5], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[0, 2], showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0.05)',
                height=400
            )
            
            st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.markdown("### Form Analysis & Corrections")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Common Form Errors Detected**")
        
        errors_detected = []
        for error, description in list(FORM_ERRORS.items())[:3]:
            errors_detected.append({
                'Error': error,
                'Description': description,
                'Frequency': f"{np.random.randint(5, 25)}%",
                'Severity': np.random.choice(['Low', 'Medium', 'High'])
            })
        
        st.dataframe(pd.DataFrame(errors_detected), hide_index=True, use_container_width=True)
        
        st.markdown("**Joint Angle Analysis**")
        
        angles = {
            'Joint': ['Knee', 'Hip', 'Ankle', 'Shoulder', 'Elbow'],
            'Current': [92, 88, 78, 145, 95],
            'Optimal': [90, 90, 75, 140, 90],
            'Status': ['✅ Good', '✅ Good', '✅ Good', '⚠️ Adjust', '⚠️ Adjust']
        }
        st.dataframe(pd.DataFrame(angles), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Form Score Breakdown**")
        
        fig2 = go.Figure(data=[go.Bar(
            x=['Depth', 'Alignment', 'Tempo', 'Range of Motion', 'Stability'],
            y=[95, 88, 92, 87, 94],
            marker=dict(
                color=[95, 88, 92, 87, 94],
                colorscale='RdYlGn',
                cmin=70,
                cmax=100
            ),
            text=['95', '88', '92', '87', '94'],
            textposition='auto'
        )])
        fig2.update_layout(yaxis=dict(range=[70, 100]), height=250)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("**Improvement Over Time**")
        
        weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
        form_improvement = [78, 84, 89, 93]
        
        fig3 = go.Figure(data=[go.Scatter(
            x=weeks, y=form_improvement,
            mode='lines+markers',
            line=dict(color='#f43f5e', width=3),
            fill='tozeroy',
            fillcolor='rgba(244, 63, 94, 0.1)'
        )])
        fig3.update_layout(yaxis_title='Form Score (%)', height=200)
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.markdown("### Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Workouts This Month", "24", "+6")
    col2.metric("Avg Form Score", "92.3%", "+8.7%")
    col3.metric("Total Reps", "2,847", "+634")
    col4.metric("Improvement", "+14.3%", "4 weeks")
    
    st.markdown("**Workout History**")
    
    workout_data = {
        'Date': ['Jan 25', 'Jan 23', 'Jan 20', 'Jan 18', 'Jan 15'],
        'Exercise': ['Squat', 'Deadlift', 'Bench Press', 'Squat', 'Deadlift'],
        'Reps': [30, 25, 20, 30, 25],
        'Form Score': ['94.5%', '89.2%', '91.8%', '92.1%', '87.3%'],
        'Corrections': [2, 5, 3, 3, 6]
    }
    st.dataframe(pd.DataFrame(workout_data), hide_index=True, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Muscle Group Coverage**")
        
        muscle_counts = [12, 8, 10, 18, 6, 4]
        
        fig4 = go.Figure(data=[go.Pie(
            labels=['Legs', 'Back', 'Chest', 'Core', 'Arms', 'Shoulders'],
            values=muscle_counts,
            hole=0.4,
            marker=dict(colors=['#f43f5e', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'])
        )])
        fig4.update_layout(height=250)
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        st.markdown("**Monthly Progress**")
        
        months = ['Month 1', 'Month 2', 'Month 3']
        strength = [100, 115, 132]
        form = [78, 86, 92]
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=months, y=strength, mode='lines+markers', name='Strength', line=dict(color='#f43f5e', width=3)))
        fig5.add_trace(go.Scatter(x=months, y=form, mode='lines+markers', name='Form', line=dict(color='#10b981', width=3)))
        fig5.update_layout(yaxis_title='Index (Month 1 = 100)', height=250)
        st.plotly_chart(fig5, use_container_width=True)

with tab4:
    st.markdown("### Computer Vision Technology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**3D Pose Estimation**")
        st.markdown("""
        - ✅ Real-time skeletal tracking (60 FPS)
        - ✅ 25 joint keypoint detection
        - ✅ 3D depth estimation
        - ✅ Sub-centimeter accuracy
        - ✅ Occlusion handling
        - ✅ Multi-person tracking
        """)
        
        st.markdown("**Form Analysis**")
        st.markdown("""
        - ✅ Joint angle measurement
        - ✅ Range of motion tracking
        - ✅ Movement velocity analysis
        - ✅ Balance & stability detection
        - ✅ Tempo consistency checking
        - ✅ Injury risk assessment
        """)
    
    with col2:
        st.markdown("**Hardware**")
        st.markdown("""
        - ✅ Proprietary 3D sensor
        - ✅ Multi-camera depth sensing
        - ✅ Embedded CV processing
        - ✅ C++/OpenCL optimization
        - ✅ Edge AI inference
        - ✅ Low-latency feedback (<50ms)
        """)
        
        st.markdown("**Data Science**")
        st.markdown("""
        - ✅ Human motion dataset (millions of reps)
        - ✅ Exercise technique classification
        - ✅ Personalized form benchmarking
        - ✅ Progression tracking
        - ✅ Injury prediction models
        - ✅ Performance optimization
        """)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #ffe4e6 0%, #fecdd3 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #881337; font-size: 24px; font-weight: 800;">💡 System Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #e11d48; font-weight: 700; margin: 0 0 6px 0;">✓ 3D Pose Tracking</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">60 FPS, 25 keypoints, sub-cm accuracy</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #e11d48; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Feedback</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;"><50ms latency, voice cues</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #e11d48; font-weight: 700; margin: 0 0 6px 0;">✓ Form Correction</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">8-12 checks per exercise</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #e11d48; font-weight: 700; margin: 0 0 6px 0;">✓ Injury Prevention</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Risk detection & alerts</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #f43f5e 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Tempo</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)