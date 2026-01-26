"""
Verne Robotics - AI Models That Teach Robots New Skills
Learn new tasks in hours, not months
Built for Verne Robotics by Anju Nandhakumar
"""

import streamlit as st
from utils.sidebar import render_sidebar
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(page_title="Verne Robotics - Robot Learning", layout="wide")
render_sidebar()

# Initialize session state
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False

# Robot tasks
ROBOT_TASKS = {
    "Pick and Place - Assembly Line": {
        "task_type": "Manipulation",
        "difficulty": "Medium",
        "training_time_traditional": "4-6 weeks",
        "training_time_verne": "8 hours",
        "success_rate": 0.96,
        "demonstrations_needed": 50,
        "description": "Robot learns to pick components from bin and place on assembly line with precise orientation"
    },
    "Quality Inspection - Vision": {
        "task_type": "Vision + Decision",
        "difficulty": "High",
        "training_time_traditional": "8-12 weeks",
        "training_time_verne": "12 hours",
        "success_rate": 0.94,
        "demonstrations_needed": 100,
        "description": "Robot inspects products for defects and sorts into pass/fail bins"
    },
    "Warehouse Navigation - Mobile Robot": {
        "task_type": "Navigation",
        "difficulty": "Low",
        "training_time_traditional": "2-4 weeks",
        "training_time_verne": "4 hours",
        "success_rate": 0.98,
        "demonstrations_needed": 30,
        "description": "Robot navigates warehouse, avoids obstacles, finds optimal paths to destinations"
    }
}

def simulate_training_progress(task_data):
    """Simulate robot learning progress"""
    
    # Generate learning curve
    demonstrations = task_data['demonstrations_needed']
    epochs = np.arange(0, demonstrations, demonstrations//20)
    
    # Success rate improves with demonstrations
    success_curve = 1 - np.exp(-epochs / (demonstrations * 0.3))
    success_curve = success_curve * task_data['success_rate']
    
    # Add some noise for realism
    success_curve += np.random.normal(0, 0.02, len(success_curve))
    success_curve = np.clip(success_curve, 0, 1)
    
    return epochs.tolist(), success_curve.tolist()

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(6, 182, 212, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #06b6d4 0%, #22d3ee 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(6, 182, 212, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🤖</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">Verne Robotics</h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Teach Robots New Skills in Hours</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">AI models that learn from demonstrations, not code</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Imitation Learning</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Reinforcement Learning</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Computer Vision</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">YC Backed</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">Built for <strong style="color: white;">Verne Robotics</strong> by <strong style="color: white;">Anju Nandhakumar</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Value Prop
st.markdown("""
<div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); padding: 25px; border-radius: 15px; border: 2px solid #059669; margin-bottom: 30px;">
    <h3 style="color: #065f46; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">🎯 The Robot Programming Problem</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #ef4444; font-weight: 700; margin: 0 0 8px 0;">❌ Today</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Programming robots takes 4-12 weeks per task. Requires robotics engineers. Hard-coded, brittle. Fails in new scenarios.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #f59e0b; font-weight: 700; margin: 0 0 8px 0;">💰 Cost Impact</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">$200K+ per task (engineering time). 2-3 months deployment. Limited adaptability. New task = start over.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #10b981; font-weight: 700; margin: 0 0 8px 0;">✅ With Verne</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Show robot 50 demos, it learns in 8 hours. No coding. Adapts to variations. New task in hours, not months.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🤖 Train Robot", "📊 Learning Progress", "🧠 How It Works"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Select Task to Teach</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Robot learns from human demonstrations using imitation learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        task_name = st.selectbox("Robot Task", list(ROBOT_TASKS.keys()))
        task = ROBOT_TASKS[task_name]
        
        st.markdown(f"""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb; margin-bottom: 20px;">
            <h3 style="color: #1f2937; margin: 0 0 15px 0; font-size: 18px;">Task Details</h3>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0 0 12px 0;">{task['description']}</p>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                <div style="background: #f9fafb; padding: 12px; border-radius: 8px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Task Type</p>
                    <p style="color: #1f2937; font-size: 16px; font-weight: 700; margin: 3px 0 0 0;">{task['task_type']}</p>
                </div>
                <div style="background: #f9fafb; padding: 12px; border-radius: 8px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Difficulty</p>
                    <p style="color: #f59e0b; font-size: 16px; font-weight: 700; margin: 3px 0 0 0;">{task['difficulty']}</p>
                </div>
                <div style="background: #f9fafb; padding: 12px; border-radius: 8px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Demos Needed</p>
                    <p style="color: #3b82f6; font-size: 16px; font-weight: 700; margin: 3px 0 0 0;">{task['demonstrations_needed']}</p>
                </div>
                <div style="background: #f9fafb; padding: 12px; border-radius: 8px;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Training Time</p>
                    <p style="color: #10b981; font-size: 16px; font-weight: 700; margin: 3px 0 0 0;">{task['training_time_verne']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            st.session_state.training_complete = True
            st.session_state.current_task = task_name
            epochs, success = simulate_training_progress(task)
            st.session_state.learning_curve = (epochs, success)
    
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #92400e; margin: 0 0 12px 0; font-size: 16px;">💡 Imitation Learning Process</h4>
            <ol style="color: #78350f; font-size: 13px; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li><strong>Human demonstrates</strong> task 50-100 times via teleoperation</li>
                <li><strong>AI observes</strong> camera feed, joint positions, end-effector state</li>
                <li><strong>Model learns</strong> state → action mapping using neural network</li>
                <li><strong>Robot practices</strong> in simulation, refines policy via RL</li>
                <li><strong>Deploy to hardware</strong> - robot executes autonomously</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.training_complete:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); padding: 20px; border-radius: 12px; border: 2px solid #059669; margin-top: 20px;">
                <h4 style="color: #065f46; margin: 0 0 12px 0; font-size: 16px;">✅ Training Complete!</h4>
                <div style="background: white; padding: 15px; border-radius: 10px; text-align: center;">
                    <p style="color: #6b7280; font-size: 13px; margin: 0;">Success Rate</p>
                    <p style="color: #059669; font-size: 42px; font-weight: 900; margin: 8px 0;">{task['success_rate']:.0%}</p>
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Ready for deployment</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Learning Performance</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">How quickly robots learn new tasks with Verne AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.training_complete:
        epochs, success = st.session_state.learning_curve
        task = ROBOT_TASKS[st.session_state.current_task]
        
        # Learning curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=success,
            mode='lines+markers',
            line=dict(color='#059669', width=3),
            fill='tonexty',
            fillcolor='rgba(5, 150, 105, 0.1)',
            name='Success Rate'
        ))
        fig.update_layout(
            title=f"Learning Curve: {st.session_state.current_task}",
            xaxis_title="Demonstrations",
            yaxis_title="Success Rate",
            yaxis_range=[0, 1],
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Time comparison
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"""
            <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
                <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">⏱️ Training Time Comparison</h3>
                <div style="background: #fef2f2; padding: 20px; border-radius: 10px; margin-bottom: 15px; text-align: center;">
                    <p style="color: #6b7280; font-size: 13px; margin: 0;">Traditional Programming</p>
                    <p style="color: #ef4444; font-size: 36px; font-weight: 900; margin: 8px 0;">{task['training_time_traditional']}</p>
                </div>
                <div style="background: #ecfdf5; padding: 20px; border-radius: 10px; text-align: center;">
                    <p style="color: #6b7280; font-size: 13px; margin: 0;">With Verne AI</p>
                    <p style="color: #059669; font-size: 36px; font-weight: 900; margin: 8px 0;">{task['training_time_verne']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            speedup = 168 / 8 if "8 hours" in task['training_time_verne'] else 672 / 12 if "12 hours" in task['training_time_verne'] else 672 / 4
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); padding: 25px; border-radius: 15px; border: 2px solid #059669;">
                <h3 style="color: #065f46; margin: 0 0 20px 0; font-size: 20px;">🚀 Speedup</h3>
                <div style="background: white; padding: 25px; border-radius: 10px; text-align: center;">
                    <p style="color: #059669; font-size: 56px; font-weight: 900; margin: 0;">{speedup:.0f}x</p>
                    <p style="color: #6b7280; font-size: 14px; margin: 8px 0;">Faster than traditional</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 10px; margin-top: 15px; text-align: center;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Final Success Rate</p>
                    <p style="color: #3b82f6; font-size: 28px; font-weight: 900; margin: 5px 0;">{task['success_rate']:.0%}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Imitation + RL Pipeline</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">How Verne AI teaches robots from demonstrations</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">🎓 Learning Pipeline</h3>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #06b6d4; margin-bottom: 12px;">
                <h4 style="color: #0891b2; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">1. Data Collection</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Human demonstrates via teleoperation, collect state-action pairs</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #8b5cf6; margin-bottom: 12px;">
                <h4 style="color: #7c3aed; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">2. Behavioral Cloning</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Neural network learns to map observations to actions</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #10b981; margin-bottom: 12px;">
                <h4 style="color: #059669; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">3. RL Fine-Tuning</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Robot practices in sim, improves via reinforcement learning</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6;">
                <h4 style="color: #2563eb; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">4. Sim-to-Real Transfer</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Deploy to physical robot, continue learning from experience</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">🔧 Technical Approach</h3>
            <div style="background: #ecfdf5; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #047857; font-weight: 700; font-size: 14px; margin: 0;">✓ Vision System</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">RGB-D cameras, object detection, scene understanding</p>
            </div>
            <div style="background: #eff6ff; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #1e40af; font-weight: 700; font-size: 14px; margin: 0;">✓ Policy Network</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Transformer architecture for state → action mapping</p>
            </div>
            <div style="background: #fef3c7; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #92400e; font-weight: 700; font-size: 14px; margin: 0;">✓ RL Algorithm</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">PPO/SAC for policy refinement in simulation</p>
            </div>
            <div style="background: #f3e8ff; padding: 12px 15px; border-radius: 8px;">
                <p style="color: #6b21a8; font-weight: 700; font-size: 14px; margin: 0;">✓ Sim-to-Real</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Domain randomization, reality gap bridging</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #0891b2; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Verne Robotics</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 21x Faster</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">8 hours vs 4-6 weeks traditional. New tasks deployed in hours. No robotics engineer bottleneck. Factories adapt instantly.</p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 $200K Saved</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">Per task. No engineering time, no months of development. Show demos, robot learns. Scale to hundreds of tasks.</p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🤖 96% Success</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">Production-ready from demos alone. Adapts to variations. Handles edge cases. Continues learning on the job.</p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Manufacturing Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">21x faster deployment:</strong> 8 hours vs 4-6 weeks</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">$200K saved per task:</strong> No engineering time required</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">96% success rate:</strong> Production-ready from demonstrations</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Continuous learning:</strong> Robot improves with experience</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Stack</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Imitation Learning</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Behavioral cloning from demonstrations</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Reinforcement Learning</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">PPO/SAC for policy optimization</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Computer Vision</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">RGB-D perception, object detection</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Simulation</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Isaac Gym for rapid iteration</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(6, 182, 212, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">Built for <strong style="color: white;">Verne Robotics</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong></p>
        <div style="margin: 25px 0; padding: 22px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.2);">
            <p style="margin: 8px 0; font-size: 16px;">📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: white; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a></p>
            <p style="margin: 8px 0; font-size: 16px;">💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">LinkedIn</a> | 💻 <a href="https://github.com/Av1352" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">GitHub</a> | 🌐 <a href="https://vxanju.com" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">Portfolio</a></p>
        </div>
        <p style="font-size: 15px; margin: 18px 0; font-weight: 700;"><strong style="color: white;">Tech Stack:</strong> Imitation Learning • Reinforcement Learning • Computer Vision • Robotics</p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">Demo showcasing AI-powered robot learning from human demonstrations.<br>Imitation learning • Reinforcement learning • Sim-to-real transfer • Rapid task deployment</p>
    </div>
    """, unsafe_allow_html=True)