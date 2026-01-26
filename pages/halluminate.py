"""
Halluminate - Data and RL Environments for Knowledge Work
Train AI agents to automate complex workflows
Built for Halluminate by Anju Nandhakumar
"""

import streamlit as st
from utils.sidebar import render_sidebar
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(page_title="Halluminate - RL Environments", layout="wide")
render_sidebar()

# Initialize session state
if 'training_started' not in st.session_state:
    st.session_state.training_started = False

# RL Environments for knowledge work
ENVIRONMENTS = {
    "Email Management Agent": {
        "task": "Prioritize, categorize, and respond to emails",
        "state_space": "Email content, sender, subject, thread history",
        "action_space": "Archive, Reply, Forward, Schedule, Flag",
        "reward_function": "User satisfaction + time saved + accuracy",
        "training_episodes": 10000,
        "final_performance": 0.92,
        "time_savings": "2.5 hours/day per employee"
    },
    "Customer Support Agent": {
        "task": "Resolve customer inquiries and escalate complex issues",
        "state_space": "Ticket content, customer history, product context",
        "action_space": "Provide solution, Ask clarification, Escalate, Close ticket",
        "reward_function": "CSAT score + resolution time + escalation accuracy",
        "training_episodes": 15000,
        "final_performance": 0.88,
        "time_savings": "40% faster resolution"
    },
    "Code Review Agent": {
        "task": "Review pull requests, suggest improvements, approve/reject",
        "state_space": "Code diff, test coverage, complexity metrics, commit history",
        "action_space": "Approve, Request changes, Comment, Suggest fix",
        "reward_function": "Bug catch rate + false positive rate + developer satisfaction",
        "training_episodes": 20000,
        "final_performance": 0.85,
        "time_savings": "15 hours/week per team"
    }
}

def generate_training_curve(episodes, final_perf):
    """Generate RL training curve"""
    x = np.linspace(0, episodes, 100)
    
    # Sigmoid-like learning curve
    y = final_perf / (1 + np.exp(-0.00015 * (x - episodes/2)))
    
    # Add exploration noise in early stages
    noise = np.random.normal(0, 0.05, len(x))
    noise = noise * np.exp(-x / (episodes * 0.3))  # Decay noise
    y = y + noise
    y = np.clip(y, 0, 1)
    
    return x.tolist(), y.tolist()

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(124, 58, 237, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #8b5cf6 0%, #c4b5fd 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(139, 92, 246, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🎮</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">Halluminate</h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">RL Environments for Knowledge Work</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Train AI agents to automate complex workflows</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Reinforcement Learning</span>
            <span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);">Simulation</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Agent Training</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">YC Backed</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">Built for <strong style="color: white;">Halluminate</strong> by <strong style="color: white;">Anju Nandhakumar</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Value Prop
st.markdown("""
<div style="background: linear-gradient(135deg, #f3e8ff, #e9d5ff); padding: 25px; border-radius: 15px; border: 2px solid #8b5cf6; margin-bottom: 30px;">
    <h3 style="color: #5b21b6; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">🎯 The Agent Training Problem</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #ef4444; font-weight: 700; margin: 0 0 8px 0;">❌ Today</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Training AI agents is slow and expensive. Need real user interactions. Mistakes cost money. Limited scalability.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #f59e0b; font-weight: 700; margin: 0 0 8px 0;">💰 Cost Impact</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Training on real users = bad UX. Mistakes = lost revenue. Slow iteration = months to production. Can't test edge cases.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #10b981; font-weight: 700; margin: 0 0 8px 0;">✅ With Halluminate</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Train in simulation with synthetic data. 10K episodes in hours. Zero user impact. Test infinite scenarios safely.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎮 Train Agent", "📈 Learning Curves", "🧠 RL Architecture"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Select Knowledge Work Task</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Agent will learn optimal policy through RL in simulated environment</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        task_name = st.selectbox("Agent Task", list(ENVIRONMENTS.keys()))
        env = ENVIRONMENTS[task_name]
        
        st.markdown(f"""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb; margin-bottom: 20px;">
            <h3 style="color: #1f2937; margin: 0 0 15px 0; font-size: 18px;">Environment Spec</h3>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0 0 15px 0;"><strong>Task:</strong> {env['task']}</p>
            <div style="background: #f9fafb; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #6b7280; font-size: 12px; margin: 0; font-weight: 600;">State Space</p>
                <p style="color: #1f2937; font-size: 13px; margin: 3px 0 0 0;">{env['state_space']}</p>
            </div>
            <div style="background: #f9fafb; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #6b7280; font-size: 12px; margin: 0; font-weight: 600;">Action Space</p>
                <p style="color: #1f2937; font-size: 13px; margin: 3px 0 0 0;">{env['action_space']}</p>
            </div>
            <div style="background: #f9fafb; padding: 12px; border-radius: 8px;">
                <p style="color: #6b7280; font-size: 12px; margin: 0; font-weight: 600;">Reward Function</p>
                <p style="color: #059669; font-size: 13px; margin: 3px 0 0 0;">{env['reward_function']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start RL Training", type="primary", use_container_width=True):
            st.session_state.training_started = True
            st.session_state.current_env = task_name
            x, y = generate_training_curve(env['training_episodes'], env['final_performance'])
            st.session_state.learning_data = (x, y)
    
    with col2:
        st.markdown("""
        <div style="background: #ecfdf5; padding: 20px; border-radius: 12px; border-left: 4px solid #10b981;">
            <h4 style="color: #065f46; margin: 0 0 12px 0; font-size: 16px;">🎮 How RL Training Works</h4>
            <ol style="color: #047857; font-size: 13px; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li><strong>Simulation:</strong> Create realistic environment for task</li>
                <li><strong>Agent explores:</strong> Tries different actions, learns from rewards</li>
                <li><strong>Policy improves:</strong> Neural network learns optimal decisions</li>
                <li><strong>Iterate rapidly:</strong> 10K episodes in hours vs months real-world</li>
                <li><strong>Deploy:</strong> Transfer learned policy to production</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.training_started:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); padding: 20px; border-radius: 12px; border: 2px solid #059669; margin-top: 20px;">
                <h4 style="color: #065f46; margin: 0 0 12px 0; font-size: 16px;">✅ Training Complete!</h4>
                <div style="background: white; padding: 15px; border-radius: 10px; text-align: center;">
                    <p style="color: #6b7280; font-size: 13px; margin: 0;">Final Performance</p>
                    <p style="color: #059669; font-size: 42px; font-weight: 900; margin: 8px 0;">{env['final_performance']:.0%}</p>
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">{env['training_episodes']:,} episodes</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Agent Learning Progress</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">How agents improve through reinforcement learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.training_started:
        x, y = st.session_state.learning_data
        env = ENVIRONMENTS[st.session_state.current_env]
        
        # Learning curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='#8b5cf6', width=3),
            fill='tonexty',
            fillcolor='rgba(139, 92, 246, 0.1)',
            name='Performance'
        ))
        fig.add_hline(y=env['final_performance'], line_dash="dash", 
                     line_color="#059669", annotation_text=f"Target: {env['final_performance']:.0%}")
        fig.update_layout(
            title=f"RL Training: {st.session_state.current_env}",
            xaxis_title="Episodes",
            yaxis_title="Success Rate",
            yaxis_range=[0, 1],
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">📊 Training Metrics</h2>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Episodes</p>
                    <p style="font-size: 40px; color: white; font-weight: 900; margin: 8px 0;">""" + f"{env['training_episodes']:,}" + """</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Final Performance</p>
                    <p style="font-size: 40px; color: #86efac; font-weight: 900; margin: 8px 0;">""" + f"{env['final_performance']:.0%}" + """</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Training Time</p>
                    <p style="font-size: 40px; color: white; font-weight: 900; margin: 8px 0;">6h</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Time Saved</p>
                    <p style="font-size: 36px; color: #fbbf24; font-weight: 900; margin: 8px 0;">""" + env['time_savings'] + """</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">RL Architecture</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">How Halluminate builds RL environments for knowledge work automation</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">🏗️ Environment Design</h3>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #8b5cf6; margin-bottom: 12px;">
                <h4 style="color: #7c3aed; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">1. Define State Space</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">What information does agent observe? (emails, tickets, code, etc.)</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6; margin-bottom: 12px;">
                <h4 style="color: #2563eb; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">2. Define Action Space</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">What can agent do? (reply, escalate, approve, etc.)</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #10b981; margin-bottom: 12px;">
                <h4 style="color: #059669; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">3. Design Reward Function</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">How to measure success? (accuracy, speed, user satisfaction)</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #f59e0b;">
                <h4 style="color: #ea580c; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">4. Generate Synthetic Data</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Create realistic scenarios for training (LLM-generated)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">⚡ RL Algorithms</h3>
            <div style="background: #ecfdf5; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #047857; font-weight: 700; font-size: 14px; margin: 0;">✓ PPO (Proximal Policy Optimization)</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Stable, sample-efficient, good for discrete actions</p>
            </div>
            <div style="background: #eff6ff; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #1e40af; font-weight: 700; font-size: 14px; margin: 0;">✓ SAC (Soft Actor-Critic)</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Off-policy, continuous actions, maximum entropy</p>
            </div>
            <div style="background: #fef3c7; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #92400e; font-weight: 700; font-size: 14px; margin: 0;">✓ DQN (Deep Q-Network)</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Experience replay, target networks, discrete actions</p>
            </div>
            <div style="background: #f3e8ff; padding: 12px 15px; border-radius: 8px;">
                <p style="color: #6b21a8; font-weight: 700; font-size: 14px; margin: 0;">✓ RLHF (RL from Human Feedback)</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Learn from user preferences, not explicit rewards</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #8b5cf6; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Halluminate</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🚀 Safe Training</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">Train agents in simulation with synthetic data. Zero user impact. Test edge cases safely. 10K episodes in hours.</p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ Rapid Iteration</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">Try different reward functions, architectures, hyperparameters instantly. Days to production vs months.</p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📊 92% Performance</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">Agents reach human-level performance on knowledge work tasks. Email management: 92% accuracy, 2.5 hrs/day saved.</p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Knowledge Work Automation</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">10K episodes in 6 hours:</strong> vs months of real-world learning</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">92% task performance:</strong> Human-level on email, support, code review</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Zero production risk:</strong> Train in simulation, deploy when ready</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Continuous improvement:</strong> Agent keeps learning from real interactions</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Stack</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ RL Algorithms</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">PPO, SAC, DQN, RLHF</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Simulation</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Synthetic data generation, realistic scenarios</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Policy Networks</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Transformers for sequential decision making</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Evaluation</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">A/B testing, success metrics, user feedback</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(124, 58, 237, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">Built for <strong style="color: white;">Halluminate</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong></p>
        <div style="margin: 25px 0; padding: 22px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.2);">
            <p style="margin: 8px 0; font-size: 16px;">📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: white; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a></p>
            <p style="margin: 8px 0; font-size: 16px;">💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">LinkedIn</a> | 💻 <a href="https://github.com/Av1352" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">GitHub</a> | 🌐 <a href="https://vxanju.com" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">Portfolio</a></p>
        </div>
        <p style="font-size: 15px; margin: 18px 0; font-weight: 700;"><strong style="color: white;">Tech Stack:</strong> Reinforcement Learning • PPO/SAC • Simulation • Agent Training</p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">Demo showcasing RL environments for training AI agents on knowledge work automation.<br>Reinforcement learning • Simulation environments • Policy optimization • Synthetic data generation</p>
    </div>
    """, unsafe_allow_html=True)