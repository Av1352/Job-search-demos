"""
hud - Platform for Building RL Environments and Evals
Create custom RL environments and evaluations for agent training
Built for hud by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="hud", page_icon="🎮", layout="wide")

# RL Environments
ENVIRONMENTS = {
    'CartPole-v1': {'difficulty': 'Easy', 'episodes': 500, 'max_steps': 500, 'solved_threshold': 475},
    'LunarLander-v2': {'difficulty': 'Medium', 'episodes': 1000, 'max_steps': 1000, 'solved_threshold': 200},
    'BipedalWalker-v3': {'difficulty': 'Hard', 'episodes': 2000, 'max_steps': 1600, 'solved_threshold': 300},
    'Custom-GridWorld': {'difficulty': 'Easy', 'episodes': 300, 'max_steps': 100, 'solved_threshold': 90},
    'Custom-TradingEnv': {'difficulty': 'Hard', 'episodes': 1500, 'max_steps': 1000, 'solved_threshold': 0}
}

# RL Algorithms
RL_ALGORITHMS = ['DQN', 'PPO', 'A2C', 'SAC', 'TD3']

# Generate training data
def generate_training_curve(env_name, algorithm, episodes):
    np.random.seed(42)
    solved_threshold = ENVIRONMENTS[env_name]['solved_threshold']
    
    # Simulate learning curve
    x = np.arange(episodes)
    
    # Different learning patterns by algorithm
    if algorithm == 'PPO':
        base = np.log(x + 1) * 100 + np.random.normal(0, 20, episodes)
    elif algorithm == 'DQN':
        base = np.sqrt(x) * 15 + np.random.normal(0, 25, episodes)
    elif algorithm == 'SAC':
        base = np.log(x + 1) * 120 + np.random.normal(0, 15, episodes)
    else:
        base = x * 0.3 + np.random.normal(0, 30, episodes)
    
    rewards = np.clip(base, -200, 600)
    return x, rewards

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #f59e0b 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🎮</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">hud</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Platform for Building RL Environments and Evals</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Create • Train • Evaluate • Deploy RL agents</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎮 Environment Builder", "📊 Training Dashboard", "🎯 Evaluation Suite", "💡 Platform Features"])

with tab1:
    st.markdown("### Custom RL Environment Creation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Environment Configuration**")
        
        env_type = st.selectbox(
            "Environment Type",
            ["Pre-built (OpenAI Gym)", "Custom GridWorld", "Custom Continuous", "Custom Multi-Agent"]
        )
        
        if env_type == "Pre-built (OpenAI Gym)":
            env_name = st.selectbox("Select Environment", list(ENVIRONMENTS.keys())[:3])
        else:
            env_name = st.text_input("Environment Name", "MyCustomEnv-v1")
        
        st.markdown("**State Space**")
        state_dim = st.slider("State Dimensions", 2, 100, 4)
        state_type = st.radio("State Type", ["Discrete", "Continuous", "Mixed"])
        
        st.markdown("**Action Space**")
        action_dim = st.slider("Action Dimensions", 2, 20, 2)
        action_type = st.radio("Action Type", ["Discrete", "Continuous"])
        
        st.markdown("**Reward Design**")
        reward_type = st.selectbox("Reward Function", ["Sparse", "Dense", "Shaped", "Custom"])
        
        create_btn = st.button("🎮 Create Environment", type="primary", use_container_width=True)
    
    with col2:
        if create_btn:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f59e0b 0%, #73BA9B 100%); padding: 30px; border-radius: 16px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 28px; font-weight: 900;">Environment Created! 🎉</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Environment</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{env_name}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Type</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{env_type.split()[0]}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">State Space</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{state_dim}D {state_type}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Action Space</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{action_dim}D {action_type}</p>
                    </div>
                </div>
                <div style="margin-top: 20px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Reward Function</p>
                    <p style="font-size: 16px; color: white; margin: 0;">{reward_type} reward structure with automatic shaping</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Environment Code (Generated)**")
            st.code(f"""
import gym
from gym import spaces
import numpy as np

class {env_name.replace('-', '_')}(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=({state_dim},), dtype=np.float32
        )
        self.action_space = spaces.{'Discrete' if action_type == 'Discrete' else 'Box'}(
            {'n=' + str(action_dim) if action_type == 'Discrete' else f'low=-1, high=1, shape=({action_dim},)'}
        )
        
    def reset(self):
        self.state = np.zeros({state_dim})
        return self.state
        
    def step(self, action):
        # Update state
        self.state += np.random.randn({state_dim}) * 0.1
        
        # Calculate reward ({reward_type.lower()})
        reward = self._compute_reward(self.state, action)
        
        # Check termination
        done = self._is_done(self.state)
        
        return self.state, reward, done, {{}}
        
    def _compute_reward(self, state, action):
        # {reward_type} reward implementation
        return -np.sum(np.square(state))
        
    def _is_done(self, state):
        return np.linalg.norm(state) > 10
""", language="python")

with tab2:
    st.markdown("### RL Agent Training Dashboard")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Training Configuration**")
        train_env = st.selectbox("Environment", list(ENVIRONMENTS.keys()))
        train_algo = st.selectbox("Algorithm", RL_ALGORITHMS)
        train_episodes = st.slider("Training Episodes", 100, 2000, ENVIRONMENTS[train_env]['episodes'])
        
        train_btn = st.button("🚀 Start Training", type="primary", use_container_width=True)
    
    with col2:
        if train_btn:
            # Generate training data
            x, rewards = generate_training_curve(train_env, train_algo, train_episodes)
            
            # Moving average
            window = 50
            ma_rewards = pd.Series(rewards).rolling(window=window).mean()
            
            # Training curve
            st.markdown("**Training Progress**")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=x, y=rewards,
                mode='lines',
                name='Episode Reward',
                line=dict(color='rgba(59, 130, 246, 0.3)', width=1)
            ))
            fig1.add_trace(go.Scatter(
                x=x, y=ma_rewards,
                mode='lines',
                name=f'{window}-Episode MA',
                line=dict(color='#10b981', width=3)
            ))
            if ENVIRONMENTS[train_env]['solved_threshold'] > 0:
                fig1.add_hline(
                    y=ENVIRONMENTS[train_env]['solved_threshold'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Solved Threshold"
                )
            fig1.update_layout(
                xaxis_title='Episode',
                yaxis_title='Reward',
                height=300
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Training metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Reward (Last 100)", f"{rewards[-100:].mean():.1f}", f"+{abs(rewards[-100:].mean() - rewards[-200:-100].mean()):.1f}")
            col2.metric("Best Episode", f"{rewards.max():.1f}", "🏆")
            col3.metric("Episodes Trained", train_episodes, "100%")
            col4.metric("Wall Time", "45.3 min", "⏱️")

with tab3:
    st.markdown("### Evaluation Suite")
    
    st.markdown("**Standard Evaluation Metrics**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Evaluation results
        eval_data = {
            'Metric': [
                'Mean Episode Reward',
                'Std Episode Reward',
                'Success Rate',
                'Episode Length (Mean)',
                'Episode Length (Std)',
                'Sample Efficiency'
            ],
            'Value': [
                f"{np.random.uniform(350, 450):.1f}",
                f"{np.random.uniform(30, 60):.1f}",
                f"{np.random.uniform(85, 98):.1f}%",
                f"{np.random.uniform(180, 250):.1f}",
                f"{np.random.uniform(20, 40):.1f}",
                f"{np.random.uniform(0.7, 0.95):.2f}"
            ],
            'Status': ['✅ Good', '✅ Good', '✅ Excellent', '✅ Good', '✅ Good', '✅ High']
        }
        st.dataframe(pd.DataFrame(eval_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Robustness Tests**")
        robustness = {
            'Test': ['Noisy Observations', 'Delayed Rewards', 'Partial Observability', 'Domain Randomization'],
            'Performance Drop': ['8.2%', '12.5%', '15.3%', '6.8%'],
            'Status': ['✅ Robust', '✅ Robust', '⚠️ Moderate', '✅ Robust']
        }
        st.dataframe(pd.DataFrame(robustness), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Performance Distribution**")
        
        # Generate reward distribution
        rewards_dist = np.random.normal(400, 50, 1000)
        
        fig2 = go.Figure(data=[go.Histogram(
            x=rewards_dist,
            nbinsx=40,
            marker=dict(color='#f59e0b')
        )])
        fig2.update_layout(
            xaxis_title='Episode Reward',
            yaxis_title='Frequency',
            height=200
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("**Sample Trajectories**")
        
        # Sample trajectory visualization
        fig3 = go.Figure()
        for i in range(5):
            traj_x = np.cumsum(np.random.randn(50) * 0.5)
            traj_y = np.cumsum(np.random.randn(50) * 0.5)
            fig3.add_trace(go.Scatter(
                x=traj_x, y=traj_y,
                mode='lines+markers',
                name=f'Trajectory {i+1}',
                line=dict(width=2)
            ))
        fig3.update_layout(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            height=200,
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.markdown("### Platform Features & Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Environment Creation**")
        st.markdown("""
        - ✅ Pre-built OpenAI Gym environments
        - ✅ Custom discrete/continuous spaces
        - ✅ Multi-agent environments
        - ✅ Reward shaping automation
        - ✅ Automatic observation normalization
        - ✅ Environment wrappers & modifiers
        """)
        
        st.markdown("**RL Algorithms**")
        st.markdown("""
        - ✅ DQN (Deep Q-Network)
        - ✅ PPO (Proximal Policy Optimization)
        - ✅ A2C (Advantage Actor-Critic)
        - ✅ SAC (Soft Actor-Critic)
        - ✅ TD3 (Twin Delayed DDPG)
        - ✅ Custom algorithm support
        """)
    
    with col2:
        st.markdown("**Evaluation Suite**")
        st.markdown("""
        - ✅ Standard performance metrics
        - ✅ Success rate tracking
        - ✅ Robustness testing
        - ✅ Sample efficiency analysis
        - ✅ Trajectory visualization
        - ✅ Distribution analysis
        """)
        
        st.markdown("**Integration & Deployment**")
        st.markdown("""
        - ✅ Python API
        - ✅ REST API endpoints
        - ✅ Model export (ONNX, TorchScript)
        - ✅ Cloud deployment
        - ✅ Real-time monitoring
        - ✅ Version control
        """)
    
    # Supported environments table
    st.markdown("**Pre-built Environments**")
    env_df = pd.DataFrame([
        {'Environment': env, 'Difficulty': ENVIRONMENTS[env]['difficulty'], 
         'Episodes to Solve': ENVIRONMENTS[env]['episodes'], 
         'Max Steps': ENVIRONMENTS[env]['max_steps']}
        for env in ENVIRONMENTS.keys()
    ])
    st.dataframe(env_df, hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #78350f; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #d97706; font-weight: 700; margin: 0 0 6px 0;">✓ Custom Environments</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Build any RL environment in minutes</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #d97706; font-weight: 700; margin: 0 0 6px 0;">✓ 5 RL Algorithms</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">DQN, PPO, A2C, SAC, TD3</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #d97706; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Training</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Live training curves & metrics</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #d97706; font-weight: 700; margin: 0 0 6px 0;">✓ Evaluation Suite</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Comprehensive robustness testing</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #f59e0b 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for hud</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)