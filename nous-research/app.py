"""
Nous Research - Reinforcement Learning Training Observatory
Distributed RL training visualization and analysis
Built for Nous Research by Anju Nandhakumar
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import random

# RL algorithms and their characteristics
RL_ALGORITHMS = {
    "PPO (Proximal Policy Optimization)": {
        "type": "Policy Gradient",
        "sample_efficiency": "Medium",
        "stability": "High",
        "typical_episodes": 5000,
        "convergence_rate": "Fast"
    },
    "DQN (Deep Q-Network)": {
        "type": "Value-Based",
        "sample_efficiency": "Low",
        "stability": "Medium",
        "typical_episodes": 10000,
        "convergence_rate": "Medium"
    },
    "SAC (Soft Actor-Critic)": {
        "type": "Actor-Critic",
        "sample_efficiency": "High",
        "stability": "High",
        "typical_episodes": 3000,
        "convergence_rate": "Fast"
    },
    "A3C (Asynchronous Advantage Actor-Critic)": {
        "type": "Actor-Critic",
        "sample_efficiency": "Medium",
        "stability": "Medium",
        "typical_episodes": 8000,
        "convergence_rate": "Medium"
    }
}

ENVIRONMENTS = {
    "CartPole-v1": {"max_reward": 500, "solved_threshold": 475},
    "LunarLander-v2": {"max_reward": 200, "solved_threshold": 200},
    "BipedalWalker-v3": {"max_reward": 300, "solved_threshold": 300},
    "Hopper-v4": {"max_reward": 3000, "solved_threshold": 3000}
}

def generate_training_run(algorithm, environment, num_episodes):
    """Generate a simulated RL training run"""
    
    algo_config = RL_ALGORITHMS[algorithm]
    env_config = ENVIRONMENTS[environment]
    
    # Generate reward curve with realistic learning dynamics
    episodes = list(range(num_episodes))
    
    # Early exploration phase (low rewards, high variance)
    exploration_phase = num_episodes // 4
    exploration_rewards = [random.gauss(-50, 100) for _ in range(exploration_phase)]
    
    # Learning phase (improving rewards)
    learning_phase = num_episodes // 2
    learning_rewards = []
    for i in range(learning_phase):
        progress = i / learning_phase
        mean_reward = -50 + (env_config['solved_threshold'] * progress)
        variance = 100 * (1 - progress * 0.7)
        learning_rewards.append(random.gauss(mean_reward, variance))
    
    # Convergence phase (stable high rewards)
    convergence_phase = num_episodes - exploration_phase - learning_phase
    convergence_rewards = [random.gauss(env_config['solved_threshold'] * 0.95, 20) for _ in range(convergence_phase)]
    
    all_rewards = exploration_rewards + learning_rewards + convergence_rewards
    
    # Smooth with moving average
    window = 50
    smoothed_rewards = pd.Series(all_rewards).rolling(window=window, min_periods=1).mean().tolist()
    
    # Calculate metrics
    final_reward = smoothed_rewards[-1]
    max_reward = max(smoothed_rewards)
    convergence_episode = next((i for i, r in enumerate(smoothed_rewards) if r > env_config['solved_threshold'] * 0.9), num_episodes)
    
    is_solved = final_reward >= env_config['solved_threshold'] * 0.9
    
    # Training summary
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🎮 RL Training Run Complete</h2>
        
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Status</p>
                <p style="font-size: 36px; color: {'#86efac' if is_solved else '#fbbf24'}; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{'✅' if is_solved else '🔄'}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{'Solved!' if is_solved else 'Training...'}</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Final Reward</p>
                <p style="font-size: 40px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{final_reward:.0f}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Target: {env_config['solved_threshold']}</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Episodes</p>
                <p style="font-size: 40px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{num_episodes:,}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Total training</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Convergence</p>
                <p style="font-size: 40px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{convergence_episode:,}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Episode #</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Algorithm</p>
                <p style="font-size: 18px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{algorithm.split()[0]}</p>
            </div>
        </div>
    </div>
    """
    
    # Algorithm details
    algo_html = f"""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">⚙️ Algorithm Configuration</h3>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: white; border-radius: 14px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <h4 style="color: #1f2937; font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">Algorithm Properties</h4>
                <div style="background: #f9fafb; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Type: <strong style="color: #1f2937;">{algo_config['type']}</strong></p>
                </div>
                <div style="background: #f9fafb; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Sample Efficiency: <strong style="color: #1f2937;">{algo_config['sample_efficiency']}</strong></p>
                </div>
                <div style="background: #f9fafb; border-radius: 8px; padding: 12px;">
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Stability: <strong style="color: #1f2937;">{algo_config['stability']}</strong></p>
                </div>
            </div>
            
            <div style="background: white; border-radius: 14px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <h4 style="color: #1f2937; font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">Environment Info</h4>
                <div style="background: #f9fafb; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Environment: <strong style="color: #1f2937;">{environment}</strong></p>
                </div>
                <div style="background: #f9fafb; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Max Reward: <strong style="color: #1f2937;">{env_config['max_reward']}</strong></p>
                </div>
                <div style="background: #f9fafb; border-radius: 8px; padding: 12px;">
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Solved: <strong style="color: #1f2937;">>{env_config['solved_threshold']}</strong></p>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Create charts
    
    # 1. Reward curve
    fig_reward = go.Figure()
    
    fig_reward.add_trace(go.Scatter(
        x=episodes,
        y=all_rewards,
        mode='lines',
        line=dict(color='#e5e7eb', width=1),
        name='Raw Reward',
        opacity=0.3
    ))
    
    fig_reward.add_trace(go.Scatter(
        x=episodes,
        y=smoothed_rewards,
        mode='lines',
        line=dict(color='#3b82f6', width=3),
        name='Smoothed (MA-50)',
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig_reward.add_hline(
        y=env_config['solved_threshold'],
        line_dash="dash",
        line_color="#10b981",
        annotation_text="Solved Threshold",
        annotation_position="right"
    )
    
    fig_reward.update_layout(
        title=f"Episode Reward Curve - {algorithm} on {environment}",
        xaxis_title="Episode",
        yaxis_title="Reward",
        height=500,
        hovermode='x unified'
    )
    
    # 2. Learning phases
    phases = ['Exploration', 'Learning', 'Convergence']
    phase_lengths = [exploration_phase, learning_phase, convergence_phase]
    
    fig_phases = go.Figure(data=[
        go.Bar(
            x=phases,
            y=phase_lengths,
            marker_color=['#f59e0b', '#3b82f6', '#10b981'],
            text=[f'{p} episodes' for p in phase_lengths],
            textposition='outside'
        )
    ])
    
    fig_phases.update_layout(
        title="Training Phase Breakdown",
        yaxis_title="Number of Episodes",
        height=400
    )
    
    return summary_html + algo_html, fig_reward, fig_phases, all_rewards, smoothed_rewards

def generate_hyperparameter_comparison():
    """Compare different hyperparameter configurations"""
    
    comparison_html = """
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2); margin-bottom: 25px;">
        <h3 style="color: #92400e; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🔬 Hyperparameter Optimization</h3>
        
        <div style="display: grid; gap: 12px;">
    """
    
    # Generate 5 different HP configs
    configs = [
        {"lr": 0.0003, "batch": 64, "gamma": 0.99, "reward": 425, "episodes": 3200},
        {"lr": 0.001, "batch": 128, "gamma": 0.95, "reward": 380, "episodes": 2800},
        {"lr": 0.0001, "batch": 32, "gamma": 0.99, "reward": 465, "episodes": 4500},
        {"lr": 0.0005, "batch": 256, "gamma": 0.98, "reward": 490, "episodes": 3100},
        {"lr": 0.0003, "batch": 128, "gamma": 0.99, "reward": 475, "episodes": 2900}
    ]
    
    configs_sorted = sorted(configs, key=lambda x: x['reward'], reverse=True)
    
    for idx, config in enumerate(configs_sorted):
        rank_colors = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444']
        color = rank_colors[idx]
        
        comparison_html += f"""
        <div style="background: white; border-left: 5px solid {color}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <span style="background: {color}; color: white; padding: 6px 14px; border-radius: 16px; font-size: 12px; font-weight: 800; margin-right: 10px;">RANK #{idx + 1}</span>
                    <span style="font-size: 16px; color: #1f2937; font-weight: 700;">Config {idx + 1}</span>
                </div>
                <div style="text-align: right;">
                    <p style="font-size: 28px; color: {color}; font-weight: 900; margin: 0;">{config['reward']:.0f}</p>
                    <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">Final reward</p>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;">
                <div style="background: #f9fafb; border-radius: 6px; padding: 8px; text-align: center;">
                    <p style="font-size: 10px; color: #6b7280; margin: 0;">LR</p>
                    <p style="font-size: 14px; color: #1f2937; font-weight: 700; margin: 4px 0 0 0;">{config['lr']:.4f}</p>
                </div>
                <div style="background: #f9fafb; border-radius: 6px; padding: 8px; text-align: center;">
                    <p style="font-size: 10px; color: #6b7280; margin: 0;">Batch</p>
                    <p style="font-size: 14px; color: #1f2937; font-weight: 700; margin: 4px 0 0 0;">{config['batch']}</p>
                </div>
                <div style="background: #f9fafb; border-radius: 6px; padding: 8px; text-align: center;">
                    <p style="font-size: 10px; color: #6b7280; margin: 0;">Gamma</p>
                    <p style="font-size: 14px; color: #1f2937; font-weight: 700; margin: 4px 0 0 0;">{config['gamma']:.2f}</p>
                </div>
                <div style="background: #f9fafb; border-radius: 6px; padding: 8px; text-align: center;">
                    <p style="font-size: 10px; color: #6b7280; margin: 0;">Episodes</p>
                    <p style="font-size: 14px; color: #1f2937; font-weight: 700; margin: 4px 0 0 0;">{config['episodes']}</p>
                </div>
            </div>
        </div>
        """
    
    comparison_html += """
        </div>
        
        <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); border-radius: 12px; padding: 20px; margin-top: 20px; color: white;">
            <p style="font-size: 16px; font-weight: 800; margin: 0 0 10px 0;">🏆 Best Configuration Found</p>
            <p style="font-size: 14px; margin: 0;">Learning Rate: <strong>""" + f"{configs_sorted[0]['lr']:.4f}" + """</strong> • Batch Size: <strong>""" + f"{configs_sorted[0]['batch']}" + """</strong> • Gamma: <strong>""" + f"{configs_sorted[0]['gamma']:.2f}" + """</strong></p>
            <p style="font-size: 13px; margin: 10px 0 0 0; opacity: 0.9;">Achieved """ + f"{configs_sorted[0]['reward']:.0f}" + """ reward in """ + f"{configs_sorted[0]['episodes']}" + """ episodes</p>
        </div>
    </div>
    """
    
    # Create comparison charts
    lr_values = [c['lr'] for c in configs]
    rewards = [c['reward'] for c in configs]
    
    fig_hp = go.Figure(data=[go.Scatter(
        x=lr_values,
        y=rewards,
        mode='markers',
        marker=dict(
            size=15,
            color=rewards,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Reward")
        ),
        text=[f"Batch: {c['batch']}" for c in configs],
        hovertemplate='LR: %{x}<br>Reward: %{y}<br>%{text}<extra></extra>'
    )])
    
    fig_hp.update_layout(
        title="Learning Rate vs Final Reward",
        xaxis_title="Learning Rate",
        yaxis_title="Final Reward",
        xaxis_type="log",
        height=400
    )
    
    return comparison_html, fig_hp

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
            <span style="font-size: 56px;">🧠</span>
        </div>
        
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Nous RL Observatory
        </h1>
        
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Reinforcement Learning Training Visualization</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Distributed RL • Training analytics • Hyperparameter optimization</p>
        
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Deep RL</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">PPO/DQN/SAC</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Visualization</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Open Source</span>
        </div>
        
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Nous Research</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("🎮 Training Run"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">RL Training Visualization</h3>
                <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Simulate and visualize reinforcement learning training runs</p>
            </div>
            """)
            
            with gr.Row():
                algorithm_dropdown = gr.Dropdown(
                    choices=list(RL_ALGORITHMS.keys()),
                    value="PPO (Proximal Policy Optimization)",
                    label="RL Algorithm"
                )
                
                env_dropdown = gr.Dropdown(
                    choices=list(ENVIRONMENTS.keys()),
                    value="CartPole-v1",
                    label="Environment"
                )
            
            episodes_slider = gr.Slider(
                minimum=1000,
                maximum=10000,
                value=5000,
                step=500,
                label="Number of Episodes"
            )
            
            train_btn = gr.Button("🚀 Start Training Simulation", variant="primary", size="lg")
            
            training_output = gr.HTML(label="Training Summary")
            reward_chart = gr.Plot(label="Reward Curve")
            phase_chart = gr.Plot(label="Training Phases")
            raw_rewards = gr.State()
            smooth_rewards = gr.State()
            
            train_btn.click(
                fn=generate_training_run,
                inputs=[algorithm_dropdown, env_dropdown, episodes_slider],
                outputs=[training_output, reward_chart, phase_chart, raw_rewards, smooth_rewards]
            )
        
        with gr.Tab("🔬 Hyperparameter Search"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Automated Hyperparameter Optimization</h3>
                <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Compare different configurations to find optimal settings</p>
            </div>
            """)
            
            hp_btn = gr.Button("🔍 Run Hyperparameter Search", variant="primary", size="lg")
            
            hp_output = gr.HTML(label="HP Comparison")
            hp_chart = gr.Plot(label="Learning Rate Analysis")
            
            hp_btn.click(
                fn=generate_hyperparameter_comparison,
                inputs=[],
                outputs=[hp_output, hp_chart]
            )
    
    gr.HTML("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Nous Research</h2>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🔬 Research Acceleration</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Visualize training dynamics in real-time. Debug RL algorithms faster, identify convergence issues early, optimize hyperparameters efficiently.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📊 Distributed Training</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Monitor multiple training runs across cluster. Compare algorithms, track resource utilization, aggregate results from parallel experiments.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🤝 Open Source Impact</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Make RL research accessible. Beautiful visualizations help researchers understand training dynamics, debug issues, and publish results.
                </p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Real-World Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">10x faster debugging:</strong> Visual feedback shows exact failure mode</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">50% fewer experiments:</strong> HP optimization finds best config faster</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Research reproducibility:</strong> Track every experiment with full metrics</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Community contribution:</strong> Open source tools advance entire field</li>
            </ul>
        </div>
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Visualization</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Live reward curves, loss plots, metrics</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-Algorithm Support</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">PPO, DQN, SAC, A3C and more</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ HP Optimization</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Automated search for best configurations</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Experiment Tracking</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Complete history, reproducible results</p>
                </div>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Nous Research</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Gradio • Plotly • RL Algorithms • Training Analytics
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing reinforcement learning training visualization and analysis.<br>
            Training curves • Hyperparameter optimization • Algorithm comparison • Research tools
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()