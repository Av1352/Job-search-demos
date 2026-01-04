# 🧠 Nous Research - RL Training Observatory

**Reinforcement learning training visualization and analysis platform**

Built for **Nous Research** by Anju Nandhakumar

🔗 **[Live Demo](https://huggingface.co/spaces/av1352/nous-rl-observatory)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)** 

---

## What This Does

Interactive observability tool for RL training runs: rewards, stability, and hyperparameters in one place.

**Features:**
- Live reward curves with smoothing and phase breakdown (exploration → learning → convergence)  
- Algorithm comparison (PPO, DQN, SAC, A3C) across classic control and MuJoCo-style environments  
- Hyperparameter search view with 5+ configs ranked by final reward and episodes to convergence  
- Training analytics: stability, sample efficiency, basic resource usage and run metadata

**Example Flow:**  
Pick algorithm + environment → simulate 1K–10K episodes → view live reward curve, convergence status, and ranked configs → download best settings for your own training loop. 

---

## Why It Matters

**Problem:** RL training is a black box; researchers rely on print logs and ad‑hoc plots, making debugging and HP tuning slow and compute‑wasteful.
**Solution:** Central dashboard for metrics, comparisons, and alerts so bad runs are caught early and good configs surface quickly. 
**Impact:** Fewer failed runs, faster iteration loops, and more reproducible RL experiments for open-source researchers.

---

## Demo Features

✓ Training tab with simulated PPO/DQN/SAC/A3C curves and solved/unsolved status  
✓ Environment presets (CartPole, LunarLander, BipedalWalker, Hopper-style) with target rewards  
✓ Hyperparameter tab comparing LR, batch size, and gamma across 5 configs  
✓ Visuals: reward curves (raw + MA), phase breakdown, and ranked config list

---

## Tech Stack

Python • Gradio • NumPy • Matplotlib/Plotly-style charts • RL-inspired simulation logic

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)  

Built with ❤️ for Nous Research