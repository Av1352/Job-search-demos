# 🛒 Spur – AI Shopper Simulation Platform

**Automated A/B testing with AI shoppers for faster e‑commerce optimization**

Built for **Spur** by **Anju Nandhakumar**  

🔗 **[Live Demo](https://huggingface.co/spaces/Av1352/spur-shopper-simulation)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)**  

---

## What This Does

Simulation engine that uses AI shopper personas to test e‑commerce experiences before exposing real traffic.  

**Features:**
- 5 behavioral personas (Budget Hunter, Impulse Buyer, Research Shopper, Loyal Customer, Window Shopper)  
- 4 scenario types: product page, checkout flow, pricing, homepage layout  
- 100–10,000 simulated sessions per run with multi-variant tests (up to 4 variants)  
- Outputs conversion, revenue, and winner recommendation per test  

**Example Flow:**  
Pick persona + scenario + variants → run simulation → compare conversion/revenue per variant → pick a winner in minutes instead of weeks.  

---

## Why It Matters

Traditional A/B testing is slow, traffic-hungry, and risks real revenue on bad variants.  

This demo shows how Spur-style simulation can:  
- De-risk radical design and pricing changes  
- Let teams run many more experiments than real traffic allows  
- Provide persona-level insight instead of a single blended metric  

---

## Demo Features

**Run Simulation tab:**
- Configure persona, scenario, and simulated session count  
- See ranked variants with conversion, revenue, and lift vs baseline  
- Persona insight block (session time, cart value, price sensitivity)  

**Test Dashboard tab:**
- Monthly summary of tests run, total simulated sessions, and aggregate conversion lift  
- Breakdown by scenario type (product page, checkout, pricing, homepage)  

---

## Tech Stack

Python simulation engine • Persona behavior models • Basic stats (lift, significance-style metrics) • Plotly-style charts • Gradio UI • Hugging Face Spaces  

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)  

Built with ❤️ for Spur