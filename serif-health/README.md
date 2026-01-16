# 🏥 Serif Health – Healthcare Price Predictor

**Machine learning demo for transparent, explainable healthcare price prediction**

Built for **Serif Health** by **Anju Vilashni Nandhakumar**  

🔗 **[Live Demo](https://vxanju-demos.streamlit.app/serif_health)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)**  

---

## What This Does

End-to-end ML pipeline that predicts procedure prices and explains the drivers behind each prediction.  

**Features:**
- Custom linear regression with gradient descent (no off-the-shelf trainer)  
- 500-sample synthetic healthcare pricing dataset with 4 categorical features  
- Metrics: R², MAE, RMSE + loss-curve visualization  
- Feature importance and per-prediction contribution views  

**Example Flow:**  
Choose procedure, city, insurance, and facility type → run training/prediction → see prices across 4 facilities, model metrics, and explanation charts.  

---

## Model & Training

**Model:**
- Algorithm: Linear regression trained via gradient descent  
- Inputs: Procedure type, geography, insurance plan, facility type  
- Training: 1,000 iterations, learning rate 0.001, <1s per run  

**Performance (on synthetic data):**
- R² ≈ 0.85+  
- MAE ≈ \$250  
- RMSE ≈ \$300  

---

## Why It Matters

Healthcare prices for the same procedure can vary wildly across facilities and insurance plans.  

This demo shows how ML can:  
- Surface expected price ranges before care  
- Make variation visible and explorable  
- Provide explanations that are suitable for regulators and consumers  

---

## Tech Stack

NumPy (custom GD) • Plotly (charts) • Python • Gradio UI • Hugging Face Spaces  

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)  

Built with ❤️ for Serif Health