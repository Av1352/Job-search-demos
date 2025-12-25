---
title: Serif Health ML Price Predictor
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🏥 Serif Health - Healthcare Price Predictor

**Built by Anju Vilashni Nandhakumar** | MS in AI, Northeastern University (2025)

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/Av1352/serif-health-ml-demo)

## 🎯 Project Overview

A complete machine learning pipeline for predicting healthcare prices with transparency and explainability. This demo was built for Serif Health's ML Engineer position to showcase:

- ✅ Real ML model training (not just API calls)
- ✅ Feature engineering & importance analysis
- ✅ Model evaluation metrics (R², MAE, RMSE)
- ✅ Prediction explanations (SHAP-style)
- ✅ Interactive data visualizations

## 🤖 ML Model Details

### Architecture
- **Algorithm:** Linear Regression with Gradient Descent
- **Training Data:** 500 synthetic healthcare pricing samples
- **Features:** 4 categorical variables
  - Procedure Type (6 options)
  - Geographic Location (5 cities)
  - Insurance Type (5 plans)
  - Facility Type (4 categories)
- **Training:** 1000 iterations, learning rate 0.001

### Performance
- **R² Score:** 85%+ (strong predictive power)
- **MAE:** ~$250 (average prediction error)
- **RMSE:** ~$300 (prediction variance)
- **Training Time:** <1 second

### Key Features
1. **Training Visualization** - Loss curve shows model convergence
2. **Feature Importance** - Bar chart of feature weights
3. **Prediction Explanations** - Individual feature contributions
4. **Facility Comparisons** - Compare 4 facilities simultaneously
5. **Interactive Charts** - Built with Plotly for exploration

## 🚀 Try It Out

1. Select a medical procedure (e.g., MRI, Surgery, Lab Test)
2. Choose your location and insurance type
3. Click "Run ML Prediction" to see:
   - Predicted prices across 4 facility types
   - Feature importance analysis
   - Model training progress
   - Prediction explanations

## 💡 Why This Matters

Healthcare price transparency is critical:
- Prices vary by **300%+** for the same procedure
- Patients can save **$1,000+** by comparing facilities
- ML models can predict costs with **85%+ accuracy**
- Explainability builds trust and meets regulatory requirements

## 🏗️ Technical Implementation

### Files
- `app.py` - Main Gradio interface
- `model.py` - ML model implementation
- `requirements.txt` - Python dependencies

### Tech Stack
- **ML:** NumPy for gradient descent implementation
- **Visualization:** Plotly for interactive charts
- **UI:** Gradio for web interface
- **Deployment:** Hugging Face Spaces

### Code Highlights
```python
# Custom gradient descent implementation
def train(self, learning_rate=0.001, iterations=1000):
    for iter_num in range(iterations):
        predictions = self.weights[0] + X @ self.weights[1:]
        loss = np.mean((predictions - y) ** 2)
        
        error = predictions - y
        grad_bias = np.mean(error)
        grad_weights = (X.T @ error) / n
        
        self.weights[0] -= learning_rate * grad_bias
        self.weights[1:] -= learning_rate * grad_weights
```

## 📊 Production Enhancements

For a production system, this would include:

1. **Model Improvements**
   - XGBoost/LightGBM for better accuracy
   - Confidence intervals with bootstrapping
   - Ensemble methods for robustness

2. **Data Pipeline**
   - Real-time claims data ingestion
   - Automated data validation
   - Feature store for consistency

3. **MLOps**
   - Automated retraining on new data
   - A/B testing framework
   - Model monitoring dashboard
   - Performance drift detection

4. **Explainability**
   - SHAP integration for regulatory compliance
   - Counterfactual explanations
   - Fairness metrics

## 👤 About Me

**Anju Vilashni Nandhakumar**
- MS in AI, Northeastern University (2025)
- ML Engineer specializing in Medical Imaging & Computer Vision
- 96% accuracy on histopathology tumor classification
- 6 months of active job searching with targeted technical demonstrations
- Passionate about healthcare AI and price transparency

**Why Serif Health?**
Healthcare price transparency directly impacts millions of Americans. I've experienced the confusion of medical billing firsthand, and I'm passionate about building ML systems that empower patients with clear, actionable information. Serif Health's mission aligns perfectly with my technical expertise in medical imaging and my drive to make healthcare more accessible and equitable.

**What I Bring:**
- Strong ML engineering foundation with hands-on model training experience
- Healthcare domain expertise from medical imaging projects
- Full-stack capabilities (React, Python, deployment)
- Proven ability to build production-ready ML systems
- Commitment to explainable AI for regulatory compliance

## 📞 Connect

- 💼 **LinkedIn:** [linkedin.com/in/anju-vilashni](https://linkedin.com/in/anju-vilashni)
- 💻 **GitHub:** [github.com/Av1352](https://github.com/Av1352)
- 🌐 **Portfolio:** [vxanju.com](https://vxanju.com)
- 📧 **Email:** nandhakumar.anju@gmail.com

---

## 📄 License

MIT License - Feel free to use this code for learning or your own projects!

---

*Built with ❤️ for Serif Health | December 2024*
```

---
