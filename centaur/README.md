# 🔍 Centaur AI - Real-Time ML Model Monitor

**Production-ready quality assurance for machine learning systems**

Built for **Centaur AI** by Anju Nandhakumar

🔗 **[Live Demo](https://huggingface.co/spaces/av1352/centaur-ml-monitor)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)**

---

## What This Does

Real-time ML model monitoring with statistical drift detection and performance tracking.

**Features:**
- Multi-method drift detection (KS test, PSI, JS divergence)
- Real-time performance metrics (accuracy, precision, recall, F1)
- Intelligent alerting (LOW/MEDIUM/HIGH severity)
- Sliding window analysis (configurable 50-500 predictions)

**Example:** Deploy model → PSI score hits 0.25 → Alert: "Significant drift detected - retrain immediately" → Prevents bad predictions

---

## Why It Matters

**Problem:** ML models degrade in production when data distributions shift  
**Solution:** Real-time drift detection catches issues 1-2 weeks before users notice quality problems

**ROI:** Prevents costly prediction errors, automates 80% of manual QA work, compliance-ready audit trails

---

## Demo Features

✓ 5 drift scenarios (No Drift, Gradual, Sudden, Seasonal, Feature)  
✓ Statistical rigor (KS test p-values, PSI thresholds 0.1/0.2, JS divergence)  
✓ Performance trends (20-point time series visualization)  
✓ Actionable alerts (with specific recommendations)

---

## Statistical Methods

- **Population Stability Index (PSI)**: Banking industry standard
  - < 0.1: No drift ✅
  - 0.1-0.2: Moderate drift ⚠️
  - \> 0.2: Significant drift - retrain 🚨

- **Kolmogorov-Smirnov Test**: Non-parametric distribution comparison
- **Jensen-Shannon Divergence**: Symmetric KL divergence (0-1 scale)
- **Statistical Distance**: Mean and variance shift detection

---

## Tech Stack

Python • Gradio • SciPy • NumPy • Plotly • Statistical ML

---

## Impact

- Early detection prevents model degradation
- Automated QA reduces manual labor by 80%
- Compliance audit trails for regulated industries
- Clear recommendations, not just alerts

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)  

Built with ❤️ for Centaur AI