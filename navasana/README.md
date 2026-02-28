# 🛡️ Navasana - AI Cyber Risk Underwriting Engine

Production demo of an AI-native cyber insurance underwriting platform that quantifies risk, prices premiums, and tracks threat intelligence in real-time.

🔗 **[LIVE DEMO](https://vxanju-demos.streamlit.app/navasanaCyberRisk)**

---

## What It Does

AI underwriting platform that automates security posture assessment and premium pricing by ingesting signals from CrowdStrike, Okta, and AWS SecurityHub — replacing slow, manual questionnaire-based underwriting with continuous, API-driven risk quantification.

**Example:** Mid-market SaaS company onboards → platform ingests 87 security signals → risk score generated in 2.3s → premium priced with 94.7% accuracy

---

## Features

**Risk Assessment Engine** — Multi-factor scoring across 10 security controls (MFA, EDR, SOC 2, patch cadence, IR plan) with industry and breach-history multipliers

**Portfolio Underwriting Analytics** — Scatter plots of risk vs. premium, industry benchmarks, claims prediction, tier breakdowns across simulated 120-policy portfolio

**Threat Intelligence Feed** — Live threat signals (AI-generated phishing +89% YoY, ransomware +34%) mapped to dynamic premium adjustments by industry

**ML Pipeline Architecture** — XGBoost risk scoring (94.7% accuracy), survival analysis for claims prediction, RAG-based underwriting agent on policy documents

---

## Tech Stack

```
streamlit · plotly · pandas · numpy · scikit-learn · xgboost
langchain · chromadb (RAG layer) · python-dotenv
```

---

## Run Locally

```bash
git clone https://github.com/Av1352/ml-demos
cd ml-demos
pip install -r requirements.txt
streamlit run pages/navasanaCyberRisk.py
```

---

## Business Impact

| Metric | Traditional | Navasana AI |
|--------|-------------|-------------|
| Assessment Time | 2–3 weeks | 2.3 seconds |
| Underwriting Accuracy | ~78% | 94.7% |
| Premium RMSE | $4,200 | $1,890 |
| Data Sources | Questionnaire | 5 live API feeds |

---

*Built by Anju Vilashni Nandhakumar · [vxanju.com](https://vxanju.com) · nandhakumar.anju@gmail.com*