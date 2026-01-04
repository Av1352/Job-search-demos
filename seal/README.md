# 🔬 Seal – GxP Data Validation Platform

**Automated data validation and quality control for biotech and pharma**

Built for **Seal** by **Anju Nandhakumar**  

🔗 **[Live Demo](https://huggingface.co/spaces/av1352/seal-gxp-validation)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)**  

---

## What This Does

GxP-focused engine that validates clinical and lab data against ALCOA+ and regulatory rules, with full audit trails.  

**Features:**
- Clinical data checks: completeness, ranges, and outlier detection on vitals/labs  
- ALCOA+ coverage: Attributable, Legible, Contemporaneous, Original, Accurate, Complete, Consistent, Enduring, Available  
- Automated quality scoring and “Compliant / Needs Review / Non‑Compliant” flags  
- Audit trail viewer: who changed what, when, and why  

**Example Flow:**  
Upload or stream trial data → run validation rules → view quality scores, flagged records, and auto-generated audit entries.  

---

## Why It Matters

Manual GxP validation is slow, error‑prone, and directly tied to FDA delays and trial risk.  

This demo shows how Seal-style automation can:  
- Catch issues at point of entry instead of months later  
- Provide 100% record coverage with consistent rules  
- Generate inspection‑ready audit trails by default  

---

## Demo Features

**Data Validation tab:**
- 50 synthetic trial records with vitals and labs  
- Completeness, normal‑range, and outlier checks per field  
- Per-record quality score and overall compliance status  

**Audit Trail tab:**
- Sample change history entries (user, timestamp, old→new, reason)  
- Activity timeline to illustrate inspection readiness  

---

## Tech Stack

Python • Pandas • NumPy • Rules/statistical validation engine • Plotly-style visuals • Gradio UI  

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)  

Built with ❤️ for Seal