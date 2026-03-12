# 🔬 LayerLens - Clinical Chart Abstractor

**AI-powered structured data extraction from clinical notes with source-linked evidence**

Built by **Anju Nandhakumar**

🔗 **[Live Demo](https://vxanju-demos.streamlit.app/layerLens)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)**

---

## What This Does

Clinical NLP platform that extracts structured fields from free-text clinical notes — and links every extracted value back to its exact source span in the original note, highlighted by field. Supports three abstraction workflows: Registry, Quality Measurement, and Research Cohort.

**Features:**
- Paste any clinical note (EHR export, dictation, scanned text)
- Select abstraction task: Registry, Quality Measurement, or Research Cohort
- Claude extracts 8–14 structured fields with task-specific schemas
- Every field shows: name, value, confidence score (0–100%), and verbatim source quote
- Original note rendered with color-coded highlights — one color per field, hover to see field name
- Analytics tab: confidence bar chart, tier distribution, field-level detail table
- 3 pre-loaded sample notes (STEMI, T2DM, NSCLC staging)

**Example:** Paste a cardiology admission note → Registry Abstraction → extracts primary diagnosis (ICD-10), troponin value, procedure (PCI + CPT), door-to-balloon time, discharge medications, attending provider, comorbidities — each field highlighted in the original text at 94%+ confidence

---

## Why It Matters

**Problem:** Clinical chart abstraction for registries, quality measures, and research cohorts is manual, slow (15–45 min/chart), and error-prone — costing health systems millions in abstractor hours  
**Solution:** LLM-powered extraction with verbatim source linking gives abstractors AI-assisted speed with human-verifiable evidence

**ROI:** 10x faster abstraction (45 min → 3–5 min/chart), source-linked confidence scoring reduces need for secondary review, structured output drops directly into registry or research databases

---

## Demo Features

✓ Free-text clinical note input (any format)  
✓ 3 pre-loaded sample notes (Cardiology, Endocrinology, Oncology)  
✓ Registry Abstraction — ICD-10, CPT, procedures, vitals, labs, disposition  
✓ Quality Measurement — HbA1c, BP, screenings, vaccination, care gaps  
✓ Research Cohort — biomarkers, staging, ECOG, comorbidities, exclusion factors  
✓ Per-field confidence scoring (High / Medium / Low tiers)  
✓ Color-coded source highlighting with hover tooltips  
✓ Verbatim source quote shown under each extracted field  
✓ Analytics: confidence bar chart, tier pie, KPI summary, detail table  
✓ Deploy-ready: Streamlit Cloud + secrets.toml API key management  

---

## Extraction Capabilities

**Registry Abstraction:**
- Patient demographics (age, sex)
- Primary diagnosis with ICD-10 code
- Key procedures with CPT codes
- Admission/discharge dates and disposition
- Relevant lab values (troponin, HbA1c, BNP, eGFR)
- Discharge medications and follow-up plan

**Quality Measurement (HEDIS / CMS / Joint Commission):**
- Blood pressure and HbA1c values with trends
- Preventive screenings completed (with dates)
- Medication classes prescribed
- Vaccination status, BMI, smoking status
- Referrals placed, patient education documented

**Research Cohort:**
- Disease stage/severity (AJCC, GOLD, etc.)
- Biomarkers (mutation status, PD-L1 TPS, SUVmax)
- Functional status (ECOG, ADLs)
- Comorbidities as potential exclusion factors
- Prior treatments and key imaging findings

---

## Tech Stack

Python • Streamlit • Anthropic Claude claude-opus-4-6 • Plotly • Pandas • Regex span matching

---

## Impact

- 10x faster chart abstraction (45 min → 3–5 min per chart)
- Source-linked evidence reduces secondary review burden
- Task-specific field schemas for 3 major abstraction workflows
- Confidence scoring enables prioritized human review
- Verbatim source quotes for auditability and compliance
- Deploy-ready with Streamlit Cloud secrets management

---

## Business Value

**For Health Systems & Registries:**
- Slash abstractor FTE costs for quality reporting
- Accelerate registry submission deadlines
- Audit-ready source evidence for every extracted field
- Drop structured output directly into registry databases

**For Clinical Research Teams:**
- Rapidly identify research cohort candidates
- Extract inclusion/exclusion criteria elements at scale
- Biomarker and staging data ready for analysis
- Reproducible extraction with confidence thresholds

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)

Built with ❤️ by Anju Vilashni Nandhakumar | Clinical NLP • Chart Abstraction • Healthcare AI