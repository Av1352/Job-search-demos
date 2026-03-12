# 🔬 LayerLens - Clinical Chart Abstractor

**AI-powered structured data extraction from clinical notes with source-linked evidence**

Built for **Layer Health** by Anju Nandhakumar

🔗 **[Live Demo](https://vxanju-demos.streamlit.app/layerLens)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)**

---

## What This Does

Clinical NLP platform that extracts structured fields from free-text clinical notes — and links every extracted value back to its exact source span in the original note, highlighted by field. Directly mirrors Layer Health's core capability: physician-level reasoning over longitudinal patient charts to answer complex clinical questions with evidence-based justifications.

**Features:**
- Paste any clinical note (EHR export, dictation, scanned text)
- Select abstraction task: Registry, Quality Measurement, or Research Cohort
- LLM extracts 8–14 structured fields with task-specific schemas
- Every field shows: name, value, confidence score (0–100%), and verbatim source quote
- Original note rendered with color-coded highlights — one color per field, hover to see field name
- Analytics tab: confidence bar chart, tier distribution, field-level detail table
- 3 pre-loaded sample notes (STEMI, T2DM, NSCLC staging)

**Example:** Paste a cardiology admission note → Registry Abstraction → extracts primary diagnosis (ICD-10), troponin value, procedure (PCI + CPT), door-to-balloon time, discharge medications, comorbidities — each field highlighted in the original text at 94%+ confidence

---

## Why It Matters

**Problem:** Clinical chart abstraction for registries, quality measures, and research cohorts is manual, slow (15–45 min/chart), and error-prone — the exact problem Layer Health is solving at scale for health systems like White Plains Hospital, Froedtert, and Intermountain Health  
**Solution:** LLM-powered extraction with verbatim source linking gives abstractors AI-assisted speed with human-verifiable evidence — mirroring Layer Health's approach of "teeing up" answers grounded in the medical record

**ROI:** 10x faster abstraction (45 min → 3–5 min/chart), source-linked confidence scoring reduces secondary review burden, structured output drops directly into registry or research databases

---

## Demo Features

✓ Free-text clinical note input (any format)  
✓ 3 pre-loaded sample notes (Cardiology / STEMI, Endocrinology / T2DM, Oncology / NSCLC)  
✓ Registry Abstraction — ICD-10, CPT, procedures, vitals, labs, disposition  
✓ Quality Measurement — HbA1c, BP, screenings, vaccination status, care gaps  
✓ Research Cohort — biomarkers, staging, ECOG, comorbidities, exclusion factors  
✓ Per-field confidence scoring (High / Medium / Low tiers)  
✓ Color-coded source highlighting with hover tooltips  
✓ Verbatim source quote shown under each extracted field  
✓ Analytics: confidence bar chart, tier pie, KPI summary, detail table  
✓ Deploy-ready: Streamlit Cloud + secrets.toml API key management  

---

## Extraction Capabilities

**Registry Abstraction (Cardiovascular, Oncology, Surgery):**
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

**Research Cohort (Real-World Evidence / Clinical Trials):**
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
- Source-linked evidence reduces secondary review burden by abstractors
- Task-specific field schemas for 3 major abstraction workflows
- Confidence scoring enables prioritized human review queue
- Verbatim source quotes for auditability and compliance
- Deploy-ready with Streamlit Cloud secrets management

---

## Business Value

**For Health Systems & Registries:**
- Automate cardiovascular, oncology, and surgery registry submissions
- Reduce abstractor FTE costs for quality reporting
- Audit-ready source evidence for every extracted field
- Scale registry reporting without additional staffing

**For Clinical Research & Life Sciences:**
- Rapidly identify research cohort candidates from EHR data
- Extract real-world evidence at scale (hours vs. months)
- Biomarker and staging data ready for downstream analysis
- Reproducible extraction with confidence thresholds for GCP compliance

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)

Built with ❤️ for Layer Health | Clinical NLP • Chart Abstraction • Registry AI • Real-World Evidence