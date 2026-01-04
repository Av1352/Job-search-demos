# 🏥 Paratus Health – AI Pre-Visit Intake Assistant

**Structured clinical summaries from patient conversations using a multi-model ML pipeline**

Built for **Paratus Health** by Anju Nandhakumar  

🔗 **[Live Demo](https://huggingface.co/spaces/av1352/paratus-health-intake)** | 💻 **[Source](https://github.com/Av1352/Job-search-demos/tree/main/paratus-health)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)**  

---

## What This Does

AI-powered pre-visit intake that turns free-text patient conversations into structured, doctor-ready clinical summaries.  

**Features:**
- Multi-model pipeline: BioBERT (NER), T5 (clinical summarization), DistilBERT (severity) – 236M params total  
- SOAP note generation (Subjective, Objective, Assessment, Plan)  
- Schmitt–Thompson protocol matching and red-flag detection  
- Pre-visit summaries formatted for EHR workflows (Epic, Athena, Cerner-ready)  

**Example Flow:**  
Patient describes symptoms → models extract entities, summarize HPI, and score severity → engine maps to triage protocol and generates a structured SOAP-style summary for the clinician.  

---

## Why It Matters

**Problem:** Static intake forms miss critical details and force doctors to spend the first 8–10 minutes of each visit rebuilding history.  
**Solution:** Conversational AI that runs intake before the visit, captures nuanced clinical context, and hands physicians a complete summary at chart open.  

**Impact:**  
- 8–10 minutes saved per visit  
- Higher completeness of history vs. forms  
- Less admin burden and better prepared encounters  

---

## Demo Features

✓ Example scenarios (cardiac, respiratory, dermatology, headache, etc.)  
✓ Auto-generated SOAP notes with HPI, assessment hints, and plan scaffolding  
✓ Highlighted red flags plus mapped Schmitt–Thompson-style protocol IDs  
✓ Full model pipeline view (NER → summarization → severity → protocols)  

---

## Tech Stack

BioBERT • T5-small • DistilBERT • PyTorch • Hugging Face Transformers • Python • Gradio UI  

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)  

Built with ❤️ for Paratus Health