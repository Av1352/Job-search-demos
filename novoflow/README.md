# 🏥 Novoflow Medical Triage AI

**Intelligent symptom triage and scheduling powered by BioBERT medical NLP**  

Built for **Novoflow** by Anju Nandhakumar  

🔗 **[Live Demo](https://vxanju-demos.streamlit.app/novoflow)** | 💻 **[Source](https://github.com/Av1352/Job-search-demos/tree/main/novoflow)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)**  

---

## What This Does

AI triage assistant that understands free-text symptoms, classifies clinical urgency, and generates an appropriate appointment plan.  

**Features:**
- BioBERT-based medical NER on PubMed-trained `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`  
- ESI-style urgency levels: Emergency, Urgent, Specialist, Routine with confidence scores  
- Smart scheduling suggestions: timing + specialty routing  
- Multilingual support concept (25+ languages) and safety-first triage rules  

**Example Flow:**  
Patient types symptoms → BioBERT extracts medical entities → rule engine applies ESI-inspired logic → system returns urgency, reasoning, and a proposed appointment window.  

---

## Why It Matters

**Problem:** Clinics lose revenue and delay care because symptom intake, triage, and booking still rely on overworked staff and legacy EHR workflows.  
**Solution:** A hybrid ML + rules engine that pre-triages patients, routes them to the right provider, and automates scheduling on top of any EHR.  

**Impact:** Supports the vision of Novoflow’s AI employees that handle phone calls, appointments, and cancellation recovery while maintaining clinical safety and HIPAA-aware workflows.  

---

## Demo Features

✓ Example scenarios: chest pain, high fever, rash, routine checkup  
✓ Live BioBERT NER visualization for extracted entities and body regions  
✓ Urgency label + confidence, clinical reasoning text, and suggested appointment window  
✓ Simple architecture view showing NLP → rules → scheduling steps  

---

## Tech Stack

BioBERT (PubMedBERT) • PyTorch • Hugging Face Transformers • Python • Gradio UI  

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)  

Built with ❤️ for Novoflow