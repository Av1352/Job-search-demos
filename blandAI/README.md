# 🧠 blandHealthCallML - Clinical Triage + Post-Call ML Analysis

**Bland AI voice intake + Claude NLP layer: symptom NER, urgency classification, sentiment analysis, clinical summary**

Built for **Bland AI** by Anju Nandhakumar

🔗 **[Live Demo](https://vxanju-demos.streamlit.app/blandHealthCallML)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)**

---

## What This Does

Two-layer clinical AI system: Bland AI handles the outbound voice call and structured intake, then Claude runs a full NLP analysis pass on the transcript — extracting symptoms, classifying urgency, reading patient sentiment, and generating a clinical summary a nurse can act on immediately.

**Features:**
- Bland AI triggers outbound call (Maya voice, enhanced model)
- AI asks 3 structured intake questions: symptoms, severity 1–10, urgency
- Live call status tracking: queued → ringing → in-progress → completed
- Transcript rendered as chat bubbles (agent left, patient right)
- One click: **Run ML Analysis** sends transcript to Claude
- ML layer returns: symptom NER, urgency tier, sentiment, clinical summary, recommended action
- Patient profile radar chart (severity, urgency, distress, symptom count, phrase density)
- Export ML results as JSON for EHR integration

**Example:** Patient answers call → describes chest tightness + shortness of breath → severity 8 → urgent → Claude extracts symptoms ["chest tightness", "shortness of breath"], classifies Urgent 🚨, detects Distressed 😰 sentiment, generates: *"Patient reports acute chest tightness with severity 8/10, shortness of breath, and requests urgent attention. Recommend immediate nurse callback within 15 minutes."*

---

## Why It Matters

**Problem:** Even with AI voice intake, transcript data sits unstructured — nurses still read raw call logs and manually decide urgency  
**Solution:** Post-call ML layer instantly structures the transcript into clinical NER, urgency classification, and a ready-to-act summary — zero manual triage

**ROI:** Nurse decision time cut from 10–15 min/call review to 30 seconds with structured ML output, severity-routed escalations, and one-line recommended action

---

## Demo Features

✓ Real Bland AI outbound call (POST /v1/calls)  
✓ Live status polling (GET /v1/calls/{call_id})  
✓ Chat bubble transcript rendering  
✓ One-click Claude ML analysis on transcript  
✓ Symptom NER — extracted as purple tag chips  
✓ Urgency classification: Urgent 🚨 / Monitor ⚠️ / Routine ✅  
✓ Urgency reasoning — one-sentence explanation  
✓ Sentiment analysis: Distressed / Anxious / Neutral / Calm / Relieved  
✓ Sentiment reasoning — one-sentence explanation  
✓ Severity score extracted from patient speech  
✓ Auto-generated clinical summary (2–3 sentences)  
✓ Recommended action for care team  
✓ Key phrases extracted as amber chips  
✓ Patient profile radar chart (5-dimension visualization)  
✓ Export ML results as JSON  

---

## ML Analysis Outputs

**Symptom NER:**
- Entity extraction of all symptoms mentioned by patient
- Rendered as visual tag chips for fast nurse scanning

**Urgency Classification (3-tier):**
- Urgent — immediate escalation, nurse callback <15 min
- Monitor — same-day appointment, track symptom progression
- Routine — standard scheduling, no acute concern

**Sentiment Analysis:**
- Distressed / Anxious / Neutral / Calm / Relieved
- One-sentence reasoning grounded in patient's speech patterns

**Clinical Summary:**
- 2–3 sentence structured summary written for clinical staff
- Includes symptoms, severity, urgency, and patient state

**Recommended Action:**
- Specific next step: "Dispatch nurse callback immediately" vs "Schedule 3-day follow-up"

---

## Tech Stack

Python • Streamlit • Bland AI Voice API • Anthropic Claude Sonnet • Plotly • Pandas • Requests

---

## Deployment

Both keys load from `.streamlit/secrets.toml` — no input fields shown once set:
```toml
BLAND_API_KEY     = "sk_..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## Impact

- Zero-wait patient intake via Bland AI outbound call
- Structured ML output replaces 10–15 min manual transcript review
- Urgency routing reduces critical escalation misses
- JSON export integrates directly with EHR/CRM workflows
- Scales to thousands of calls with consistent NLP quality
- Two AI systems working in sequence — voice layer + reasoning layer

---

## Business Value

**For Health Systems:**
- Automate triage from call → structured clinical note in one pipeline
- Nurses receive pre-analyzed summaries, not raw transcripts
- Urgency flags reduce risk of missed escalations
- Audit-ready JSON output for every patient interaction

**For Bland AI:**
- Demonstrates the full stack of what's possible beyond the call itself
- NLP analysis layer dramatically increases the value of voice data
- Shows enterprise healthcare use case with clinical-grade outputs

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)

Built with ❤️ for Bland AI | Voice AI • Clinical NLP • Symptom NER • Urgency Classification • Sentiment Analysis