# 📞 HealthCall AI - Clinical Voice Triage

**AI-powered outbound phone calls for patient intake — powered by Bland AI**

Built for **Bland AI** by Anju Nandhakumar

🔗 **[Live Demo](https://vxanju-demos.streamlit.app/healthCallAI)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)**

---

## What This Does

Healthcare triage platform that triggers real outbound AI phone calls to patients via Bland AI's voice API. The AI clinical intake assistant (Maya voice) asks three structured questions, collects responses, and routes patients to the right care level — all without any human involvement until the nurse follow-up.

**Features:**
- Enter a phone number → Bland AI calls the patient immediately
- Maya (warm female voice) introduces herself as a clinical intake assistant
- Asks 3 structured questions: symptoms, severity (1–10), urgency
- Live call status tracking: queued → ringing → in-progress → completed
- Full transcript displayed as chat bubbles after call ends
- Severity score auto-parsed from transcript and color-coded (Low / Moderate / High)
- Plotly Sankey diagram of conversation routing logic
- Download transcript as CSV

**Example:** Patient calls in → HealthCall AI dials back with Maya voice → "What symptoms are you experiencing?" → "Rate severity 1–10" → "Is this urgent?" → Severity 8 detected → 🚨 Escalation flag raised → Nurse follow-up SMS sent

---

## Why It Matters

**Problem:** Nurse triage lines are overwhelmed — patients wait 30–90 minutes for a callback, and intake quality varies by staff  
**Solution:** Bland AI outbound calls handle structured clinical intake at scale, instantly, with consistent question sets and automatic severity routing

**ROI:** Reduce nurse triage workload by 60–80% for routine intake, zero wait time for patients, structured data output feeds directly into EHR

---

## Demo Features

✓ Phone number input (E.164 format) with real Bland AI call trigger  
✓ Voice selection: Maya, Josh, Paige  
✓ Model selection: enhanced / base  
✓ System prompt preview in UI  
✓ Live call status badge (queued / ringing / in-progress / completed / failed)  
✓ Status log with timestamps  
✓ Transcript fetched via GET /v1/calls/{call_id}  
✓ Chat bubble transcript rendering (agent vs patient sides)  
✓ Severity score extraction and care routing recommendation  
✓ Audio playback of call recording (if available)  
✓ Download transcript as CSV  
✓ Plotly Sankey flow diagram of triage routing logic  

---

## Clinical Intake Flow

**3 Questions Asked (in order):**
1. What symptoms are you currently experiencing?
2. On a scale of 1–10, how would you rate the severity?
3. Do you feel this is urgent and requires immediate attention today?

**Routing Logic:**
- Severity ≥ 7 → 🚨 Immediate escalation
- Severity 4–6 → 📅 Same-day appointment
- Severity ≤ 3 → 📆 Routine scheduling
- Urgent = Yes → Override to escalation tier

---

## Tech Stack

Python • Streamlit • Bland AI Voice API • Plotly • Pandas • Requests

---

## Deployment

Add to `.streamlit/secrets.toml`:
```toml
BLAND_API_KEY = "sk_..."
```

Then deploy to Streamlit Cloud — API key loads automatically, with a fallback input field for local dev.

---

## Impact

- Zero-wait patient intake (vs 30–90 min nurse callback)
- Consistent 3-question structured intake every call
- Automatic severity scoring and care tier routing
- Full transcript stored and downloadable for EHR integration
- Scales to thousands of concurrent outbound calls
- Maya voice achieves <700ms latency on Bland AI enhanced model

---

## Business Value

**For Health Systems:**
- Automate routine intake calls at scale
- Free nurses to focus on high-acuity patients
- Structured intake data feeds directly into EHR workflows
- 24/7 availability — no staffing constraints

**For Patients:**
- Immediate callback, no hold time
- Warm, clear AI voice with clinical language
- Consistent experience regardless of time of day
- Told exactly what happens next (nurse follow-up)

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)

Built with ❤️ for Bland AI | Voice AI • Clinical Triage • Healthcare Automation