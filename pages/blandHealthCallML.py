"""
blandHealthCallML - Clinical Voice Triage + Post-Call ML Analysis
Bland AI outbound intake call + Claude-powered clinical NLP layer
Built for Bland AI by Anju Vilashni Nandhakumar
"""

import streamlit as st
import requests
import json
import re
import os
import html
import anthropic
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="blandHealthCallML – Clinical Triage + ML Analysis",
    page_icon="🧠",
    layout="wide"
)

from utils.sidebar import render_sidebar
render_sidebar()

# ── Constants ──────────────────────────────────────────────────────────────────
BRAND      = "#73BA9B"
BLAND_BASE = "https://api.bland.ai/v1"

TASK_PROMPT = (
    "You are HealthCall AI, a clinical intake assistant. "
    "Introduce yourself warmly, then ask exactly three questions one at a time: "
    "1) What symptoms are you currently experiencing? "
    "2) On a scale of 1 to 10, how would you rate the severity? "
    "3) Do you feel this is urgent and requires immediate attention today? "
    "Be warm, clear, and concise. After all three answers, summarize what you heard "
    "and say: 'A nurse from our care team will follow up with you shortly. "
    "Thank you for calling. Take care and feel better soon.'"
)

URGENCY_CONFIG = {
    "urgent":  {"color": "#ef4444", "bg": "#fef2f2", "icon": "🚨", "label": "Urgent"},
    "monitor": {"color": "#f59e0b", "bg": "#fffbeb", "icon": "⚠️",  "label": "Monitor"},
    "routine": {"color": "#22c55e", "bg": "#f0fdf4", "icon": "✅",  "label": "Routine"},
}

SENTIMENT_CONFIG = {
    "distressed":  {"color": "#ef4444", "icon": "😰"},
    "anxious":     {"color": "#f59e0b", "icon": "😟"},
    "calm":        {"color": "#22c55e", "icon": "😌"},
    "relieved":    {"color": BRAND,     "icon": "😊"},
    "neutral":     {"color": "#6b7280", "icon": "😐"},
}

STATUS_COLOR = {
    "queued": "#f59e0b", "ringing": "#3b82f6", "in-progress": "#8b5cf6",
    "completed": "#22c55e", "failed": "#ef4444", "no-answer": "#ef4444",
    "busy": "#f59e0b", "cancelled": "#6b7280",
}
STATUS_ICON = {
    "queued": "🟡", "ringing": "📳", "in-progress": "🔵",
    "completed": "✅", "failed": "❌", "no-answer": "📵",
    "busy": "🔴", "cancelled": "⚪",
}

# ── API Keys ───────────────────────────────────────────────────────────────────
def get_secret(key):
    try:
        return st.secrets[key]
    except (KeyError, AttributeError):
        return os.environ.get(key, "")

bland_key     = get_secret("BLAND_API_KEY")
anthropic_key = get_secret("ANTHROPIC_API_KEY")

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "call_id": None, "call_status": None, "transcript_data": None,
    "call_started": None, "call_log": [], "ml_results": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Bland helpers ──────────────────────────────────────────────────────────────
def bland_headers(key):
    return {"authorization": key, "Content-Type": "application/json"}

def start_call(phone, voice, model, key):
    r = requests.post(f"{BLAND_BASE}/calls", headers=bland_headers(key), json={
        "phone_number": phone, "task": TASK_PROMPT,
        "voice": voice, "model": model,
        "max_duration": 5, "record": True,
        "language": "en-US", "wait_for_greeting": True,
        "interruption_threshold": 150,
    }, timeout=15)
    r.raise_for_status()
    return r.json()

def get_call(call_id, key):
    r = requests.get(f"{BLAND_BASE}/calls/{call_id}", headers=bland_headers(key), timeout=10)
    r.raise_for_status()
    return r.json()

def parse_turns(data):
    turns = []
    raw = data.get("transcripts") or data.get("transcript") or []
    if isinstance(raw, list):
        for t in raw:
            if isinstance(t, dict):
                turns.append({
                    "role": t.get("user", t.get("role", "agent")),
                    "text": t.get("text", t.get("content", "")),
                })
    elif isinstance(raw, str) and raw.strip():
        for line in raw.strip().split("\n"):
            role, _, text = line.partition(":") if ":" in line else ("agent", "", line)
            turns.append({"role": role.strip(), "text": text.strip()})
    return turns

# ── Claude ML analysis ─────────────────────────────────────────────────────────
ML_SYSTEM = """You are a clinical NLP system. Analyze the patient side of a clinical intake 
call transcript and return ONLY a JSON object — no markdown, no preamble:
{
  "symptoms": ["symptom1", "symptom2"],
  "urgency": "urgent|monitor|routine",
  "urgency_reason": "one sentence explanation",
  "sentiment": "distressed|anxious|neutral|calm|relieved",
  "sentiment_reasoning": "one sentence",
  "severity_score": <integer 1-10 or null if not mentioned>,
  "clinical_summary": "2-3 sentence clinical summary a nurse would read",
  "recommended_action": "specific next step for the care team",
  "key_phrases": ["phrase1", "phrase2", "phrase3"]
}"""

def run_ml_analysis(turns, ant_key):
    patient_text = "\n".join(
        f"Patient: {t['text']}" for t in turns
        if "patient" in str(t.get("role","")).lower() or "user" in str(t.get("role","")).lower()
    )
    if not patient_text.strip():
        # fall back to full transcript
        patient_text = "\n".join(f"{t['role']}: {t['text']}" for t in turns)

    client = anthropic.Anthropic(api_key=ant_key)
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=ML_SYSTEM,
        messages=[{"role": "user", "content": f"Transcript:\n{patient_text}"}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)

# ── UI helpers ─────────────────────────────────────────────────────────────────
def metric_card(col, value, label, color):
    with col:
        st.markdown(f"""
        <div style="background:white;border:1px solid #e5e7eb;border-radius:10px;
        padding:16px;text-align:center;margin-bottom:14px;">
          <div style="font-size:22px;font-weight:800;color:{color};">{value}</div>
          <div style="font-size:11px;color:#6b7280;margin-top:2px;">{label}</div>
        </div>""", unsafe_allow_html=True)

def section(title, color=BRAND):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{color}18,{color}08);
    border:1px solid {color}44;border-radius:10px;padding:12px 16px;margin-bottom:12px;">
      <div style="font-size:12px;font-weight:700;color:#1f2937;
      text-transform:uppercase;letter-spacing:.8px;">{title}</div>
    </div>""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f2820 0%,#1e3a2f 55%,#2d5a44 100%);
border-radius:16px;padding:30px 40px;margin-bottom:22px;border-left:5px solid {BRAND};">
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="font-size:46px;">🧠</div>
    <div>
      <h1 style="color:white;font-size:28px;font-weight:800;margin:0;">
        blandHealthCallML
        <span style="color:{BRAND};">· Clinical Triage + ML Analysis</span>
      </h1>
      <p style="color:#94a3b8;font-size:13px;margin:6px 0 0 0;">
        Powered by <strong style="color:{BRAND};">Bland AI</strong> voice ·
        <strong style="color:{BRAND};">Claude</strong> NLP ·
        Symptom NER · Urgency Classification · Sentiment · Clinical Summary
      </p>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:24px;">
    {"".join([f'''<div style="background:rgba(255,255,255,0.07);border-radius:10px;
    padding:11px;text-align:center;">
      <div style="color:{BRAND};font-size:18px;font-weight:800;">{v}</div>
      <div style="color:#94a3b8;font-size:10px;margin-top:2px;">{l}</div>
    </div>''' for v,l in [
        ("Maya","Bland Voice"),("3","Intake Qs"),
        ("NER","Symptom Extract"),("3-tier","Urgency Class"),("Claude","ML Layer"),
    ]])}
  </div>
</div>
""", unsafe_allow_html=True)

# ── API key fallback inputs ────────────────────────────────────────────────────
with st.expander("🔑 API Keys", expanded=not (bland_key and anthropic_key)):
    c1, c2 = st.columns(2)
    with c1:
        if not bland_key:
            bland_key = st.text_input("Bland AI Key", type="password", placeholder="sk_...")
        else:
            st.success("✅ Bland AI key loaded")
    with c2:
        if not anthropic_key:
            anthropic_key = st.text_input("Anthropic Key", type="password", placeholder="sk-ant-...")
        else:
            st.success("✅ Anthropic key loaded")

st.markdown("---")

# ── Main layout: 2 columns ─────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Call Initiation + Transcript
# ══════════════════════════════════════════════════════════════════════════════
with col_left:

    # ── Call form ──────────────────────────────────────────────────────────────
    section("📞 Initiate Intake Call")

    phone = st.text_input("Patient Phone Number", placeholder="+1XXXXXXXXXX",
                          help="E.164 format")

    vc1, vc2 = st.columns(2)
    with vc1:
        voice = st.selectbox("Voice", ["maya","josh","paige"], index=0)
    with vc2:
        model = st.selectbox("Model", ["enhanced","base"], index=0)

    call_btn = st.button(
        "📞 Call Patient Now",
        disabled=not (phone.strip() and bland_key),
        use_container_width=True, type="primary",
    )

    # ── Status ─────────────────────────────────────────────────────────────────
    if st.session_state.call_id:
        cid    = st.session_state.call_id
        status = st.session_state.call_status or "queued"
        sc     = STATUS_COLOR.get(status, "#6b7280")
        si     = STATUS_ICON.get(status, "📞")

        st.markdown(f"""
        <div style="background:white;border:1px solid #e5e7eb;
        border-left:5px solid {sc};border-radius:10px;
        padding:14px 18px;margin:10px 0;">
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:24px;">{si}</span>
            <div>
              <div style="font-weight:800;font-size:16px;color:{sc};
              text-transform:capitalize;">{status}</div>
              <div style="font-size:11px;color:#9ca3af;margin-top:2px;">
                ID: <code>{cid[:20]}…</code>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            if st.button("🔄 Refresh Status", use_container_width=True):
                try:
                    data   = get_call(cid, bland_key)
                    status = data.get("status", status)
                    st.session_state.call_status    = status
                    st.session_state.call_log.append({
                        "time": datetime.now().strftime("%H:%M:%S"), "status": status
                    })
                    if status in ("completed","failed","no-answer"):
                        st.session_state.transcript_data = data
                    st.rerun()
                except Exception as e:
                    st.error(f"Status error: {e}")
        with rc2:
            fetch_btn = st.button("📥 Fetch Transcript", use_container_width=True,
                                  disabled=not bland_key)
            if fetch_btn:
                try:
                    data = get_call(cid, bland_key)
                    st.session_state.transcript_data = data
                    st.session_state.call_status     = data.get("status", status)
                    st.rerun()
                except Exception as e:
                    st.error(f"Fetch error: {e}")

    # ── Transcript ─────────────────────────────────────────────────────────────
    if st.session_state.transcript_data:
        turns = parse_turns(st.session_state.transcript_data)

        section("💬 Call Transcript")

        if turns:
            for t in turns:
                role     = str(t.get("role","")).lower()
                text     = t.get("text","")
                is_agent = "agent" in role or "assistant" in role or "ai" in role
                align    = "flex-start" if is_agent else "flex-end"
                bg       = f"{BRAND}22"  if is_agent else "#eff6ff"
                border   = BRAND         if is_agent else "#3b82f6"
                label    = "🤖 HealthCall AI" if is_agent else "👤 Patient"
                st.markdown(f"""
                <div style="display:flex;justify-content:{align};margin-bottom:8px;">
                  <div style="max-width:85%;background:{bg};border:1px solid {border}44;
                  border-radius:12px;padding:9px 13px;">
                    <div style="font-size:10px;font-weight:700;color:{border};
                    margin-bottom:3px;">{label}</div>
                    <div style="font-size:13px;color:#1f2937;line-height:1.5;">{html.escape(text)}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Run ML button
            st.markdown("")
            ml_btn = st.button(
                "🧠 Run ML Analysis",
                use_container_width=True, type="primary",
                disabled=not anthropic_key,
            )
            if ml_btn:
                with st.spinner("Running NER · Urgency · Sentiment · Summary…"):
                    try:
                        results = run_ml_analysis(turns, anthropic_key)
                        st.session_state.ml_results = results
                        st.rerun()
                    except json.JSONDecodeError as e:
                        st.error(f"JSON parse error: {e}")
                    except Exception as e:
                        st.error(f"ML analysis failed: {e}")
        else:
            st.info("Transcript empty — call may still be in progress.", icon="⏳")

# ── Call trigger ───────────────────────────────────────────────────────────────
if call_btn and phone.strip() and bland_key:
    with st.spinner("Triggering Bland AI outbound call…"):
        try:
            resp = start_call(phone.strip(), voice, model, bland_key)
            cid  = resp.get("call_id") or resp.get("id","")
            if not cid:
                st.error(f"No call_id in response: {resp}")
            else:
                st.session_state.call_id       = cid
                st.session_state.call_status   = "queued"
                st.session_state.call_started  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.call_log      = [{"time": datetime.now().strftime("%H:%M:%S"), "status": "queued"}]
                st.session_state.transcript_data = None
                st.session_state.ml_results      = None
                st.rerun()
        except requests.HTTPError as e:
            st.error(f"Bland API {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(f"Call failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — ML Analysis Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with col_right:
    if not st.session_state.ml_results:
        st.markdown(f"""
        <div style="background:#f8fafc;border:2px dashed #d1d5db;
        border-radius:14px;padding:80px 30px;text-align:center;
        color:#9ca3af;margin-top:4px;">
          <div style="font-size:52px;margin-bottom:14px;">🧠</div>
          <div style="font-size:16px;font-weight:600;margin-bottom:8px;">
            ML Analysis Ready
          </div>
          <div style="font-size:13px;line-height:1.6;">
            1. Initiate a call and wait for it to complete<br>
            2. Click <strong>Fetch Transcript</strong><br>
            3. Click <strong>Run ML Analysis</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        res = st.session_state.ml_results

        # ── Urgency + Sentiment KPIs ───────────────────────────────────────────
        urgency   = res.get("urgency","routine").lower()
        sentiment = res.get("sentiment","neutral").lower()
        severity  = res.get("severity_score")
        uc  = URGENCY_CONFIG.get(urgency,   URGENCY_CONFIG["routine"])
        smc = SENTIMENT_CONFIG.get(sentiment, SENTIMENT_CONFIG["neutral"])

        k1, k2, k3 = st.columns(3)
        metric_card(k1, f"{uc['icon']} {uc['label']}", "Urgency",   uc["color"])
        metric_card(k2, f"{smc['icon']} {sentiment.title()}", "Patient Sentiment", smc["color"])
        metric_card(k3, f"{severity}/10" if severity else "N/A", "Severity Score",
                    "#ef4444" if (severity or 0) >= 7 else "#f59e0b" if (severity or 0) >= 4 else "#22c55e")

        # ── Urgency banner ─────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:{uc['bg']};border:1px solid {uc['color']}55;
        border-left:5px solid {uc['color']};border-radius:10px;
        padding:14px 18px;margin-bottom:14px;">
          <div style="font-size:13px;font-weight:700;color:{uc['color']};
          margin-bottom:4px;">{uc['icon']} {uc['label'].upper()} — Urgency Classification</div>
          <div style="font-size:13px;color:#374151;">{res.get("urgency_reason","")}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Clinical Summary ───────────────────────────────────────────────────
        section("📋 Clinical Summary", BRAND)
        st.markdown(f"""
        <div style="background:white;border:1px solid #e5e7eb;border-radius:10px;
        padding:16px 18px;font-size:14px;color:#1f2937;line-height:1.7;">
          {html.escape(res.get("clinical_summary",""))}
        </div>
        """, unsafe_allow_html=True)

        # ── Recommended Action ─────────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:{BRAND}18;border:1px solid {BRAND}55;
        border-radius:10px;padding:12px 16px;margin-bottom:14px;">
          <div style="font-size:11px;font-weight:700;color:{BRAND};
          text-transform:uppercase;letter-spacing:.7px;margin-bottom:4px;">
            🏥 Recommended Action
          </div>
          <div style="font-size:13px;color:#1f2937;">{html.escape(res.get("recommended_action",""))}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Symptom NER ────────────────────────────────────────────────────────
        section("🔍 Extracted Symptoms (NER)", "#8b5cf6")
        symptoms = res.get("symptoms", [])
        if symptoms:
            st.markdown(
                " ".join([
                    f'<span style="background:#f3e8ff;color:#7c3aed;'
                    f'padding:4px 10px;border-radius:20px;font-size:13px;'
                    f'font-weight:600;margin:3px;display:inline-block;">'
                    f'🔹 {html.escape(s)}</span>'
                    for s in symptoms
                ]),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No specific symptoms extracted.")

        # ── Key Phrases ────────────────────────────────────────────────────────
        phrases = res.get("key_phrases", [])
        if phrases:
            section("💬 Key Phrases", "#f59e0b")
            st.markdown(
                " ".join([
                    f'<span style="background:#fef3c7;color:#92400e;'
                    f'padding:4px 10px;border-radius:20px;font-size:12px;'
                    f'margin:3px;display:inline-block;">'
                    f'"{html.escape(p)}"</span>'
                    for p in phrases
                ]),
                unsafe_allow_html=True,
            )

        # ── Sentiment detail ───────────────────────────────────────────────────
        section("😊 Sentiment Analysis", smc["color"])
        st.markdown(f"""
        <div style="background:white;border:1px solid #e5e7eb;
        border-left:4px solid {smc['color']};border-radius:10px;
        padding:12px 16px;font-size:13px;color:#374151;line-height:1.6;">
          <strong>{smc['icon']} {sentiment.title()}</strong> —
          {html.escape(res.get("sentiment_reasoning",""))}
        </div>
        """, unsafe_allow_html=True)

        # ── Radar chart: multi-dimension patient profile ───────────────────────
        section("📊 Patient Profile Radar", BRAND)
        sev_val     = min((severity or 5), 10)
        urgency_val = {"urgent": 9, "monitor": 5, "routine": 2}.get(urgency, 5)
        distress    = {"distressed": 9, "anxious": 7, "neutral": 5,
                       "calm": 3, "relieved": 2}.get(sentiment, 5)
        symptom_n   = min(len(symptoms) * 2, 10)
        phrase_n    = min(len(phrases), 10)

        cats = ["Severity", "Urgency", "Distress", "Symptom Count", "Phrase Density"]
        vals = [sev_val, urgency_val, distress, symptom_n, phrase_n]
        vals_closed = vals + [vals[0]]
        cats_closed = cats + [cats[0]]

        fig = go.Figure(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself",
            fillcolor=f"rgba(115,186,155,0.2)",
            line=dict(color=BRAND, width=2.5),
            marker=dict(size=6, color=BRAND),
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,10], tickfont=dict(size=9)),
                angularaxis=dict(tickfont=dict(size=11)),
            ),
            showlegend=False, height=310,
            paper_bgcolor="white",
            margin=dict(t=20, b=20, l=40, r=40),
            font=dict(family="system-ui"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Export JSON ────────────────────────────────────────────────────────
        st.download_button(
            "⬇️ Export ML Results (JSON)",
            data=json.dumps(res, indent=2),
            file_name=f"ml_analysis_{st.session_state.call_id[:8] if st.session_state.call_id else 'results'}.json",
            mime="application/json",
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:14px;font-size:12px;color:#9ca3af;">
  blandHealthCallML · Built for <strong style="color:{BRAND};">Bland AI</strong> ·
  <strong style="color:#1f2937;">Anju Vilashni Nandhakumar</strong> ·
  <a href="https://vxanju.com" style="color:{BRAND};">vxanju.com</a> ·
  <a href="https://linkedin.com/in/anju-vilashni" style="color:{BRAND};">LinkedIn</a> ·
  <a href="https://github.com/Av1352" style="color:{BRAND};">GitHub</a>
</div>
""", unsafe_allow_html=True)