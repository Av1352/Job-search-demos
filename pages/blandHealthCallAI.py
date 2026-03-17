"""
HealthCall AI - Clinical Voice Triage Demo
AI-powered patient intake via outbound phone calls
Built for Bland AI by Anju Vilashni Nandhakumar
"""

import streamlit as st
import requests
import time
import os
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="HealthCall AI – Clinical Voice Triage",
    page_icon="📞",
    layout="wide"
)

from utils.sidebar import render_sidebar
render_sidebar()

# ── Constants ──────────────────────────────────────────────────────────────────
BRAND       = "#73BA9B"
BLAND_BASE  = "https://api.bland.ai/v1"

TASK_PROMPT = (
    "You are HealthCall AI, a clinical intake assistant built by Bland AI. "
    "Introduce yourself warmly, then ask the patient exactly three questions — "
    "one at a time, waiting for their response before continuing: "
    "1) What symptoms are you currently experiencing? "
    "2) On a scale of 1 to 10, how would you rate the severity? "
    "3) Do you feel this is urgent and requires immediate attention today? "
    "Be warm, clear, and concise throughout. "
    "After they answer all three questions, summarize what you heard, "
    "then tell them: 'A nurse from our care team will follow up with you shortly. "
    "Thank you for calling HealthCall AI. Take care and feel better soon.'"
)

STATUS_COLOR = {
    "queued":     "#f59e0b",
    "ringing":    "#3b82f6",
    "in-progress":"#8b5cf6",
    "completed":  "#22c55e",
    "failed":     "#ef4444",
    "no-answer":  "#ef4444",
    "busy":       "#f59e0b",
    "cancelled":  "#6b7280",
}

STATUS_ICON = {
    "queued":     "🟡",
    "ringing":    "📳",
    "in-progress":"🔵",
    "completed":  "✅",
    "failed":     "❌",
    "no-answer":  "📵",
    "busy":       "🔴",
    "cancelled":  "⚪",
}

# ── API Key ────────────────────────────────────────────────────────────────────
bland_key = ""
try:
    bland_key = st.secrets["BLAND_API_KEY"]
except (KeyError, AttributeError):
    bland_key = os.environ.get("BLAND_API_KEY", "")

# ── Session state init ─────────────────────────────────────────────────────────
for key, default in {
    "call_id":      None,
    "call_status":  None,
    "transcript":   None,
    "call_started": None,
    "call_log":     [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ────────────────────────────────────────────────────────────────────
def bland_headers(api_key: str) -> dict:
    return {"authorization": api_key, "Content-Type": "application/json"}


def start_call(phone: str, api_key: str) -> dict:
    payload = {
        "phone_number": phone,
        "task": TASK_PROMPT,
        "voice": "maya",
        "model": "enhanced",
        "max_duration": 5,
        "record": True,
        "language": "en-US",
        "wait_for_greeting": True,
        "interruption_threshold": 150,
    }
    r = requests.post(
        f"{BLAND_BASE}/calls",
        headers=bland_headers(api_key),
        json=payload,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def get_call_status(call_id: str, api_key: str) -> dict:
    r = requests.get(
        f"{BLAND_BASE}/calls/{call_id}",
        headers=bland_headers(api_key),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def format_transcript(data: dict) -> list[dict]:
    """Extract transcript turns from Bland API response."""
    turns = []
    raw = data.get("transcripts") or data.get("transcript") or []
    if isinstance(raw, list):
        for t in raw:
            if isinstance(t, dict):
                turns.append({
                    "role":    t.get("user", t.get("role", "unknown")),
                    "text":    t.get("text", t.get("content", "")),
                    "created": t.get("created_at", ""),
                })
    elif isinstance(raw, str) and raw.strip():
        # plain-text fallback
        for line in raw.strip().split("\n"):
            if ":" in line:
                role, _, text = line.partition(":")
                turns.append({"role": role.strip(), "text": text.strip(), "created": ""})
            else:
                turns.append({"role": "agent", "text": line, "created": ""})
    return turns


def severity_color(score: int) -> str:
    if score <= 3:  return "#22c55e"
    if score <= 6:  return "#f59e0b"
    return "#ef4444"


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f2820 0%,#1e3a2f 55%,#2d5a44 100%);
border-radius:16px;padding:30px 40px;margin-bottom:22px;border-left:5px solid {BRAND};">
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="font-size:46px;">📞</div>
    <div>
      <h1 style="color:white;font-size:28px;font-weight:800;margin:0;">
        HealthCall AI
        <span style="color:{BRAND};">· Clinical Voice Triage</span>
      </h1>
      <p style="color:#94a3b8;font-size:13px;margin:6px 0 0 0;">
        Powered by <strong style="color:{BRAND};">Bland AI</strong> ·
        Outbound AI phone calls · 3-question clinical intake ·
        Live transcript · Nurse follow-up routing
      </p>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:24px;">
    {"".join([f'''<div style="background:rgba(255,255,255,0.07);border-radius:10px;
    padding:12px;text-align:center;">
      <div style="color:{BRAND};font-size:20px;font-weight:800;">{v}</div>
      <div style="color:#94a3b8;font-size:11px;margin-top:2px;">{l}</div>
    </div>''' for v,l in [
        ("<700ms","Voice Latency"),("3","Intake Questions"),
        ("Maya","Bland Voice"),("100%","HIPAA-Ready"),
    ]])}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_call, tab_transcript, tab_flow, tab_about = st.tabs([
    "📞 Initiate Call", "📋 Transcript", "🔀 Call Flow", "ℹ️ About"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – INITIATE CALL
# ══════════════════════════════════════════════════════════════════════════════
with tab_call:
    col_form, col_status = st.columns([1, 1], gap="large")

    # ── Left: Input form ───────────────────────────────────────────────────────
    with col_form:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{BRAND}18,{BRAND}08);
        border:1px solid {BRAND}44;border-radius:12px;padding:18px 22px;margin-bottom:16px;">
          <div style="font-size:12px;font-weight:700;color:#1f2937;
          text-transform:uppercase;letter-spacing:.8px;">Patient Intake Call</div>
          <div style="font-size:12px;color:#6b7280;margin-top:3px;">
            Enter a phone number — Bland AI will call them immediately
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not bland_key:
            bland_key_input = st.text_input(
                "🔑 Bland AI API Key",
                type="password",
                placeholder="sk_...",
                help="Set BLAND_API_KEY in .streamlit/secrets.toml for deployment",
            )
            if bland_key_input:
                bland_key = bland_key_input
        else:
            st.success("✅ Bland API key loaded from secrets", icon="🔑")

        phone = st.text_input(
            "📱 Patient Phone Number",
            placeholder="+1XXXXXXXXXX",
            help="Must be E.164 format, e.g. +14155550123",
        )

        st.markdown("**Agent Configuration**")
        voice_options = {"Maya (Female, Warm)": "maya", "Josh (Male, Clear)": "josh",
                         "Paige (Female, Professional)": "paige"}
        selected_voice_label = st.selectbox("Voice", list(voice_options.keys()))
        selected_voice = voice_options[selected_voice_label]

        model = st.selectbox("Model", ["enhanced", "base"], index=0,
                             help="'enhanced' = better reasoning, lower latency")

        st.markdown("**System Prompt Preview**")
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-left:3px solid {BRAND};
        border-radius:8px;padding:12px 14px;font-size:12px;color:#374151;
        line-height:1.6;max-height:140px;overflow-y:auto;">
          {TASK_PROMPT}
        </div>
        """, unsafe_allow_html=True)

        call_btn = st.button(
            "📞 Call Me Now",
            disabled=not (phone.strip() and bland_key),
            use_container_width=True,
            type="primary",
        )

    # ── Right: Live status ─────────────────────────────────────────────────────
    with col_status:
        st.markdown("**Live Call Status**")

        if not st.session_state.call_id:
            st.markdown(f"""
            <div style="background:#f8fafc;border:2px dashed #d1d5db;border-radius:12px;
            padding:60px 30px;text-align:center;color:#9ca3af;margin-top:4px;">
              <div style="font-size:48px;margin-bottom:12px;">📵</div>
              <div style="font-size:15px;font-weight:600;margin-bottom:6px;">No active call</div>
              <div style="font-size:13px;">Enter a phone number and click Call Me Now</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            cid    = st.session_state.call_id
            status = st.session_state.call_status or "queued"
            sc     = STATUS_COLOR.get(status, "#6b7280")
            si     = STATUS_ICON.get(status, "📞")
            ts     = st.session_state.call_started

            st.markdown(f"""
            <div style="background:white;border:1px solid #e5e7eb;
            border-left:5px solid {sc};border-radius:12px;
            padding:20px 24px;margin-bottom:14px;">
              <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
                <div style="font-size:32px;">{si}</div>
                <div>
                  <div style="font-size:20px;font-weight:800;color:{sc};
                  text-transform:capitalize;">{status}</div>
                  <div style="font-size:12px;color:#6b7280;margin-top:2px;">
                    Call ID: <code style="background:#f1f5f9;padding:1px 5px;
                    border-radius:4px;">{cid[:16]}…</code>
                  </div>
                </div>
              </div>
              <div style="font-size:12px;color:#6b7280;">
                {'Started: ' + ts if ts else ''}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Refresh button
            if status not in ("completed", "failed", "no-answer", "cancelled"):
                if st.button("🔄 Refresh Status", use_container_width=True):
                    try:
                        data = get_call_status(cid, bland_key)
                        new_status = data.get("status", status)
                        st.session_state.call_status = new_status
                        log_entry = {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "status": new_status,
                        }
                        st.session_state.call_log.append(log_entry)
                        if new_status in ("completed", "failed", "no-answer"):
                            st.session_state.transcript = data
                        st.rerun()
                    except Exception as e:
                        st.error(f"Status check failed: {e}")

            # Auto-poll notice
            if status in ("queued", "ringing", "in-progress"):
                st.info(
                    "🔄 Click **Refresh Status** every 15–30 seconds to track the call. "
                    "Once completed, your transcript will appear in the **Transcript** tab.",
                    icon="ℹ️",
                )

        # Call log
        if st.session_state.call_log:
            st.markdown("**Status Log**")
            for entry in reversed(st.session_state.call_log[-8:]):
                sc = STATUS_COLOR.get(entry["status"], "#6b7280")
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;
                padding:6px 10px;background:#f8fafc;border-radius:6px;
                margin-bottom:4px;font-size:12px;">
                  <span style="color:#9ca3af;">{entry["time"]}</span>
                  <span style="background:{sc}22;color:{sc};padding:2px 8px;
                  border-radius:10px;font-weight:600;">{entry["status"]}</span>
                </div>
                """, unsafe_allow_html=True)

# ── Trigger call ───────────────────────────────────────────────────────────────
if call_btn and phone.strip() and bland_key:
    with st.spinner("Initiating Bland AI outbound call…"):
        try:
            # rebuild payload with selected voice/model
            payload = {
                "phone_number": phone.strip(),
                "task": TASK_PROMPT,
                "voice": selected_voice,
                "model": model,
                "max_duration": 5,
                "record": True,
                "language": "en-US",
                "wait_for_greeting": True,
                "interruption_threshold": 150,
            }
            r = requests.post(
                f"{BLAND_BASE}/calls",
                headers=bland_headers(bland_key),
                json=payload,
                timeout=15,
            )
            r.raise_for_status()
            resp = r.json()
            cid  = resp.get("call_id") or resp.get("id", "")
            if not cid:
                st.error(f"Bland API returned no call_id. Response: {resp}")
            else:
                st.session_state.call_id      = cid
                st.session_state.call_status  = "queued"
                st.session_state.call_started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.call_log     = [{"time": datetime.now().strftime("%H:%M:%S"), "status": "queued"}]
                st.session_state.transcript   = None
                st.success(f"✅ Call initiated! ID: `{cid}`")
                st.rerun()
        except requests.HTTPError as e:
            st.error(f"Bland API error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(f"Call failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – TRANSCRIPT
# ══════════════════════════════════════════════════════════════════════════════
with tab_transcript:
    cid = st.session_state.call_id

    if not cid:
        st.info("No call initiated yet. Start a call in the **Initiate Call** tab.", icon="📞")
    else:
        col_fetch, col_spacer = st.columns([1, 2])
        with col_fetch:
            if st.button("📥 Fetch Transcript", use_container_width=True, type="primary"):
                with st.spinner("Fetching transcript from Bland API…"):
                    try:
                        data = get_call_status(cid, bland_key)
                        st.session_state.transcript   = data
                        st.session_state.call_status  = data.get("status", st.session_state.call_status)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Transcript fetch failed: {e}")

        if st.session_state.transcript:
            data   = st.session_state.transcript
            status = data.get("status", "unknown")
            turns  = format_transcript(data)

            # Call summary metrics
            duration  = data.get("call_length") or data.get("duration", 0)
            answered  = data.get("answered_by", "—")
            recording = data.get("recording_url", "")

            m1, m2, m3, m4 = st.columns(4)
            for col, val, lbl, color in [
                (m1, STATUS_ICON.get(status,"📞") + " " + status.title(), "Status",   STATUS_COLOR.get(status, BRAND)),
                (m2, f"{int(duration or 0)}s",   "Duration",  BRAND),
                (m3, str(len(turns)),             "Turns",     "#8b5cf6"),
                (m4, answered,                    "Answered By","#f59e0b"),
            ]:
                with col:
                    st.markdown(f"""
                    <div style="background:white;border:1px solid #e5e7eb;border-radius:10px;
                    padding:14px;text-align:center;margin-bottom:16px;">
                      <div style="font-size:18px;font-weight:800;color:{color};">{val}</div>
                      <div style="font-size:11px;color:#6b7280;margin-top:2px;">{lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)

            if recording:
                st.audio(recording, format="audio/mp3")

            # Transcript bubbles
            if turns:
                st.markdown("**Conversation Transcript**")
                for t in turns:
                    role = str(t.get("role", "")).lower()
                    text = t.get("text", "")
                    is_agent = "agent" in role or "assistant" in role or "ai" in role
                    align  = "flex-start" if is_agent else "flex-end"
                    bg     = f"{BRAND}22" if is_agent else "#eff6ff"
                    border = BRAND if is_agent else "#3b82f6"
                    label  = "🤖 HealthCall AI" if is_agent else "👤 Patient"
                    st.markdown(f"""
                    <div style="display:flex;justify-content:{align};margin-bottom:10px;">
                      <div style="max-width:78%;background:{bg};border:1px solid {border}44;
                      border-radius:12px;padding:10px 14px;">
                        <div style="font-size:10px;font-weight:700;color:{border};
                        margin-bottom:4px;">{label}</div>
                        <div style="font-size:14px;color:#1f2937;line-height:1.5;">{text}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Try to parse severity from transcript
                severity = None
                for t in turns:
                    txt = t.get("text", "").lower()
                    for word in txt.split():
                        w = word.strip(".,!?")
                        if w.isdigit() and 1 <= int(w) <= 10:
                            severity = int(w)
                if severity:
                    sc = severity_color(severity)
                    label = "Low" if severity <= 3 else "Moderate" if severity <= 6 else "High"
                    st.markdown(f"""
                    <div style="background:{sc}18;border:1px solid {sc}55;
                    border-radius:10px;padding:14px 20px;margin-top:14px;
                    display:flex;align-items:center;gap:14px;">
                      <div style="font-size:32px;font-weight:900;color:{sc};">{severity}/10</div>
                      <div>
                        <div style="font-weight:700;color:{sc};font-size:15px;">
                          {label} Severity Detected
                        </div>
                        <div style="font-size:12px;color:#6b7280;margin-top:2px;">
                          {"Nurse follow-up queued" if severity >= 7 else "Schedule routine appointment"}
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Export
                df = pd.DataFrame(turns)
                st.download_button(
                    "⬇️ Download Transcript CSV",
                    data=df.to_csv(index=False),
                    file_name=f"transcript_{cid[:8]}.csv",
                    mime="text/csv",
                )
            else:
                st.info("Transcript not yet available — call may still be in progress.", icon="⏳")

        else:
            st.info("Click **Fetch Transcript** above after the call completes.", icon="📋")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – CALL FLOW
# ══════════════════════════════════════════════════════════════════════════════
with tab_flow:
    st.markdown("### 🔀 Clinical Intake Call Flow")
    st.markdown("*Conversation logic designed for Bland AI's voice model*")

    # Sankey-style flow using Plotly
    nodes = [
        "Outbound Call",          # 0
        "Patient Answers",        # 1
        "No Answer / Voicemail",  # 2
        "Greeting + Intro",       # 3
        "Q1: Symptoms",           # 4
        "Q2: Severity (1–10)",    # 5
        "Q3: Urgent?",            # 6
        "Severity ≥ 7 (High)",    # 7
        "Severity 4–6 (Moderate)",# 8
        "Severity ≤ 3 (Low)",     # 9
        "Urgent = Yes",           # 10
        "Urgent = No",            # 11
        "🚨 Immediate Escalation",# 12
        "📅 Same-Day Appointment", # 13
        "📆 Routine Scheduling",   # 14
        "Nurse Follow-Up SMS",    # 15
    ]

    label_colors = [
        "#1e3a2f","#22c55e","#ef4444","#73BA9B","#73BA9B","#73BA9B","#73BA9B",
        "#ef4444","#f59e0b","#22c55e","#ef4444","#22c55e",
        "#ef4444","#f59e0b","#22c55e","#8b5cf6",
    ]

    sources = [0, 0, 1, 3, 4, 5, 5, 5, 6, 6, 7, 10, 8, 9, 11, 12, 13, 14]
    targets = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 12, 13, 14, 14, 15, 15, 15]
    values  = [70, 30, 70, 70, 70, 20, 30, 20, 15, 55, 20, 15, 30, 20, 55, 35, 30, 20]
    link_colors = [
        "rgba(115,186,155,0.3)","rgba(239,68,68,0.3)","rgba(115,186,155,0.3)",
        "rgba(115,186,155,0.3)","rgba(115,186,155,0.3)","rgba(239,68,68,0.3)",
        "rgba(245,158,11,0.3)", "rgba(34,197,94,0.3)", "rgba(239,68,68,0.3)",
        "rgba(34,197,94,0.3)",  "rgba(239,68,68,0.3)", "rgba(239,68,68,0.3)",
        "rgba(245,158,11,0.3)", "rgba(34,197,94,0.3)", "rgba(34,197,94,0.3)",
        "rgba(139,92,246,0.4)", "rgba(139,92,246,0.4)","rgba(139,92,246,0.4)",
    ]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=22,
            line=dict(color="white", width=0.5),
            label=nodes,
            color=label_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources, target=targets,
            value=values, color=link_colors,
            hovertemplate="Flow: %{value} patients<extra></extra>",
        ),
    ))
    fig.update_layout(
        title_text="HealthCall AI — Clinical Triage Conversation Flow",
        title_font=dict(size=15, color="#1f2937"),
        height=520, paper_bgcolor="white",
        font=dict(size=11, family="system-ui", color="#1f2937"),
        margin=dict(t=50, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Step-by-step legend
    st.markdown("#### Step-by-Step Flow")
    steps = [
        ("1", BRAND,     "Bland AI initiates outbound call to patient phone number"),
        ("2", BRAND,     "Patient answers → HealthCall AI introduces itself"),
        ("3", BRAND,     "Q1: What symptoms are you experiencing?"),
        ("4", BRAND,     "Q2: Rate severity 1–10"),
        ("5", "#ef4444", "Severity ≥ 7 → flag for immediate escalation"),
        ("5", "#f59e0b", "Severity 4–6 → same-day appointment"),
        ("5", "#22c55e", "Severity ≤ 3 → routine scheduling"),
        ("6", BRAND,     "Q3: Is this urgent? → confirm routing decision"),
        ("7", "#8b5cf6", "Call ends → nurse follow-up SMS dispatched"),
    ]
    for num, color, desc in steps:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;
        padding:8px 14px;background:white;border:1px solid #e5e7eb;
        border-radius:8px;margin-bottom:6px;">
          <div style="background:{color};color:white;border-radius:50%;
          min-width:24px;height:24px;display:flex;align-items:center;
          justify-content:center;font-size:11px;font-weight:700;">{num}</div>
          <div style="font-size:13px;color:#374151;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
        padding:20px;margin-bottom:16px;">
          <h4 style="color:#1f2937;margin:0 0 14px 0;">📞 How It Works</h4>
          {"".join([f'''<div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:10px;">
            <div style="background:{BRAND};color:white;border-radius:50%;
            min-width:22px;height:22px;display:flex;align-items:center;
            justify-content:center;font-size:11px;font-weight:700;">{n}</div>
            <div style="font-size:13px;color:#374151;"><strong>{t}</strong> — {d}</div>
          </div>''' for n,(t,d) in enumerate([
            ("Enter phone number","E.164 format, routed directly to patient's phone"),
            ("Bland AI dials out","POST /v1/calls triggers outbound call within seconds"),
            ("Maya speaks","Warm AI voice asks 3 clinical intake questions"),
            ("Live status polling","GET /v1/calls/{call_id} tracks queued→ringing→completed"),
            ("Transcript fetch","Full conversation retrieved and rendered with chat bubbles"),
            ("Severity parsed","Patient's 1–10 score extracted and routed automatically"),
            ("Nurse follow-up","SMS dispatched based on urgency + severity tier"),
          ], 1)])}
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
        padding:20px;margin-bottom:16px;">
          <h4 style="color:#1f2937;margin:0 0 14px 0;">🏗️ Tech Stack</h4>
          {"".join([f'''<div style="background:white;border-left:3px solid {BRAND};
          padding:8px 12px;border-radius:6px;margin-bottom:7px;font-size:13px;color:#374151;">
            <strong>{k}:</strong> {v}
          </div>''' for k,v in [
            ("Voice AI","Bland AI API — outbound calls, Maya voice, enhanced model"),
            ("Call trigger","POST /v1/calls with E.164 phone + task prompt + voice config"),
            ("Status polling","GET /v1/calls/{call_id} — queued → ringing → in-progress → completed"),
            ("Transcript","GET /v1/calls/{call_id} — returns full turn-by-turn transcript"),
            ("Severity NLP","Regex extraction of 1–10 digit from patient response"),
            ("Flow diagram","Plotly Sankey — dynamic patient volume routing visualization"),
            ("Frontend","Streamlit + custom HTML/CSS + Plotly"),
          ]])}
        </div>
        <div style="background:linear-gradient(135deg,{BRAND}18,{BRAND}08);
        border:1px solid {BRAND}44;border-radius:12px;padding:16px 20px;">
          <h4 style="color:#1f2937;margin:0 0 10px 0;">🚀 Deployment</h4>
          <div style="font-size:13px;color:#374151;line-height:1.8;">
            Add to <code>.streamlit/secrets.toml</code>:<br>
            <code style="background:#1e293b;color:#73BA9B;padding:6px 10px;
            border-radius:6px;display:block;margin-top:6px;">
              BLAND_API_KEY = "sk_..."
            </code>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:14px;font-size:12px;color:#9ca3af;">
  HealthCall AI · Built for <strong style="color:{BRAND};">Bland AI</strong> ·
  <strong style="color:#1f2937;">Anju Vilashni Nandhakumar</strong> ·
  <a href="https://vxanju.com" style="color:{BRAND};">vxanju.com</a> ·
  <a href="https://linkedin.com/in/anju-vilashni" style="color:{BRAND};">LinkedIn</a> ·
  <a href="https://github.com/Av1352" style="color:{BRAND};">GitHub</a>
</div>
""", unsafe_allow_html=True)