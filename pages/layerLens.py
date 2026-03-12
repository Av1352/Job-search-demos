"""
LayerLens - Clinical Chart Abstractor
AI-powered structured data extraction from clinical notes
Built by Anju Vilashni Nandhakumar
"""

import streamlit as st
import anthropic
import json
import re
import html
import os
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="LayerLens – Clinical Chart Abstractor",
    page_icon="🔬",
    layout="wide"
)

from utils.sidebar import render_sidebar
render_sidebar()

# ── Constants ──────────────────────────────────────────────────────────────────
BRAND = "#73BA9B"
HIGHLIGHT_PALETTE = [
    "#fef08a", "#bbf7d0", "#bfdbfe", "#fecaca",
    "#fed7aa", "#e9d5ff", "#fce7f3", "#cffafe",
    "#d9f99d", "#fde68a", "#f5d0fe", "#a7f3d0",
]

# ── Sample Clinical Notes ──────────────────────────────────────────────────────
SAMPLE_NOTES = {
    "— Select a sample note —": "",

    "🫀 Cardiology – Acute Inferior STEMI": """\
ADMISSION NOTE – CARDIOLOGY
Date: 2024-11-15  |  MRN: 1042837  |  Attending: Dr. Sarah Chen, MD

Chief Complaint: Chest pain, onset 2 hours prior to arrival.

HPI: Mr. James Whitfield is a 58-year-old male with a history of hypertension, \
hyperlipidemia, and 30 pack-year smoking history who presents with sudden-onset, \
substernal chest pressure radiating to the left arm, associated with diaphoresis \
and mild dyspnea. Symptoms began at approximately 08:15 while at rest. Patient \
denies prior similar episodes. EMS administered aspirin 324 mg en route.

PMH: Hypertension (diagnosed 2018), Hyperlipidemia (diagnosed 2019), Former smoker (quit 2021).

Medications on Admission: Lisinopril 10 mg daily, Atorvastatin 40 mg nightly, Aspirin 81 mg daily.

Allergies: Penicillin (rash).

Vitals: BP 148/92 mmHg, HR 96 bpm, RR 18, SpO2 97% on RA, Temp 98.4°F.

Physical Exam: Alert and oriented. Diaphoretic. Cardiovascular: S1/S2 regular, no murmurs. \
Respiratory: CTA bilaterally.

Labs / Diagnostics:
- Troponin I: 2.8 ng/mL (H) at 09:45
- BNP: 340 pg/mL (H)
- CBC: WBC 11.2, Hgb 13.8, Plt 245
- BMP: Na 139, K 4.1, Cr 1.0, Glucose 142
- EKG: ST elevation in leads II, III, aVF consistent with inferior STEMI
- Chest X-ray: No acute cardiopulmonary process

Assessment: 58-year-old male presenting with inferior STEMI. Patient taken emergently to \
cardiac catheterization lab.

Procedure: Emergent PCI with drug-eluting stent placement to RCA (3.5 × 28 mm). \
TIMI 3 flow achieved post-intervention. Door-to-balloon time: 47 minutes.

Discharge Medications: Aspirin 81 mg daily, Ticagrelor 90 mg BID, Metoprolol succinate \
25 mg daily, Atorvastatin 80 mg nightly, Lisinopril 10 mg daily.

Disposition: Transferred to cardiac ICU. Estimated LOS 3–4 days.
Follow-up: Cardiac rehab referral placed. Outpatient cardiology in 2 weeks.\
""",

    "🩺 Endocrinology – T2DM Management": """\
OUTPATIENT VISIT NOTE – ENDOCRINOLOGY
Date: 2024-11-20  |  MRN: 2093841  |  Provider: Dr. Amir Patel, MD

Patient: Maria Santos, 52-year-old female
Insurance: BlueCross PPO

Reason for Visit: Diabetes management follow-up, 3-month visit.

Interval History: Patient reports improved dietary adherence. Fasting blood glucose \
averaging 145–165 mg/dL in the morning. Exercises 3×/week (30-minute walks). Denies \
hypoglycemic episodes. Reports occasional blurry vision and right foot numbness × 6 weeks.

Chronic Conditions: Type 2 Diabetes Mellitus (dx 2016), Hypertension (dx 2014), \
Obesity (BMI 32.4), Mild CKD Stage 2 (eGFR 68).

Current Medications:
- Metformin 1000 mg BID
- Empagliflozin 10 mg daily (started 3 months ago)
- Lisinopril 20 mg daily
- Amlodipine 5 mg daily

Vitals: BP 134/82 mmHg, HR 78 bpm, Weight 186 lbs, BMI 32.4.

Labs (drawn today):
- HbA1c: 7.8% (prior 8.4% three months ago)
- Fasting glucose: 152 mg/dL
- eGFR: 68 mL/min/1.73m²
- Urine ACR: 42 mg/g (microalbuminuria)
- LDL: 88 mg/dL
- TSH: 2.1 mIU/L (normal)

Foot Exam: Monofilament testing shows decreased sensation plantar surface right foot. \
No ulcers. Pedal pulses intact bilaterally.

Ophthalmology: Last dilated eye exam 14 months ago – patient delinquent on annual screening.

Assessment & Plan:
1. T2DM – Improved, HbA1c 7.8% (down from 8.4%). Target HbA1c <7.5%. Continue regimen.
2. Hypertension – Suboptimal. Increase Amlodipine to 10 mg daily.
3. CKD Stage 2 – Stable. Monitor eGFR q6 months.
4. Peripheral neuropathy – New. Added Gabapentin 300 mg QHS. Podiatry referral placed.
5. Preventive care: Flu vaccine administered today. Ophthalmology referral placed.

Follow-up: Return in 3 months. Labs prior to visit.\
""",

    "🫁 Oncology – NSCLC Staging": """\
MULTIDISCIPLINARY TUMOR BOARD NOTE
Date: 2024-11-18  |  MRN: 3847291  |  Facility: University Medical Center

Patient: Robert Kim, 67-year-old male
Referring Provider: Dr. Lisa Thompson, Pulmonology

Referral Reason: Newly diagnosed lung mass, staging and treatment planning.

HPI: 67-year-old male with 45 pack-year smoking history (currently active smoker). \
3-month history of progressive dyspnea, 15 lb unintentional weight loss, and persistent \
cough with hemoptysis × 3 weeks. CT chest revealed a 4.2 cm right upper lobe mass with \
mediastinal lymphadenopathy.

Workup:
- CT Chest/Abdomen/Pelvis (2024-11-05): 4.2 × 3.8 cm spiculated RUL mass, ipsilateral \
hilar and mediastinal lymphadenopathy (levels 4R, 7). No distant metastases.
- PET-CT (2024-11-10): Hypermetabolic RUL mass (SUVmax 12.4), hypermetabolic mediastinal nodes. \
No distant metastatic disease.
- Bronchoscopy with EBUS (2024-11-14): Biopsy of RUL mass and station 4R lymph node.
- Pathology (2024-11-17): Non-small cell lung carcinoma, adenocarcinoma histology. \
PD-L1 TPS: 65%. KRAS G12C mutation detected. ALK negative, EGFR negative, ROS1 negative.
- Brain MRI (2024-11-16): No intracranial metastases.
- Pulmonary Function Tests: FEV1 62% predicted, DLCO 58% predicted.

Performance Status: ECOG 1.

Staging: Stage IIIA NSCLC (T2bN2M0) per AJCC 8th Edition.

PMH: COPD (GOLD Stage 2), Hypertension, Type 2 Diabetes (on Metformin).

Tumor Board Recommendation: Concurrent chemoradiation (carboplatin/paclitaxel × 2 cycles \
+ 60 Gy IMRT) followed by durvalumab consolidation immunotherapy × 12 months given \
PD-L1 ≥50%. KRAS G12C mutation not amenable to targeted therapy at this stage. \
Enrollment in PACIFIC-R registry recommended.

Follow-up: Oncology in 1 week to initiate treatment planning.\
""",
}

# ── Task Definitions ───────────────────────────────────────────────────────────
TASKS = {
    "Registry Abstraction": {
        "icon": "📋",
        "desc": "Extract fields for clinical registries (cardiac, cancer, stroke). "
                "Captures diagnoses, procedures, outcomes, and quality indicators.",
        "fields": "patient age and sex, primary diagnosis with ICD-10 code, "
                  "admission date, key procedures with CPT codes, attending provider, "
                  "discharge disposition, relevant lab values (troponin, HbA1c, etc.), "
                  "discharge medications, follow-up plan, comorbidities",
    },
    "Quality Measurement": {
        "icon": "📊",
        "desc": "Extract numerator/denominator elements for quality measures "
                "(HEDIS, Joint Commission, CMS). Identifies care gaps.",
        "fields": "blood pressure reading, HbA1c value and trend, "
                  "preventive screenings completed (with dates), medication classes prescribed, "
                  "smoking status, BMI, vaccination status, nephrology/ophthalmology referrals, "
                  "patient education documented, follow-up appointment scheduled",
    },
    "Research Cohort": {
        "icon": "🔬",
        "desc": "Extract inclusion/exclusion criteria elements for clinical trial "
                "or retrospective cohort identification.",
        "fields": "age, sex, primary diagnosis, ECOG or functional status, "
                  "key biomarkers (mutation status, PD-L1, SUVmax, etc.), "
                  "comorbidities as potential exclusion factors, prior treatments, "
                  "disease stage or severity, key lab values, imaging findings",
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def build_prompt(note: str, task: str) -> str:
    cfg = TASKS[task]
    return f"""You are an expert clinical data abstractor. Extract structured fields \
from the clinical note for: {task}.

Focus on: {cfg["fields"]}.

RULES:
1. source_text = EXACT verbatim copy from the note (no paraphrase).
2. confidence: 1.0 = explicitly stated · 0.70–0.89 = clearly implied · <0.70 = inferred or absent.
3. Extract 8–14 fields. Never invent data not present.
4. If absent: value = "Not documented", confidence = 0.30, source_text = "".

Return ONLY valid JSON — no markdown fences, no preamble:
{{
  "fields": [
    {{
      "field_name": "...",
      "value": "...",
      "confidence": 0.95,
      "source_text": "exact quote"
    }}
  ],
  "summary": "2-sentence abstraction summary"
}}

CLINICAL NOTE:
{note}"""


def call_claude(note: str, task: str, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2500,
        messages=[{"role": "user", "content": build_prompt(note, task)}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def highlight_note(note_text: str, fields: list) -> str:
    """Return HTML string with source spans highlighted."""
    spans = []
    for i, f in enumerate(fields):
        src = f.get("source_text", "").strip()
        if not src or f.get("value") == "Not documented":
            continue
        for m in re.finditer(re.escape(src), note_text, re.IGNORECASE):
            spans.append((m.start(), m.end(), i, f["field_name"]))

    if not spans:
        return html.escape(note_text).replace("\n", "<br>")

    spans.sort(key=lambda x: x[0])
    # Remove overlaps — keep first occurrence
    merged, last_end = [], -1
    for s in spans:
        if s[0] >= last_end:
            merged.append(s)
            last_end = s[1]

    parts, prev = [], 0
    for start, end, idx, name in merged:
        parts.append(html.escape(note_text[prev:start]))
        color = HIGHLIGHT_PALETTE[idx % len(HIGHLIGHT_PALETTE)]
        parts.append(
            f'<mark style="background:{color};padding:1px 4px;border-radius:3px;'
            f'cursor:help;" title="{html.escape(name)}">'
            f"{html.escape(note_text[start:end])}</mark>"
        )
        prev = end
    parts.append(html.escape(note_text[prev:]))
    return "".join(parts).replace("\n", "<br>")


def conf_color(c: float) -> str:
    return "#22c55e" if c >= 0.9 else "#f59e0b" if c >= 0.7 else "#ef4444"


def conf_label(c: float) -> str:
    return "High" if c >= 0.9 else "Medium" if c >= 0.7 else "Low"


# ── API Key ────────────────────────────────────────────────────────────────────
api_key = ""
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except (KeyError, AttributeError):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f2820 0%,#1e3a2f 55%,#2d5a44 100%);
border-radius:16px;padding:30px 40px;margin-bottom:22px;border-left:5px solid {BRAND};">
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="font-size:46px;">🔬</div>
    <div>
      <h1 style="color:white;font-size:28px;font-weight:800;margin:0;">
        LayerLens
        <span style="color:{BRAND};">Clinical Chart Abstractor</span>
      </h1>
      <p style="color:#94a3b8;font-size:13px;margin:6px 0 0 0;">
        AI-powered structured extraction · source-linked evidence ·
        Registry · Quality Measurement · Research Cohort
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── API key fallback UI ────────────────────────────────────────────────────────
if not api_key:
    api_key = st.text_input(
        "🔑 Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Set ANTHROPIC_API_KEY in .streamlit/secrets.toml for deployment",
    )
    if not api_key:
        st.info(
            "Enter your Anthropic API key above, or set `ANTHROPIC_API_KEY` in "
            "`.streamlit/secrets.toml` for Streamlit Cloud deployment.",
            icon="ℹ️",
        )

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_extract, tab_analytics, tab_howto = st.tabs(
    ["🔍 Extract", "📊 Analytics", "ℹ️ How It Works"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – EXTRACT
# ══════════════════════════════════════════════════════════════════════════════
with tab_extract:
    col_in, col_out = st.columns([1, 1], gap="large")

    # ── Left: Input ────────────────────────────────────────────────────────────
    with col_in:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{BRAND}18,{BRAND}08);
        border:1px solid {BRAND}44;border-radius:12px;padding:16px 20px;margin-bottom:14px;">
          <div style="font-size:12px;font-weight:700;color:#1f2937;
          text-transform:uppercase;letter-spacing:.8px;">Clinical Note</div>
          <div style="font-size:12px;color:#6b7280;margin-top:2px;">
            Paste a note or choose a sample below
          </div>
        </div>
        """, unsafe_allow_html=True)

        sample_pick = st.selectbox(
            "Sample", list(SAMPLE_NOTES.keys()), label_visibility="collapsed"
        )
        note_input = st.text_area(
            "note",
            value=SAMPLE_NOTES[sample_pick],
            height=330,
            placeholder="Paste clinical note here…",
            label_visibility="collapsed",
        )

        st.markdown("**Abstraction Task**")
        selected_task = st.radio(
            "task", list(TASKS.keys()), horizontal=True, label_visibility="collapsed"
        )
        tcfg = TASKS[selected_task]
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;
        border-left:3px solid {BRAND};border-radius:8px;
        padding:10px 14px;margin:6px 0 14px 0;font-size:12px;color:#475569;">
          {tcfg["icon"]} <strong>{selected_task}</strong> — {tcfg["desc"]}
        </div>
        """, unsafe_allow_html=True)

        extract_btn = st.button(
            "⚡ Extract Structured Fields",
            disabled=not (note_input.strip() and api_key),
            use_container_width=True,
            type="primary",
        )

    # ── Right: Results ─────────────────────────────────────────────────────────
    with col_out:
        if "result" not in st.session_state:
            st.markdown(f"""
            <div style="background:#f8fafc;border:2px dashed #d1d5db;border-radius:12px;
            padding:70px 30px;text-align:center;color:#9ca3af;margin-top:54px;">
              <div style="font-size:48px;margin-bottom:12px;">🩺</div>
              <div style="font-size:16px;font-weight:600;margin-bottom:6px;">
                Ready to abstract
              </div>
              <div style="font-size:13px;">
                Select a note and task, then click Extract
              </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            res   = st.session_state.result
            note_ = st.session_state.note_used
            flds  = res.get("fields", [])

            # Summary banner
            if res.get("summary"):
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,{BRAND}22,{BRAND}10);
                border:1px solid {BRAND}55;border-radius:10px;
                padding:12px 16px;margin-bottom:12px;font-size:13px;color:#1f2937;">
                  <strong style="color:{BRAND};">📝 Summary:</strong>
                  {html.escape(res["summary"])}
                </div>
                """, unsafe_allow_html=True)

            # Highlighted note
            with st.expander("📄 Highlighted Source Note", expanded=True):
                hl = highlight_note(note_, flds)
                st.markdown(f"""
                <div style="background:white;border:1px solid #e2e8f0;border-radius:10px;
                padding:14px 16px;font-size:12.5px;line-height:1.85;color:#1e293b;
                max-height:300px;overflow-y:auto;font-family:'Courier New',monospace;">
                  {hl}
                </div>
                <div style="margin-top:6px;font-size:11px;color:#9ca3af;">
                  💡 Hover any highlight to see the field name
                </div>
                """, unsafe_allow_html=True)

            # Field cards
            st.markdown(
                f"**Extracted Fields** "
                f"<span style='color:#6b7280;font-size:13px;'>({len(flds)} total)</span>",
                unsafe_allow_html=True,
            )
            for i, f in enumerate(flds):
                conf  = float(f.get("confidence", 0))
                cc    = conf_color(conf)
                cl    = conf_label(conf)
                hc    = HIGHLIGHT_PALETTE[i % len(HIGHLIGHT_PALETTE)]
                src   = f.get("source_text", "").strip()
                val   = str(f.get("value", ""))
                fname = f.get("field_name", "")
                absent = val == "Not documented"
                src_snippet = (
                    f'<div style="font-size:11px;color:#9ca3af;margin-top:5px;'
                    f'font-style:italic;line-height:1.4;">'
                    f'"{html.escape(src[:130])}{"…" if len(src)>130 else ""}"</div>'
                    if src and not absent else ""
                )
                st.markdown(f"""
                <div style="background:white;border:1px solid #e5e7eb;
                border-left:4px solid {hc};border-radius:10px;
                padding:11px 15px;margin-bottom:7px;">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div style="flex:1;min-width:0;">
                      <div style="font-size:10.5px;font-weight:700;color:#6b7280;
                      text-transform:uppercase;letter-spacing:.6px;margin-bottom:3px;">
                        {html.escape(fname)}
                      </div>
                      <div style="font-size:14px;font-weight:600;
                      color:{'#9ca3af' if absent else '#1f2937'};">
                        {html.escape(val)}
                      </div>
                      {src_snippet}
                    </div>
                    <div style="text-align:center;margin-left:14px;flex-shrink:0;">
                      <div style="font-size:17px;font-weight:800;color:{cc};">
                        {int(conf*100)}%
                      </div>
                      <div style="font-size:10px;color:{cc};font-weight:600;">{cl}</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    if "result" not in st.session_state:
        st.info("Run an extraction first to see analytics.", icon="📊")
    else:
        flds = st.session_state.result.get("fields", [])
        task_used = st.session_state.get("task_used", "")

        # KPI row
        documented = sum(1 for f in flds if f.get("value") != "Not documented")
        avg_conf   = sum(f["confidence"] for f in flds) / len(flds) if flds else 0
        sourced    = sum(1 for f in flds if f.get("source_text", "").strip())

        k1, k2, k3, k4 = st.columns(4)
        for col, val, lbl, color in [
            (k1, len(flds),           "Total Fields",    BRAND),
            (k2, documented,          "Documented",      "#22c55e"),
            (k3, f"{int(avg_conf*100)}%", "Avg Confidence", "#f59e0b"),
            (k4, sourced,             "Source-Linked",   "#8b5cf6"),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:white;border:1px solid #e5e7eb;border-radius:10px;
                padding:18px;text-align:center;margin-bottom:16px;">
                  <div style="font-size:28px;font-weight:800;color:{color};">{val}</div>
                  <div style="font-size:12px;color:#6b7280;margin-top:3px;">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        col_bar, col_pie = st.columns(2)

        with col_bar:
            fig_bar = go.Figure(go.Bar(
                x=[f["field_name"] for f in flds],
                y=[f["confidence"] for f in flds],
                marker_color=[conf_color(f["confidence"]) for f in flds],
                text=[f"{int(f['confidence']*100)}%" for f in flds],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title=f"Field Confidence — {task_used}",
                xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
                yaxis=dict(range=[0, 1.18], title="Confidence", tickformat=".0%"),
                height=380, plot_bgcolor="#f8fafc", paper_bgcolor="white",
                margin=dict(t=50, b=100),
                font=dict(family="system-ui"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_pie:
            tiers = {"High ≥90%": 0, "Medium 70–89%": 0, "Low <70%": 0}
            for f in flds:
                c = f["confidence"]
                if c >= 0.9:   tiers["High ≥90%"] += 1
                elif c >= 0.7: tiers["Medium 70–89%"] += 1
                else:          tiers["Low <70%"] += 1

            fig_pie = go.Figure(go.Pie(
                labels=list(tiers.keys()),
                values=list(tiers.values()),
                marker_colors=["#22c55e", "#f59e0b", "#ef4444"],
                hole=0.44,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(
                title="Confidence Distribution",
                height=380, paper_bgcolor="white",
                font=dict(family="system-ui"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Full table
        st.markdown("#### Field-Level Detail")
        rows = []
        for f in flds:
            rows.append({
                "Field": f["field_name"],
                "Value": f["value"],
                "Confidence": f"{int(float(f['confidence'])*100)}%",
                "Tier": conf_label(float(f["confidence"])),
                "Source Text": (f.get("source_text") or "—")[:80],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_howto:
    col_a, col_b = st.columns(2)

    with col_a:
        steps = [
            ("Note ingestion",      "Raw clinical text accepted in any format — EHR export, dictation, or typed note"),
            ("Task-aware prompting","Claude receives a task-specific field schema: Registry, Quality Measure, or Research Cohort"),
            ("Structured extraction","LLM returns JSON with field name, value, confidence score, and exact source quote"),
            ("Span matching",       "Source text located via regex in original note; highlighted with unique color per field"),
            ("Confidence scoring",  "1.0 = explicitly stated · 0.7–0.89 = implied · <0.7 = inferred or absent"),
        ]
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
        padding:20px;margin-bottom:16px;">
          <h4 style="color:#1f2937;margin:0 0 14px 0;">🔄 Extraction Pipeline</h4>
          {"".join([f'''<div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:10px;">
            <div style="background:{BRAND};color:white;border-radius:50%;
            min-width:22px;height:22px;display:flex;align-items:center;
            justify-content:center;font-size:11px;font-weight:700;">{n+1}</div>
            <div style="font-size:13px;color:#374151;">
              <strong>{t}</strong> — {d}
            </div>
          </div>''' for n,(t,d) in enumerate(steps)])}
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
        padding:20px;margin-bottom:16px;">
          <h4 style="color:#1f2937;margin:0 0 14px 0;">📋 Abstraction Tasks</h4>
          {"".join([f'''<div style="background:white;border-left:3px solid {BRAND};
          padding:10px 12px;border-radius:6px;margin-bottom:8px;">
            <div style="font-weight:700;font-size:13px;color:#1f2937;">
              {cfg["icon"]} {task}
            </div>
            <div style="font-size:12px;color:#6b7280;margin-top:3px;">{cfg["desc"]}</div>
          </div>''' for task, cfg in TASKS.items()])}
        </div>

        <div style="background:linear-gradient(135deg,{BRAND}18,{BRAND}08);
        border:1px solid {BRAND}44;border-radius:12px;padding:16px 20px;">
          <h4 style="color:#1f2937;margin:0 0 10px 0;">🏗️ Tech Stack</h4>
          <div style="font-size:13px;color:#374151;line-height:1.8;">
            <strong>AI:</strong> Claude claude-opus-4-6 (Anthropic) · structured JSON extraction<br>
            <strong>Span matching:</strong> Python regex · case-insensitive verbatim linking<br>
            <strong>Frontend:</strong> Streamlit · Plotly · custom HTML/CSS<br>
            <strong>Deployment:</strong> Streamlit Cloud · <code>secrets.toml</code> key management
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:14px;font-size:12px;color:#9ca3af;">
  LayerLens · Clinical Chart Abstractor · Built by
  <strong style="color:#1f2937;">Anju Vilashni Nandhakumar</strong> ·
  <a href="https://vxanju.com" style="color:{BRAND};">vxanju.com</a> ·
  <a href="https://linkedin.com/in/anju-vilashni" style="color:{BRAND};">LinkedIn</a> ·
  <a href="https://github.com/Av1352" style="color:{BRAND};">GitHub</a>
</div>
""", unsafe_allow_html=True)

# ── Extract trigger (runs after all widget definitions) ───────────────────────
if extract_btn and note_input.strip() and api_key:
    with st.spinner(f"Abstracting fields for {selected_task}…"):
        try:
            result = call_claude(note_input, selected_task, api_key)
            st.session_state.result    = result
            st.session_state.note_used = note_input
            st.session_state.task_used = selected_task
            st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"Could not parse Claude's response as JSON — try again. ({e})")
        except anthropic.AuthenticationError:
            st.error("Invalid API key. Check your Anthropic API key and try again.")
        except anthropic.APIError as e:
            st.error(f"Anthropic API error: {e}")
        except Exception as e:
            st.error(f"Extraction failed: {e}")