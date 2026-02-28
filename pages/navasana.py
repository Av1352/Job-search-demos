"""
Navasana - AI Cyber Risk Underwriting Engine
AI-native platform for cyber insurance risk quantification
Built for Navasana by Anju Vilashni Nandhakumar
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="Navasana - Cyber Risk AI", page_icon="🛡️", layout="wide")

from utils.sidebar import render_sidebar
render_sidebar()

# ── Brand ──────────────────────────────────────────────────────────────────────
BRAND = "#73BA9B"
DANGER = "#ef4444"
WARN   = "#f59e0b"
OK     = "#22c55e"

def card(color, label, value, sub=""):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{color}18,{color}08);border:1px solid {color}44;
    border-radius:12px;padding:18px 20px;text-align:center;">
        <div style="color:{color};font-size:28px;font-weight:800;">{value}</div>
        <div style="color:#1f2937;font-size:13px;font-weight:600;margin-top:4px;">{label}</div>
        {"<div style='color:#6b7280;font-size:11px;margin-top:3px;'>"+sub+"</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 60%,#0f2820 100%);
border-radius:16px;padding:36px 40px;margin-bottom:28px;
border-left:5px solid {BRAND};">
    <div style="display:flex;align-items:center;gap:16px;">
        <div style="font-size:48px;">🛡️</div>
        <div>
            <h1 style="color:white;font-size:32px;font-weight:800;margin:0;">
                Navasana <span style="color:{BRAND};">Cyber Risk Engine</span>
            </h1>
            <p style="color:#94a3b8;font-size:15px;margin:6px 0 0 0;">
                AI-native underwriting platform · Continuous risk assessment · Real-time threat intelligence
            </p>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:28px;">
        {"".join([f'''<div style="background:rgba(255,255,255,0.07);border-radius:10px;padding:14px;text-align:center;">
            <div style="color:{BRAND};font-size:22px;font-weight:800;">{v}</div>
            <div style="color:#94a3b8;font-size:11px;margin-top:2px;">{l}</div>
        </div>''' for v, l in [("94.7%","Underwriting Accuracy"),("2.3s","Avg Assessment Time"),("$2.1M","Avg Loss Prevention"),("10,000+","Policies Assessed")]])}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Risk Assessment", "📊 Underwriting Analytics", "🌐 Threat Intelligence", "⚙️ Model Architecture"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{BRAND}18,{BRAND}08);border:1px solid {BRAND}44;
    border-radius:12px;padding:20px 24px;margin-bottom:20px;">
        <h3 style="color:#1f2937;margin:0 0 6px 0;">🔍 Automated Security Posture Assessment</h3>
        <p style="color:#6b7280;font-size:13px;margin:0;">
            Mirrors Navasana's API-driven intake: CrowdStrike · Okta · AWS SecurityHub signals
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_score = st.columns([1, 1], gap="large")

    SECURITY_CONTROLS = {
        "Multi-Factor Authentication": 15,
        "Patch Management (≤30 days)": 12,
        "Encrypted Offsite Backups": 12,
        "EDR Platform (CrowdStrike/SentinelOne)": 10,
        "SOC 2 Compliance": 10,
        "Security Awareness Training": 8,
        "Incident Response Plan": 10,
        "Vendor Risk Management": 8,
        "Data Loss Prevention (DLP)": 8,
        "Identity Provider (Okta/AAD)": 7,
    }

    INDUSTRIES = {
        "Healthcare": 1.45, "Finance / Banking": 1.40, "Retail / E-Commerce": 1.25,
        "Technology / SaaS": 1.20, "Manufacturing": 1.15, "Education": 1.10,
        "Professional Services": 1.18, "Government": 1.35, "Other": 1.0,
    }

    with col_form:
        st.markdown("**Company Profile**")
        industry = st.selectbox("Industry Vertical", list(INDUSTRIES.keys()), index=3)
        emp_band = st.selectbox("Employee Count", ["1–50 (SMB)", "51–250 (Mid-Market)", "251–1,000 (Growth)", "1,001+ (Enterprise)"], index=1)
        revenue = st.slider("Annual Revenue ($M)", 0.5, 200.0, 15.0, step=0.5)
        prior_breaches = st.selectbox("Prior Breaches (last 3 yrs)", [0, 1, 2, "3+"], index=0)

        st.markdown("---")
        st.markdown("**Security Controls Active**")
        checks = {}
        for ctrl, pts in SECURITY_CONTROLS.items():
            checks[ctrl] = st.checkbox(f"{ctrl} (+{pts}pts)", value=random.random() > 0.45)

    with col_score:
        # Compute score
        base = sum(pts for ctrl, pts in SECURITY_CONTROLS.items() if checks[ctrl])
        breach_pen = int(str(prior_breaches).replace("+", "")) * 8
        score = max(0, min(100, base - breach_pen))
        ind_mult = INDUSTRIES[industry]
        risk_adj = 0.72 if score >= 75 else 1.0 if score >= 50 else 1.52
        base_prem = {"1–50 (SMB)": 7500, "51–250 (Mid-Market)": 21000, "251–1,000 (Growth)": 62000, "1,001+ (Enterprise)": 175000}[emp_band]
        premium = int(base_prem * ind_mult * risk_adj * (1 + revenue * 0.013) * (1 + (breach_pen / 8) * 0.22))

        color = OK if score >= 75 else WARN if score >= 50 else DANGER
        tier  = "Preferred Risk" if score >= 75 else "Standard Risk" if score >= 50 else "Elevated Risk"

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Navasana Risk Score™", "font": {"size": 16, "color": "#1f2937"}},
            number={"font": {"size": 48, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#d1d5db"},
                "bar": {"color": color},
                "bgcolor": "#f8fafc",
                "steps": [
                    {"range": [0, 50],  "color": "#fee2e2"},
                    {"range": [50, 75], "color": "#fef3c7"},
                    {"range": [75, 100],"color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": score}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20), paper_bgcolor="white")
        st.plotly_chart(fig_gauge, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1: card(color, "Risk Tier", tier)
        with c2: card(BRAND, "Controls", f"{sum(checks.values())}/10")
        with c3: card(DANGER if breach_pen else OK, "Breach Penalty", f"-{breach_pen}pts")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{BRAND}22,{BRAND}10);border:2px solid {BRAND}66;
        border-radius:12px;padding:20px;margin-top:16px;text-align:center;">
            <div style="color:#6b7280;font-size:12px;font-weight:600;letter-spacing:1px;text-transform:uppercase;">AI Premium Estimate</div>
            <div style="color:{BRAND};font-size:40px;font-weight:900;margin:8px 0;">${premium:,}</div>
            <div style="color:#6b7280;font-size:13px;">per year · {ind_mult}× {industry} multiplier</div>
        </div>
        """, unsafe_allow_html=True)

        # Gaps
        gaps = [c for c, v in checks.items() if not v]
        if gaps:
            st.markdown("**⚠️ Top Remediation Priorities**")
            for g in sorted(gaps, key=lambda x: SECURITY_CONTROLS[x], reverse=True)[:4]:
                st.markdown(f"""
                <div style="background:#fff7ed;border-left:3px solid {WARN};padding:8px 12px;
                border-radius:6px;margin-bottom:6px;font-size:13px;color:#92400e;">
                    ▲ {g} — saves {SECURITY_CONTROLS[g]}pts risk
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – UNDERWRITING ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Portfolio Underwriting Analytics")

    # Simulated portfolio data
    np.random.seed(42)
    n = 120
    scores   = np.clip(np.random.normal(64, 18, n), 5, 99).astype(int)
    premiums = [int(5000 + (100 - s) * 2800 + np.random.normal(0, 3000)) for s in scores]
    industries_list = np.random.choice(list(INDUSTRIES.keys()), n)
    claims   = [1 if (s < 50 and np.random.random() > 0.55) or (s < 70 and np.random.random() > 0.82) else 0 for s in scores]

    df = pd.DataFrame({"Risk Score": scores, "Premium": premiums, "Industry": industries_list, "Claim": claims})

    col1, col2 = st.columns(2)

    with col1:
        fig_scatter = px.scatter(
            df, x="Risk Score", y="Premium", color="Claim",
            color_discrete_map={0: BRAND, 1: DANGER},
            labels={"Claim": "Claim Filed"},
            title="Risk Score vs Premium (Portfolio View)",
            hover_data=["Industry"]
        )
        fig_scatter.update_traces(marker=dict(size=7, opacity=0.75))
        fig_scatter.update_layout(height=360, plot_bgcolor="#f8fafc", paper_bgcolor="white",
                                   font=dict(family="system-ui"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        ind_avg = df.groupby("Industry")["Risk Score"].mean().sort_values()
        fig_bar = go.Figure(go.Bar(
            y=ind_avg.index, x=ind_avg.values,
            orientation="h",
            marker_color=[DANGER if v < 55 else WARN if v < 70 else BRAND for v in ind_avg.values],
            text=[f"{v:.0f}" for v in ind_avg.values],
            textposition="outside"
        ))
        fig_bar.update_layout(
            title="Avg Risk Score by Industry", height=360,
            plot_bgcolor="#f8fafc", paper_bgcolor="white",
            xaxis=dict(range=[0, 105]), font=dict(family="system-ui")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Score distribution
    col3, col4 = st.columns(2)

    with col3:
        fig_hist = go.Figure(go.Histogram(
            x=df["Risk Score"], nbinsx=20,
            marker_color=BRAND, opacity=0.85,
            name="Policies"
        ))
        fig_hist.update_layout(
            title="Portfolio Risk Score Distribution", height=300,
            plot_bgcolor="#f8fafc", paper_bgcolor="white",
            xaxis_title="Risk Score", yaxis_title="# Policies",
            font=dict(family="system-ui")
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col4:
        tier_counts = {
            "Preferred (75–100)": int((df["Risk Score"] >= 75).sum()),
            "Standard (50–74)":   int(((df["Risk Score"] >= 50) & (df["Risk Score"] < 75)).sum()),
            "Elevated (<50)":     int((df["Risk Score"] < 50).sum()),
        }
        fig_pie = go.Figure(go.Pie(
            labels=list(tier_counts.keys()),
            values=list(tier_counts.values()),
            marker_colors=[OK, WARN, DANGER],
            hole=0.45
        ))
        fig_pie.update_layout(title="Portfolio Risk Tier Breakdown", height=300, paper_bgcolor="white",
                               font=dict(family="system-ui"))
        st.plotly_chart(fig_pie, use_container_width=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: card(BRAND, "Policies in Portfolio", f"{n}")
    with c2: card(OK, "Avg Risk Score", f"{df['Risk Score'].mean():.1f}/100")
    with c3: card(WARN, "Claim Rate", f"{df['Claim'].mean()*100:.1f}%")
    with c4: card(BRAND, "Total Premium", f"${df['Premium'].sum()/1e6:.2f}M")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – THREAT INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🌐 Real-Time Threat Intelligence Feed")
    st.markdown("*Attack trends ingested to dynamically adjust underwriting multipliers*")

    THREATS = [
        {"type": "AI-Generated Phishing", "target": "All Sectors", "severity": "Critical", "yoy": "+89%", "impact": "Premium +12%"},
        {"type": "Ransomware-as-a-Service", "target": "Healthcare", "severity": "Critical", "yoy": "+34%", "impact": "Premium +18%"},
        {"type": "Supply Chain Compromise", "target": "Technology", "severity": "High", "yoy": "+52%", "impact": "Premium +9%"},
        {"type": "BEC / Wire Fraud", "target": "Finance", "severity": "High", "yoy": "+18%", "impact": "Premium +7%"},
        {"type": "API Exploitation", "target": "SaaS", "severity": "High", "yoy": "+67%", "impact": "Premium +8%"},
        {"type": "Zero-Day Exploits", "target": "Manufacturing", "severity": "Critical", "yoy": "+23%", "impact": "Premium +14%"},
        {"type": "Credential Stuffing", "target": "Retail", "severity": "Medium", "yoy": "+12%", "impact": "Premium +4%"},
        {"type": "Insider Threat", "target": "Finance", "severity": "Medium", "yoy": "+8%", "impact": "Premium +5%"},
    ]

    SEV_COLOR = {"Critical": DANGER, "High": WARN, "Medium": "#3b82f6"}

    for t in THREATS:
        sc = SEV_COLOR[t["severity"]]
        st.markdown(f"""
        <div style="background:white;border:1px solid #e5e7eb;border-left:4px solid {sc};
        border-radius:10px;padding:14px 18px;margin-bottom:10px;
        display:flex;align-items:center;justify-content:space-between;">
            <div style="display:flex;align-items:center;gap:14px;">
                <div style="width:10px;height:10px;border-radius:50%;background:{sc};flex-shrink:0;"></div>
                <div>
                    <div style="font-weight:700;color:#1f2937;font-size:14px;">{t["type"]}</div>
                    <div style="color:#6b7280;font-size:12px;">{t["target"]}</div>
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:20px;">
                <span style="background:{sc}22;color:{sc};padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;">{t["severity"]}</span>
                <span style="color:{DANGER};font-weight:700;font-size:13px;">{t["yoy"]} YoY</span>
                <span style="color:{BRAND};font-weight:600;font-size:12px;">{t["impact"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📈 Threat Trend Timeline (Last 12 Months)")

    months = [(datetime.now() - timedelta(days=30*i)).strftime("%b %Y") for i in range(11, -1, -1)]
    fig_trend = go.Figure()
    for threat, color in [("Ransomware", DANGER), ("BEC Fraud", WARN), ("API Exploits", "#8b5cf6")]:
        base = {"Ransomware": 55, "BEC Fraud": 40, "API Exploits": 30}[threat]
        vals = [base + i * 2.5 + np.random.normal(0, 3) for i in range(12)]
        fig_trend.add_trace(go.Scatter(x=months, y=vals, mode="lines+markers",
                                       name=threat, line=dict(color=color, width=2.5),
                                       marker=dict(size=6)))
    fig_trend.update_layout(
        height=320, plot_bgcolor="#f8fafc", paper_bgcolor="white",
        yaxis_title="Threat Index", legend=dict(orientation="h", y=1.1),
        font=dict(family="system-ui")
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ⚙️ Underwriting ML Pipeline Architecture")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e5e7eb;border-radius:12px;padding:20px;">
            <h4 style="color:#1f2937;margin:0 0 16px 0;">🔄 Data Ingestion Layer</h4>
            {"".join([f'''<div style="background:{BRAND}18;border-left:3px solid {BRAND};padding:8px 12px;
            border-radius:6px;margin-bottom:8px;font-size:13px;color:#1f2937;">
                <strong>{src}</strong><br><span style="color:#6b7280;font-size:11px;">{desc}</span>
            </div>''' for src, desc in [
                ("CrowdStrike API", "EDR telemetry · threat detections · device health"),
                ("Okta Event Stream", "Auth failures · MFA adoption · access patterns"),
                ("AWS SecurityHub", "Cloud posture findings · IAM misconfigs · GuardDuty"),
                ("Qualys / Tenable", "Vulnerability scan results · CVSS scoring"),
                ("Dark Web Monitor", "Credential leaks · data exposure signals"),
            ]])}
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e5e7eb;border-radius:12px;padding:20px;">
            <h4 style="color:#1f2937;margin:0 0 16px 0;">🧠 ML Model Stack</h4>
            {"".join([f'''<div style="background:#f0fdf4;border-left:3px solid {OK};padding:8px 12px;
            border-radius:6px;margin-bottom:8px;font-size:13px;color:#1f2937;">
                <strong>{model}</strong><br><span style="color:#6b7280;font-size:11px;">{desc}</span>
            </div>''' for model, desc in [
                ("Risk Scoring Model", "XGBoost ensemble · 87 security posture features · 94.7% accuracy"),
                ("Anomaly Detection", "Isolation Forest on behavioral telemetry · real-time alerts"),
                ("Premium Pricing LLM", "Fine-tuned LLM on 50k+ historical policy/claim pairs"),
                ("Claims Prediction", "Survival analysis · time-to-claim modeling"),
                ("RAG Underwriting Agent", "LangChain + Chroma · policy document Q&A"),
            ]])}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🎯 Model Performance Metrics")
    metrics = [
        {"Metric": "Risk Score Accuracy", "Train": "96.2%", "Val": "94.7%", "Notes": "XGBoost w/ SHAP explainability"},
        {"Metric": "Claims Prediction AUC", "Train": "0.91", "Val": "0.88", "Notes": "Survival model, 18-month horizon"},
        {"Metric": "Premium RMSE", "Train": "$1,240", "Val": "$1,890", "Notes": "vs actuary baseline $4,200"},
        {"Metric": "Anomaly Detection F1", "Train": "0.89", "Val": "0.85", "Notes": "Behavioral telemetry signals"},
        {"Metric": "LLM Underwriting Q&A", "Train": "—", "Val": "91.3%", "Notes": "RAG accuracy on policy docs"},
    ]
    st.dataframe(
        pd.DataFrame(metrics),
        use_container_width=True,
        hide_index=True
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:20px;color:#9ca3af;font-size:13px;">
    Built for <strong style="color:{BRAND};">Navasana Inc.</strong> by
    <strong style="color:#1f2937;">Anju Vilashni Nandhakumar</strong> ·
    <a href="https://vxanju.com" style="color:{BRAND};">vxanju.com</a> ·
    <a href="https://linkedin.com/in/anju-vilashni" style="color:{BRAND};">LinkedIn</a> ·
    <a href="https://github.com/Av1352" style="color:{BRAND};">GitHub</a>
</div>
""", unsafe_allow_html=True)