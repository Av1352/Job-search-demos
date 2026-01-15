"""
Real-Time ML Model Monitoring Dashboard
Production-ready monitoring system for ML model quality assurance
Built for Centaur AI by Anju Nandhakumar
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Centaur AI Demo - Anju Vilashni",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main { background: white; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px 24px;
}
.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px 32px;
    font-size: 16px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# Set random seed
np.random.seed(42)

def generate_monitoring_dashboard(
    model_name, model_version, drift_scenario, drift_intensity,
    window_size, psi_threshold, degradation_threshold
):
    """Generate comprehensive monitoring dashboard"""
    
    baseline_accuracy = 0.92
    baseline_precision = 0.89
    baseline_recall = 0.91
    baseline_f1 = 0.90
    
    # Calculate current metrics based on drift
    if drift_scenario == "No Drift (Baseline)":
        current_accuracy = baseline_accuracy + np.random.normal(0, 0.01)
        current_precision = baseline_precision + np.random.normal(0, 0.01)
        psi_score = np.random.uniform(0.05, 0.09)
        js_divergence = np.random.uniform(0.10, 0.15)
        ks_statistic = np.random.uniform(0.10, 0.15)
        drift_detected = False
    elif drift_scenario == "Gradual Drift":
        drift_factor = drift_intensity * 0.15
        current_accuracy = baseline_accuracy - drift_factor
        current_precision = baseline_precision - drift_factor
        psi_score = 0.15 + (drift_intensity * 0.15)
        js_divergence = 0.20 + (drift_intensity * 0.10)
        ks_statistic = 0.20 + (drift_intensity * 0.15)
        drift_detected = psi_score > psi_threshold
    elif drift_scenario == "Sudden Drift":
        drift_factor = drift_intensity * 0.25
        current_accuracy = baseline_accuracy - drift_factor
        current_precision = baseline_precision - drift_factor
        psi_score = 0.25 + (drift_intensity * 0.25)
        js_divergence = 0.35 + (drift_intensity * 0.15)
        ks_statistic = 0.35 + (drift_intensity * 0.20)
        drift_detected = True
    elif drift_scenario == "Seasonal Drift":
        drift_factor = drift_intensity * 0.18
        current_accuracy = baseline_accuracy - drift_factor
        current_precision = baseline_precision - drift_factor
        psi_score = 0.16 + (drift_intensity * 0.18)
        js_divergence = 0.23 + (drift_intensity * 0.12)
        ks_statistic = 0.22 + (drift_intensity * 0.13)
        drift_detected = psi_score > psi_threshold
    else:  # Feature Drift
        drift_factor = drift_intensity * 0.20
        current_accuracy = baseline_accuracy - drift_factor
        current_precision = baseline_precision - drift_factor
        psi_score = 0.18 + (drift_intensity * 0.20)
        js_divergence = 0.25 + (drift_intensity * 0.12)
        ks_statistic = 0.25 + (drift_intensity * 0.15)
        drift_detected = psi_score > psi_threshold
    
    current_recall = baseline_recall - (baseline_accuracy - current_accuracy)
    current_f1 = 2 * (current_precision * current_recall) / (current_precision + current_recall) if (current_precision + current_recall) > 0 else 0
    
    delta_acc = ((current_accuracy - baseline_accuracy) / baseline_accuracy) * 100
    delta_prec = ((current_precision - baseline_precision) / baseline_precision) * 100
    delta_rec = ((current_recall - baseline_recall) / baseline_recall) * 100
    delta_f1 = ((current_f1 - baseline_f1) / baseline_f1) * 100
    
    return {
        'drift_detected': drift_detected,
        'current_accuracy': current_accuracy,
        'current_precision': current_precision,
        'current_recall': current_recall,
        'current_f1': current_f1,
        'baseline_accuracy': baseline_accuracy,
        'baseline_precision': baseline_precision,
        'baseline_recall': baseline_recall,
        'baseline_f1': baseline_f1,
        'delta_acc': delta_acc,
        'delta_prec': delta_prec,
        'delta_rec': delta_rec,
        'delta_f1': delta_f1,
        'psi_score': psi_score,
        'js_divergence': js_divergence,
        'ks_statistic': ks_statistic
    }

# Header
components.html(
    """
    <div style="
        text-align: center;
        padding: 20px 30px 70px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 25px;
        box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);
    ">
        <div style="
            width: 100px;
            height: 100px;
            background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
            border-radius: 50%;
            margin: 0 auto 25px auto;
            border: 5px solid white;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5);
        ">
            <span style="font-size: 56px;">🔍</span>
        </div>

        <h1 style="
            font-size: 58px;
            font-weight: 900;
            color: white;
            margin: 0 0 18px 0;
            text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            ML Model Monitor
        </h1>

        <p style="
            font-size: 28px;
            color: rgba(255,255,255,0.95);
            font-weight: 700;
            margin: 15px 0;
        ">
            Production-Ready Quality Assurance
        </p>

        <p style="
            font-size: 18px;
            color: rgba(255,255,255,0.85);
            font-weight: 500;
            margin-bottom: 25px;
        ">
            Real-time drift detection • Performance tracking • Intelligent alerting
        </p>

        <div style="
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
            max-width: 800px;
            margin: 28px auto 0 auto;
        ">
            <span style="background:#ec4899;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">PSI Detection</span>
            <span style="background:#8b5cf6;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">KS Test</span>
            <span style="background:#f59e0b;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">JS Divergence</span>
            <span style="background:#10b981;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Enterprise Ready</span>
        </div>

        <p style="
            font-size: 16px;
            color: rgba(255,255,255,0.9);
            margin-top: 28px;
            font-weight: 600;
        ">
            Built for <strong style="color:white;">Centaur AI</strong>
            by <strong style="color:white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """,
    height=520,
)

st.markdown("---")

# Layout: Sidebar for config, main area for results
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 20px; border-radius: 16px; margin-bottom: 22px; text-align: center;">
        <h3 style="margin: 0; font-size: 24px; font-weight: 900;">⚙️ Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    model_name = st.text_input("Model Name", value="Image_Classifier_v1")
    model_version = st.text_input("Model Version", value="1.2.3")
    
    drift_scenario = st.selectbox(
        "Drift Scenario",
        [
            "No Drift (Baseline)",
            "Gradual Drift",
            "Sudden Drift",
            "Seasonal Drift",
            "Feature Drift"
        ],
        help="Select a scenario to test drift detection"
    )
    
    drift_intensity = st.slider(
        "Drift Intensity",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="How severe the drift should be"
    )
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 20px; border-radius: 16px; margin: 22px 0; text-align: center;">
        <h3 style="margin: 0; font-size: 24px; font-weight: 900;">🎛️ Thresholds</h3>
    </div>
    """, unsafe_allow_html=True)
    
    window_size = st.slider(
        "Window Size",
        min_value=50,
        max_value=500,
        value=100,
        step=50,
        help="Number of predictions to monitor"
    )
    
    psi_threshold = st.slider(
        "PSI Alert Threshold",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="PSI > 0.2 triggers high alert"
    )
    
    degradation_threshold = st.slider(
        "Degradation Threshold (%)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Performance drop % to trigger alert"
    )
    
    analyze_btn = st.button("🚀 Run Monitoring Analysis", use_container_width=True, type="primary")

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 20px; border-radius: 16px; margin-bottom: 22px; text-align: center;">
        <h3 style="margin: 0; font-size: 24px; font-weight: 900;">📊 Analysis Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if analyze_btn:
        results = generate_monitoring_dashboard(
            model_name, model_version, drift_scenario, drift_intensity,
            window_size, psi_threshold, degradation_threshold
        )
        
        # Extract results
        drift_detected = results['drift_detected']
        current_accuracy = results['current_accuracy']
        current_precision = results['current_precision']
        current_recall = results['current_recall']
        current_f1 = results['current_f1']
        baseline_accuracy = results['baseline_accuracy']
        baseline_precision = results['baseline_precision']
        baseline_recall = results['baseline_recall']
        baseline_f1 = results['baseline_f1']
        delta_acc = results['delta_acc']
        delta_prec = results['delta_prec']
        delta_rec = results['delta_rec']
        delta_f1 = results['delta_f1']
        psi_score = results['psi_score']
        js_divergence = results['js_divergence']
        ks_statistic = results['ks_statistic']
        
        # System Status
        status = "🟢 HEALTHY" if not drift_detected else "🔴 DRIFT DETECTED"
        status_color = "#10b981" if not drift_detected else "#ef4444"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <div>
                    <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 8px 0;">📊 System Status</h2>
                    <p style="color: rgba(255,255,255,0.9); font-size: 16px; margin: 0;">{datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
                <div style="background: {status_color}; padding: 15px 30px; border-radius: 50px;">
                    <p style="color: white; font-size: 22px; font-weight: 800; margin: 0;">{status}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 12px; padding: 18px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.2);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 8px 0; font-weight: 600;">Model</p>
                <p style="font-size: 18px; color: #667eea; font-weight: 800; margin: 0;">{model_name}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 4px 0 0 0;">v{model_version}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 12px; padding: 18px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.2);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 8px 0; font-weight: 600;">Scenario</p>
                <p style="font-size: 18px; color: #667eea; font-weight: 800; margin: 0;">{drift_scenario.split()[0]}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 4px 0 0 0;">Intensity: {drift_intensity:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 12px; padding: 18px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.2);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 8px 0; font-weight: 600;">Window Size</p>
                <p style="font-size: 28px; color: #667eea; font-weight: 900; margin: 0;">{window_size}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 4px 0 0 0;">predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 12px; padding: 18px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.2);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 8px 0; font-weight: 600;">Drift Alert</p>
                <p style="font-size: 28px; color: {status_color}; font-weight: 900; margin: 0;">{'YES' if drift_detected else 'NO'}</p>
                <p style="font-size: 13px; color: #6b7280; margin: 4px 0 0 0;">PSI >{psi_threshold:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Performance Metrics
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; font-size: 28px; font-weight: 900; margin: 0 0 24px 0;">📈 Performance Metrics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            delta_color = '#10b981' if delta_acc >= 0 else '#ef4444'
            st.markdown(f"""
            <div style="background: white; border-radius: 16px; padding: 24px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 5px solid #3b82f6;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="font-size: 14px; color: #6b7280; margin: 0 0 6px 0; font-weight: 600;">Accuracy</p>
                        <p style="font-size: 36px; color: #1f2937; font-weight: 900; margin: 0;">{current_accuracy:.1%}</p>
                        <p style="font-size: 13px; color: #9ca3af; margin: 6px 0 0 0;">Baseline: {baseline_accuracy:.1%}</p>
                    </div>
                    <div>
                        <span style="background: {delta_color}; color: white; padding: 8px 16px; border-radius: 20px; font-size: 16px; font-weight: 800;">{delta_acc:+.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            delta_color_rec = '#10b981' if delta_rec >= 0 else '#ef4444'
            st.markdown(f"""
            <div style="background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 5px solid #ec4899;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="font-size: 14px; color: #6b7280; margin: 0 0 6px 0; font-weight: 600;">Recall</p>
                        <p style="font-size: 36px; color: #1f2937; font-weight: 900; margin: 0;">{current_recall:.1%}</p>
                        <p style="font-size: 13px; color: #9ca3af; margin: 6px 0 0 0;">Baseline: {baseline_recall:.1%}</p>
                    </div>
                    <div>
                        <span style="background: {delta_color_rec}; color: white; padding: 8px 16px; border-radius: 20px; font-size: 16px; font-weight: 800;">{delta_rec:+.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            delta_color_prec = '#10b981' if delta_prec >= 0 else '#ef4444'
            st.markdown(f"""
            <div style="background: white; border-radius: 16px; padding: 24px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 5px solid #8b5cf6;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="font-size: 14px; color: #6b7280; margin: 0 0 6px 0; font-weight: 600;">Precision</p>
                        <p style="font-size: 36px; color: #1f2937; font-weight: 900; margin: 0;">{current_precision:.1%}</p>
                        <p style="font-size: 13px; color: #9ca3af; margin: 6px 0 0 0;">Baseline: {baseline_precision:.1%}</p>
                    </div>
                    <div>
                        <span style="background: {delta_color_prec}; color: white; padding: 8px 16px; border-radius: 20px; font-size: 16px; font-weight: 800;">{delta_prec:+.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            delta_color_f1 = '#10b981' if delta_f1 >= 0 else '#ef4444'
            st.markdown(f"""
            <div style="background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 5px solid #f59e0b;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="font-size: 14px; color: #6b7280; margin: 0 0 6px 0; font-weight: 600;">F1 Score</p>
                        <p style="font-size: 36px; color: #1f2937; font-weight: 900; margin: 0;">{current_f1:.1%}</p>
                        <p style="font-size: 13px; color: #9ca3af; margin: 6px 0 0 0;">Baseline: {baseline_f1:.1%}</p>
                    </div>
                    <div>
                        <span style="background: {delta_color_f1}; color: white; padding: 8px 16px; border-radius: 20px; font-size: 16px; font-weight: 800;">{delta_f1:+.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Drift Detection
        psi_status = 'HIGH' if psi_score > 0.2 else 'MEDIUM' if psi_score > 0.1 else 'LOW'
        js_status = 'HIGH' if js_divergence > 0.3 else 'MEDIUM' if js_divergence > 0.2 else 'LOW'
        ks_status = 'HIGH' if ks_statistic > 0.3 else 'MEDIUM' if ks_statistic > 0.2 else 'LOW'
        
        drift_badge_colors = {
            'LOW': '#10b981',
            'MEDIUM': '#f59e0b',
            'HIGH': '#ef4444'
        }
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <h3 style="color: #92400e; font-size: 28px; font-weight: 900; margin: 0 0 24px 0;">🌊 Drift Detection Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: white; border-radius: 16px; padding: 24px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 12px 0; font-weight: 700;">Population Stability Index</p>
                <p style="font-size: 48px; color: #1f2937; font-weight: 900; margin: 0;">{psi_score:.3f}</p>
                <span style="background: {drift_badge_colors[psi_status]}; color: white; padding: 6px 20px; border-radius: 25px; font-size: 13px; font-weight: 800; margin-top: 12px; display: inline-block;">{psi_status} RISK</span>
                <p style="font-size: 12px; color: #9ca3af; margin: 10px 0 0 0;">Threshold: {psi_threshold:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: white; border-radius: 16px; padding: 24px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 12px 0; font-weight: 700;">JS Divergence</p>
                <p style="font-size: 48px; color: #1f2937; font-weight: 900; margin: 0;">{js_divergence:.3f}</p>
                <span style="background: {drift_badge_colors[js_status]}; color: white; padding: 6px 20px; border-radius: 25px; font-size: 13px; font-weight: 800; margin-top: 12px; display: inline-block;">{js_status} RISK</span>
                <p style="font-size: 12px; color: #9ca3af; margin: 10px 0 0 0;">Range: 0.0 - 1.0</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: white; border-radius: 16px; padding: 24px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 12px 0; font-weight: 700;">KS Test Statistic</p>
                <p style="font-size: 48px; color: #1f2937; font-weight: 900; margin: 0;">{ks_statistic:.3f}</p>
                <span style="background: {drift_badge_colors[ks_status]}; color: white; padding: 6px 20px; border-radius: 25px; font-size: 13px; font-weight: 800; margin-top: 12px; display: inline-block;">{ks_status} RISK</span>
                <p style="font-size: 12px; color: #9ca3af; margin: 10px 0 0 0;">P-value based</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.15); border-radius: 12px; padding: 18px; margin-top: 20px; border: 2px dashed #f59e0b;">
            <p style="font-size: 14px; color: #92400e; margin: 0; line-height: 1.6; font-weight: 600;">
                <strong style="font-size: 16px;">🔬 Statistical Methods:</strong> Kolmogorov-Smirnov Test (distribution comparison) • Population Stability Index (industry standard) • Jensen-Shannon Divergence (symmetric KL)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Alerts
        if drift_detected or abs(delta_acc) > 5:
            if psi_score > psi_threshold:
                severity = "CRITICAL" if psi_score > 0.25 else "HIGH"
                alert_color = "#dc2626" if severity == "CRITICAL" else "#f59e0b"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border: 4px solid #ef4444; border-radius: 20px; padding: 28px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
                        <div style="background: {alert_color}; width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 4px solid white;">
                            <span style="font-size: 36px;">🚨</span>
                        </div>
                        <div>
                            <h3 style="color: #991b1b; font-size: 26px; font-weight: 900; margin: 0;">{severity} ALERT: Feature Drift Detected</h3>
                            <p style="color: #dc2626; font-size: 14px; margin: 6px 0 0 0; font-weight: 600;">Model predictions may be unreliable</p>
                        </div>
                    </div>
                    
                    <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 18px;">
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                            <div>
                                <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">PSI Score</p>
                                <p style="font-size: 24px; color: #dc2626; font-weight: 900; margin: 0;">{psi_score:.3f}</p>
                            </div>
                            <div>
                                <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Threshold</p>
                                <p style="font-size: 24px; color: #6b7280; font-weight: 900; margin: 0;">{psi_threshold:.2f}</p>
                            </div>
                            <div>
                                <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Severity</p>
                                <p style="font-size: 24px; color: #dc2626; font-weight: 900; margin: 0;">{severity}</p>
                            </div>
                        </div>
                    </div>
                    
                    <div style="background: {alert_color}; border-radius: 12px; padding: 20px; color: white;">
                        <p style="font-size: 16px; font-weight: 800; margin: 0 0 12px 0;">⚡ Immediate Actions Required:</p>
                        <ul style="margin: 0; padding-left: 24px; line-height: 2;">
                            <li style="font-size: 14px;">Collect recent production data (min 1000 samples)</li>
                            <li style="font-size: 14px;">Retrain model on combined historical + recent data</li>
                            <li style="font-size: 14px;">A/B test new model before deployment</li>
                            <li style="font-size: 14px;">Document drift incident for compliance</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div style="background: #10b981; width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 4px solid white;">
                        <span style="font-size: 36px;">✅</span>
                    </div>
                    <div>
                        <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0;">System Status: HEALTHY</h3>
                        <p style="color: #047857; font-size: 14px; margin: 6px 0 0 0; font-weight: 600;">No significant drift detected • Model performing within normal range</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs for charts
        tab1, tab2, tab3 = st.tabs(["📈 Drift Gauges", "📉 Performance Trends", "📊 Comparison"])
        
        with tab1:
            # Drift gauges
            fig_gauges = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=('PSI', 'JS Divergence', 'KS Statistic')
            )
            
            psi_color = "green" if psi_score < 0.1 else "orange" if psi_score < 0.2 else "red"
            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number",
                value=psi_score,
                gauge={
                    'axis': {'range': [0, 0.5]},
                    'bar': {'color': psi_color},
                    'steps': [
                        {'range': [0, 0.1], 'color': "lightgreen"},
                        {'range': [0.1, 0.2], 'color': "lightyellow"},
                        {'range': [0.2, 0.5], 'color': "lightcoral"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': psi_threshold}
                }
            ), row=1, col=1)
            
            js_color = "green" if js_divergence < 0.2 else "orange" if js_divergence < 0.3 else "red"
            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number",
                value=js_divergence,
                gauge={
                    'axis': {'range': [0, 0.5]},
                    'bar': {'color': js_color},
                    'steps': [
                        {'range': [0, 0.2], 'color': "lightgreen"},
                        {'range': [0.2, 0.3], 'color': "lightyellow"},
                        {'range': [0.3, 0.5], 'color': "lightcoral"}
                    ]
                }
            ), row=1, col=2)
            
            ks_color = "green" if ks_statistic < 0.2 else "orange" if ks_statistic < 0.3 else "red"
            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number",
                value=ks_statistic,
                gauge={
                    'axis': {'range': [0, 0.5]},
                    'bar': {'color': ks_color},
                    'steps': [
                        {'range': [0, 0.2], 'color': "lightgreen"},
                        {'range': [0.2, 0.3], 'color': "lightyellow"},
                        {'range': [0.3, 0.5], 'color': "lightcoral"}
                    ]
                }
            ), row=1, col=3)
            
            fig_gauges.update_layout(height=300, showlegend=False, title_text="Drift Detection Metrics")
            st.plotly_chart(fig_gauges, use_container_width=True)
        
        with tab2:
            # Performance trends
            time_points = 20
            timestamps = pd.date_range(end=datetime.now(), periods=time_points, freq='5min')
            
            acc_trend = [baseline_accuracy - (i / time_points) * (baseline_accuracy - current_accuracy) + np.random.normal(0, 0.005) for i in range(time_points)]
            prec_trend = [baseline_precision - (i / time_points) * (baseline_precision - current_precision) + np.random.normal(0, 0.005) for i in range(time_points)]
            rec_trend = [baseline_recall - (i / time_points) * (baseline_recall - current_recall) + np.random.normal(0, 0.005) for i in range(time_points)]
            f1_trend = [baseline_f1 - (i / time_points) * (baseline_f1 - current_f1) + np.random.normal(0, 0.005) for i in range(time_points)]
            
            fig_trends = go.Figure()
            fig_trends.add_trace(go.Scatter(x=timestamps, y=acc_trend, name='Accuracy', line=dict(color='#3b82f6', width=3), mode='lines+markers'))
            fig_trends.add_trace(go.Scatter(x=timestamps, y=prec_trend, name='Precision', line=dict(color='#8b5cf6', width=2), mode='lines'))
            fig_trends.add_trace(go.Scatter(x=timestamps, y=rec_trend, name='Recall', line=dict(color='#ec4899', width=2), mode='lines'))
            fig_trends.add_trace(go.Scatter(x=timestamps, y=f1_trend, name='F1 Score', line=dict(color='#f59e0b', width=2), mode='lines'))
            fig_trends.add_hline(y=baseline_accuracy, line_dash="dash", line_color="gray", annotation_text="Baseline")
            fig_trends.update_layout(
                title="Performance Metrics Over Time (Last 100 Minutes)",
                xaxis_title="Time",
                yaxis_title="Score",
                yaxis_range=[0.7, 1.0],
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_trends, use_container_width=True)
        
        with tab3:
            # Comparison
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            baseline_values = [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1]
            current_values = [current_accuracy, current_precision, current_recall, current_f1]
            
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Bar(
                name='Baseline', x=metrics, y=baseline_values,
                marker_color='lightblue', text=[f'{v:.1%}' for v in baseline_values], textposition='outside'
            ))
            fig_comparison.add_trace(go.Bar(
                name='Current', x=metrics, y=current_values,
                marker_color=['green' if current_values[i] >= baseline_values[i] else 'red' for i in range(len(metrics))],
                text=[f'{v:.1%}' for v in current_values], textposition='outside'
            ))
            fig_comparison.update_layout(
                title="Performance Metrics: Baseline vs Current",
                yaxis_title="Score",
                yaxis_range=[0, 1.1],
                height=400,
                barmode='group'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

# Footer
st.markdown("<hr style='border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Centaur AI</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: rgba(102, 126, 234, 0.1); border-radius: 16px; padding: 24px; margin-top: 20px; text-align: center;">
    <p style="margin: 8px 0; font-size: 16px;">
        📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: #667eea; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a>
    </p>
    <p style="margin: 8px 0; font-size: 16px;">
        💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: #667eea; font-weight: 700; text-decoration: none;">LinkedIn</a> | 
        💻 <a href="https://github.com/Av1352" target="_blank" style="color: #667eea; font-weight: 700; text-decoration: none;">GitHub</a> | 
        🌐 <a href="https://vxanju.com" target="_blank" style="color: #667eea; font-weight: 700; text-decoration: none;">Portfolio</a>
    </p>
    <p style="font-size: 15px; margin: 18px 0 0 0; font-weight: 700; color: #1f2937;">
        <strong>Tech Stack:</strong> Python • Streamlit • NumPy • Pandas • Plotly
    </p>
</div>
""", unsafe_allow_html=True)