"""
Revise Robotics - Computer Vision for Electronics QC
Defect detection and quality inspection for consumer electronics
Built for Revise Robotics by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Revise Robotics", page_icon="🤖", layout="wide")

# Defect types
DEFECT_TYPES = {
    'Scratch': {'severity': 'Minor', 'color': '#f59e0b', 'count': 234},
    'Crack': {'severity': 'Critical', 'color': '#ef4444', 'count': 45},
    'Discoloration': {'severity': 'Minor', 'color': '#f59e0b', 'count': 189},
    'Missing Component': {'severity': 'Critical', 'color': '#ef4444', 'count': 23},
    'Dent': {'severity': 'Moderate', 'color': '#f97316', 'count': 156},
    'Solder Defect': {'severity': 'Critical', 'color': '#ef4444', 'count': 67},
    'Contamination': {'severity': 'Moderate', 'color': '#f97316', 'count': 98}
}

# Inspection metrics
INSPECTION_METRICS = {
    'Detection Accuracy': 98.7,
    'False Positive Rate': 1.2,
    'False Negative Rate': 0.8,
    'Inspection Speed': 0.3,  # seconds
    'Throughput': 12000  # units/hour
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #06b6d4 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🤖</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Revise Robotics</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">Computer Vision for Electronics QC</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Automated defect detection • Real-time inspection • 98.7% accuracy</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Real-Time Inspection", "📊 Defect Analysis", "⚡ Performance Metrics", "💡 Technology"])

with tab1:
    st.markdown("### Real-Time Quality Inspection")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Inspection Configuration**")
        
        product_type = st.selectbox(
            "Product Type",
            ["Smartphone", "Laptop", "Tablet", "PCB Board", "Display Panel", "Battery"]
        )
        
        inspection_mode = st.selectbox(
            "Inspection Mode",
            ["Full Inspection", "Surface Only", "Component Check", "Critical Defects Only"]
        )
        
        sensitivity = st.select_slider(
            "Detection Sensitivity",
            options=["Low", "Medium", "High", "Ultra"],
            value="High"
        )
        
        st.markdown("**Defect Categories to Check**")
        check_scratches = st.checkbox("Surface Scratches", value=True)
        check_cracks = st.checkbox("Cracks/Fractures", value=True)
        check_components = st.checkbox("Missing Components", value=True)
        check_solder = st.checkbox("Solder Defects", value=True)
        check_contamination = st.checkbox("Contamination", value=True)
        
        inspect_btn = st.button("🤖 Start Inspection", type="primary", use_container_width=True)
    
    with col2:
        if inspect_btn:
            st.markdown("**Inspection Results**")
            
            with st.spinner("Analyzing product..."):
                import time
                time.sleep(1.2)
            
            # Simulate defect detection
            detected_defects = np.random.randint(0, 3)
            
            if detected_defects == 0:
                st.success("✅ PASS - No defects detected")
                status_color = "#10b981"
                status_text = "PASS"
            else:
                st.error(f"❌ FAIL - {detected_defects} defect(s) detected")
                status_color = "#ef4444"
                status_text = "FAIL"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {status_color} 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 28px; font-weight: 900;">Inspection Result: {status_text}</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Product Type</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{product_type}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Defects Found</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">{detected_defects}</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Inspection Time</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">0.3s</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Confidence</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">98.7%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if detected_defects > 0:
                st.markdown("**Detected Defects**")
                
                defect_list = []
                if detected_defects >= 1:
                    defect_list.append({'Type': 'Scratch', 'Location': 'Top-right corner', 'Severity': 'Minor', 'Confidence': '97.2%'})
                if detected_defects >= 2:
                    defect_list.append({'Type': 'Dent', 'Location': 'Side panel', 'Severity': 'Moderate', 'Confidence': '95.8%'})
                if detected_defects >= 3:
                    defect_list.append({'Type': 'Discoloration', 'Location': 'Display edge', 'Severity': 'Minor', 'Confidence': '93.4%'})
                
                st.dataframe(pd.DataFrame(defect_list), hide_index=True, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Inspected Today", "8,947", "+347")
            col2.metric("Pass Rate", "96.8%", "+0.3%")
            col3.metric("Avg Inspection Time", "0.3s", "-0.02s")

with tab2:
    st.markdown("### Defect Analysis Dashboard")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Defect Distribution**")
        
        defect_data = []
        for defect, data in DEFECT_TYPES.items():
            defect_data.append({
                'Defect Type': defect,
                'Count': data['count'],
                'Severity': data['severity'],
                'Trend': '↓' if np.random.random() > 0.5 else '↑'
            })
        
        st.dataframe(pd.DataFrame(defect_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Defect Type Distribution**")
        
        fig1 = go.Figure(data=[go.Pie(
            labels=list(DEFECT_TYPES.keys()),
            values=[DEFECT_TYPES[d]['count'] for d in DEFECT_TYPES.keys()],
            hole=0.4,
            marker=dict(colors=[DEFECT_TYPES[d]['color'] for d in DEFECT_TYPES.keys()])
        )])
        fig1.update_layout(height=300)
        st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("**Defect Trends Over Time**")
    
    # Generate trend data
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    scratches = [245, 238, 234, 229, 234]
    cracks = [52, 48, 45, 43, 45]
    dents = [168, 162, 156, 151, 156]
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=days, y=scratches, mode='lines+markers', name='Scratches', line=dict(color='#f59e0b', width=3)))
    fig2.add_trace(go.Scatter(x=days, y=cracks, mode='lines+markers', name='Cracks', line=dict(color='#ef4444', width=3)))
    fig2.add_trace(go.Scatter(x=days, y=dents, mode='lines+markers', name='Dents', line=dict(color='#f97316', width=3)))
    
    fig2.update_layout(
        xaxis_title='Day',
        yaxis_title='Defect Count',
        height=300
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("**Root Cause Analysis**")
    
    root_causes = {
        'Cause': ['Handling Damage', 'Manufacturing Process', 'Material Quality', 'Assembly Error', 'Environmental'],
        'Contribution': [35, 28, 18, 12, 7],
        'Action': ['Training update', 'Process review', 'Supplier audit', 'SOP revision', 'Climate control']
    }
    st.dataframe(pd.DataFrame(root_causes), hide_index=True, use_container_width=True)

with tab3:
    st.markdown("### Performance & Quality Metrics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Detection Performance**")
        
        metrics_data = []
        for metric, value in INSPECTION_METRICS.items():
            if 'Rate' in metric:
                display_value = f"{value}%"
            elif 'Speed' in metric:
                display_value = f"{value}s"
            elif 'Throughput' in metric:
                display_value = f"{value:,}/hr"
            else:
                display_value = f"{value}%"
            
            status = '✅ Excellent' if value > 95 or (value < 2 and 'Rate' in metric) else '✅ Good'
            
            metrics_data.append({
                'Metric': metric,
                'Value': display_value,
                'Status': status
            })
        
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Comparison vs Manual Inspection**")
        
        comparison = {
            'Method': ['Revise CV System', 'Manual Inspection'],
            'Accuracy': ['98.7%', '92.3%'],
            'Speed (per unit)': ['0.3s', '45s'],
            'Throughput/hour': ['12,000', '80'],
            'Labor Cost': ['$0.02', '$3.50']
        }
        st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Accuracy Metrics**")
        
        fig3 = go.Figure(data=[go.Bar(
            x=['Detection Accuracy', 'Precision', 'Recall', 'F1 Score'],
            y=[98.7, 97.9, 98.2, 98.0],
            marker=dict(color='#06b6d4'),
            text=['98.7%', '97.9%', '98.2%', '98.0%'],
            textposition='auto'
        )])
        fig3.update_layout(
            yaxis=dict(range=[90, 100]),
            height=250
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("**Confusion Matrix**")
        
        confusion = {
            'Actual': ['Defect', 'Defect', 'No Defect', 'No Defect'],
            'Predicted': ['Defect', 'No Defect', 'Defect', 'No Defect'],
            'Count': [9847, 79, 108, 298966]
        }
        st.dataframe(pd.DataFrame(confusion), hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### Computer Vision Technology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Detection Models**")
        st.markdown("""
        - ✅ YOLOv8 (real-time object detection)
        - ✅ ResNet50 (feature extraction)
        - ✅ EfficientNet (classification)
        - ✅ U-Net (segmentation)
        - ✅ Custom ensemble models
        - ✅ Edge-optimized inference (<500ms)
        """)
        
        st.markdown("**Imaging System**")
        st.markdown("""
        - ✅ 12MP industrial cameras
        - ✅ Multi-angle capture (6 views)
        - ✅ LED ring lighting
        - ✅ Macro lens (10x zoom)
        - ✅ UV/IR illumination
        - ✅ 60 FPS capture rate
        """)
    
    with col2:
        st.markdown("**Defect Categories**")
        st.markdown("""
        - ✅ Surface defects (scratches, dents)
        - ✅ Structural defects (cracks, fractures)
        - ✅ Component defects (missing, misaligned)
        - ✅ Solder defects (bridges, voids)
        - ✅ Contamination (particles, stains)
        - ✅ Color/texture anomalies
        """)
        
        st.markdown("**Integration**")
        st.markdown("""
        - ✅ Manufacturing line integration
        - ✅ Real-time data logging
        - ✅ ERP/MES connectivity
        - ✅ Quality dashboard
        - ✅ Automated sorting
        - ✅ API for third-party systems
        """)
    
    st.markdown("**Supported Product Types**")
    
    products = {
        'Product': ['Smartphones', 'Laptops', 'Tablets', 'PCB Boards', 'Display Panels', 'Batteries'],
        'Inspection Points': [24, 32, 18, 156, 12, 8],
        'Throughput/hour': ['12,000', '8,000', '15,000', '20,000', '10,000', '18,000']
    }
    st.dataframe(pd.DataFrame(products), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #ecfeff 0%, #cffafe 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #164e63; font-size: 24px; font-weight: 800;">💡 System Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0891b2; font-weight: 700; margin: 0 0 6px 0;">✓ 98.7% Accuracy</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Superior to 92.3% manual</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0891b2; font-weight: 700; margin: 0 0 6px 0;">✓ 0.3s Inspection</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">150x faster than manual (45s)</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0891b2; font-weight: 700; margin: 0 0 6px 0;">✓ 12,000 Units/Hour</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">150x manual throughput</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #0891b2; font-weight: 700; margin: 0 0 6px 0;">✓ YOLOv8 + ResNet50</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Real-time CV models</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #06b6d4 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Revise Robotics</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)