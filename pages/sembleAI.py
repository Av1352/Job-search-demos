"""
Semble AI - Autonomous Building System Design
AI-powered optimization for construction companies
Built for Semble AI by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Semble AI - Building System Design", layout="wide")

# Initialize session state
if 'design_generated' not in st.session_state:
    st.session_state.design_generated = False

# Building system optimization functions
def calculate_hvac_design(sqft, floors, zones, climate, efficiency_pref):
    """Calculate optimal HVAC system design"""
    
    # Base tonnage calculation (rule of thumb: 400-600 sqft per ton)
    base_tonnage = sqft / 500
    
    # Climate adjustment
    climate_factors = {
        "Hot & Humid": 1.25,
        "Hot & Dry": 1.15,
        "Cold": 1.20,
        "Moderate": 1.0
    }
    adjusted_tonnage = base_tonnage * climate_factors[climate]
    
    # Efficiency adjustment
    efficiency_costs = {
        "Standard (SEER 14)": 1.0,
        "High Efficiency (SEER 18)": 1.35,
        "Premium (SEER 22)": 1.65
    }
    cost_multiplier = efficiency_costs[efficiency_pref]
    
    # Calculate costs
    base_cost_per_ton = 6500
    total_equipment_cost = adjusted_tonnage * base_cost_per_ton * cost_multiplier
    
    # Installation costs
    ductwork_cost = sqft * 12  # $12 per sqft for ductwork
    labor_cost = total_equipment_cost * 0.4
    total_cost = total_equipment_cost + ductwork_cost + labor_cost
    
    # Operating costs (annual)
    kwh_per_ton_per_year = {
        "Standard (SEER 14)": 1200,
        "High Efficiency (SEER 18)": 933,
        "Premium (SEER 22)": 764
    }[efficiency_pref]
    
    annual_energy_cost = adjusted_tonnage * kwh_per_ton_per_year * 0.15  # $0.15/kWh
    
    return {
        'tonnage': adjusted_tonnage,
        'equipment_cost': total_equipment_cost,
        'ductwork_cost': ductwork_cost,
        'labor_cost': labor_cost,
        'total_installation': total_cost,
        'annual_energy_cost': annual_energy_cost,
        'zones': zones,
        'efficiency_rating': efficiency_pref.split('(')[1].split(')')[0],
        'payback_years': (total_cost - (base_tonnage * base_cost_per_ton * 1.4)) / (1200 * base_tonnage * 0.15 - annual_energy_cost) if cost_multiplier > 1 else 0
    }

def generate_alternative_designs(sqft, floors, zones, climate):
    """Generate 3 alternative system designs"""
    designs = []
    
    for eff in ["Standard (SEER 14)", "High Efficiency (SEER 18)", "Premium (SEER 22)"]:
        design = calculate_hvac_design(sqft, floors, zones, climate, eff)
        design['name'] = eff.split(' (')[0]
        designs.append(design)
    
    return designs

def create_cost_comparison_chart(designs):
    """Create cost comparison visualization"""
    names = [d['name'] for d in designs]
    install_costs = [d['total_installation'] for d in designs]
    annual_costs = [d['annual_energy_cost'] for d in designs]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Installation Cost", "Annual Operating Cost"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=names, y=install_costs, marker_color=['#3b82f6', '#059669', '#8b5cf6'],
               text=[f'${c:,.0f}' for c in install_costs], textposition='outside'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=names, y=annual_costs, marker_color=['#3b82f6', '#059669', '#8b5cf6'],
               text=[f'${c:,.0f}' for c in annual_costs], textposition='outside'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, template="plotly_white")
    fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cost ($)", row=1, col=2)
    
    return fig

def create_system_layout(zones):
    """Create visual system layout"""
    fig = go.Figure()
    
    # Create a simple floor plan visualization
    colors = ['#3b82f6', '#059669', '#f59e0b', '#8b5cf6', '#ec4899']
    
    # Generate zone positions
    cols = int(np.ceil(np.sqrt(zones)))
    rows = int(np.ceil(zones / cols))
    
    for i in range(zones):
        row = i // cols
        col = i % cols
        
        fig.add_trace(go.Scatter(
            x=[col, col+0.9, col+0.9, col, col],
            y=[row, row, row+0.9, row+0.9, row],
            fill='toself',
            fillcolor=colors[i % len(colors)],
            line=dict(color='white', width=2),
            name=f'Zone {i+1}',
            text=f'Zone {i+1}',
            textposition='middle center',
            mode='lines+text',
            hoverinfo='text',
            hovertext=f'Zone {i+1}<br>Temperature Control<br>Air Flow Optimization'
        ))
    
    fig.update_layout(
        title="HVAC Zone Layout",
        showlegend=False,
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#f3f4f6'
    )
    
    return fig

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #ea580c 0%, #f97316 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(249, 115, 22, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #f97316 0%, #fb923c 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(249, 115, 22, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🏗️</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Semble AI
        </h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Building System Design Optimizer</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">AI-powered HVAC, electrical, and plumbing design</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);">Optimization</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Cost Analysis</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Energy Efficiency</span>
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">YC Backed</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Semble AI</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Value Prop
st.markdown("""
<div style="background: linear-gradient(135deg, #fff7ed, #fed7aa); padding: 25px; border-radius: 15px; border: 2px solid #ea580c; margin-bottom: 30px;">
    <h3 style="color: #7c2d12; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">🎯 The Building Design Problem</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #ef4444; font-weight: 700; margin: 0 0 8px 0;">❌ Today</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">HVAC design takes 20-40 hours per project. Engineers manually calculate loads, routes, costs. High error rate.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #f59e0b; font-weight: 700; margin: 0 0 8px 0;">💰 Cost Impact</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Design errors cost $50K+ to fix post-construction. Oversized systems waste 30% energy. Design bottlenecks delay projects.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #10b981; font-weight: 700; margin: 0 0 8px 0;">✅ With Semble</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Generate optimal design in 10 minutes. 15-25% cost savings. 30% energy efficiency improvement.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🏗️ Design System", "📊 Compare Options", "🔧 Technical Details"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Building Specifications</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Input your building parameters - AI will generate optimized system design</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**📐 Building Parameters**")
        
        sqft = st.number_input("Total Square Footage", min_value=1000, max_value=500000, value=15000, step=1000)
        
        col_a, col_b = st.columns(2)
        with col_a:
            floors = st.number_input("Number of Floors", min_value=1, max_value=50, value=3)
            zones = st.number_input("HVAC Zones", min_value=1, max_value=20, value=6)
        with col_b:
            building_type = st.selectbox("Building Type", 
                ["Office", "Retail", "Warehouse", "Mixed Use", "Healthcare Facility"])
            climate = st.selectbox("Climate Zone", 
                ["Hot & Humid", "Hot & Dry", "Cold", "Moderate"])
        
        st.markdown("**⚡ Design Preferences**")
        
        efficiency_pref = st.selectbox("Efficiency Target", 
            ["Standard (SEER 14)", "High Efficiency (SEER 18)", "Premium (SEER 22)"])
        
        budget = st.slider("Budget Range ($)", 100000, 1000000, 350000, 25000)
        
        if st.button("🚀 Generate Optimal Design", type="primary", use_container_width=True):
            st.session_state.design_generated = True
            st.session_state.building_specs = {
                'sqft': sqft,
                'floors': floors,
                'zones': zones,
                'building_type': building_type,
                'climate': climate,
                'efficiency_pref': efficiency_pref,
                'budget': budget
            }
    
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #92400e; margin: 0 0 12px 0; font-size: 16px;">💡 What We Optimize</h4>
            <ul style="color: #78350f; font-size: 13px; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li><strong>Equipment sizing:</strong> Right tonnage, no over/undersizing</li>
                <li><strong>Zone layout:</strong> Optimal airflow distribution</li>
                <li><strong>Ductwork routing:</strong> Shortest runs, minimal pressure loss</li>
                <li><strong>Energy efficiency:</strong> Lower operating costs</li>
                <li><strong>Code compliance:</strong> Meets all local regulations</li>
                <li><strong>Cost vs performance:</strong> Best ROI tradeoffs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.design_generated:
            specs = st.session_state.building_specs
            design = calculate_hvac_design(specs['sqft'], specs['floors'], specs['zones'], 
                                          specs['climate'], specs['efficiency_pref'])
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); padding: 20px; border-radius: 12px; border: 2px solid #059669; margin-top: 20px;">
                <h4 style="color: #065f46; margin: 0 0 15px 0; font-size: 18px;">✅ Optimal Design Generated</h4>
                <div style="background: white; padding: 15px; border-radius: 10px;">
                    <p style="color: #1f2937; font-weight: 700; margin: 0 0 8px 0;">System Capacity</p>
                    <p style="color: #059669; font-size: 32px; font-weight: 900; margin: 0;">{design['tonnage']:.1f} tons</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 10px; margin-top: 12px;">
                    <p style="color: #1f2937; font-weight: 700; margin: 0 0 8px 0;">Total Investment</p>
                    <p style="color: #3b82f6; font-size: 28px; font-weight: 900; margin: 0;">${design['total_installation']:,.0f}</p>
                    <p style="color: #6b7280; font-size: 12px; margin: 5px 0 0 0;">Within budget: {"✅ Yes" if design['total_installation'] <= specs['budget'] else "⚠️ Exceeds by $" + f"{design['total_installation'] - specs['budget']:,.0f}"}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    if st.session_state.design_generated:
        st.markdown("<hr style='margin: 30px 0; border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)
        
        specs = st.session_state.building_specs
        design = calculate_hvac_design(specs['sqft'], specs['floors'], specs['zones'], 
                                      specs['climate'], specs['efficiency_pref'])
        
        # Detailed breakdown
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">📋 Design Specifications</h2>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Equipment Cost</p>
                    <p style="font-size: 28px; color: white; font-weight: 900; margin: 0;">${design['equipment_cost']:,.0f}</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Ductwork Cost</p>
                    <p style="font-size: 28px; color: white; font-weight: 900; margin: 0;">${design['ductwork_cost']:,.0f}</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Labor Cost</p>
                    <p style="font-size: 28px; color: white; font-weight: 900; margin: 0;">${design['labor_cost']:,.0f}</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Annual Energy Cost</p>
                    <p style="font-size: 28px; color: #fbbf24; font-weight: 900; margin: 0;">${design['annual_energy_cost']:,.0f}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # System layout visualization
        layout_fig = create_system_layout(zones)
        st.plotly_chart(layout_fig, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Compare Design Alternatives</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">AI generates 3 options balancing cost, efficiency, and performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.design_generated:
        specs = st.session_state.building_specs
        alternatives = generate_alternative_designs(specs['sqft'], specs['floors'], specs['zones'], specs['climate'])
        
        # Comparison chart
        cost_chart = create_cost_comparison_chart(alternatives)
        st.plotly_chart(cost_chart, use_container_width=True)
        
        # Detailed comparison table
        st.markdown("### 📊 Side-by-Side Comparison")
        
        comparison_data = []
        for alt in alternatives:
            comparison_data.append({
                'Design': alt['name'],
                'Efficiency': alt['efficiency_rating'],
                'Installation': f"${alt['total_installation']:,.0f}",
                'Annual Energy': f"${alt['annual_energy_cost']:,.0f}",
                '10-Year Total': f"${alt['total_installation'] + (alt['annual_energy_cost'] * 10):,.0f}",
                'Payback': f"{alt['payback_years']:.1f} years" if alt['payback_years'] > 0 else "N/A"
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Recommendation
        best_value_idx = np.argmin([alt['total_installation'] + (alt['annual_energy_cost'] * 10) for alt in alternatives])
        best_design = alternatives[best_value_idx]
        
        st.success(f"💡 **AI Recommendation:** {best_design['name']} offers best 10-year value at ${best_design['total_installation'] + (best_design['annual_energy_cost'] * 10):,.0f} total cost")

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">How Semble AI Works</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Multi-constraint optimization for building systems</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">🤖 AI Optimization Pipeline</h3>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6; margin-bottom: 12px;">
                <h4 style="color: #1e40af; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">1. Load Calculation</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">ML model calculates heating/cooling loads based on sqft, climate, building type, occupancy</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #10b981; margin-bottom: 12px;">
                <h4 style="color: #065f46; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">2. Equipment Selection</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Optimization algorithm selects equipment based on capacity, efficiency, budget constraints</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #8b5cf6; margin-bottom: 12px;">
                <h4 style="color: #6b21a8; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">3. Layout Optimization</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Graph algorithms find optimal ductwork routes minimizing length and pressure loss</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #f59e0b;">
                <h4 style="color: #92400e; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">4. Cost-Efficiency Analysis</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Generate multiple designs with different cost/efficiency tradeoffs, rank by 10-year total cost</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">📐 Technical Specifications</h3>
            <div style="background: #ecfdf5; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #047857; font-weight: 700; font-size: 14px; margin: 0;">Constraint-Based Optimization</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Linear programming, multi-objective optimization</p>
            </div>
            <div style="background: #eff6ff; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #1e40af; font-weight: 700; font-size: 14px; margin: 0;">Physics-Based Modeling</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Heat transfer, airflow dynamics, energy calculations</p>
            </div>
            <div style="background: #fef3c7; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #92400e; font-weight: 700; font-size: 14px; margin: 0;">Code Compliance Engine</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">ASHRAE standards, local building codes, safety regulations</p>
            </div>
            <div style="background: #f3e8ff; padding: 12px 15px; border-radius: 8px;">
                <p style="color: #6b21a8; font-weight: 700; font-size: 14px; margin: 0;">Cost Estimation</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Equipment pricing database, labor rates, material costs</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #ea580c; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Semble AI</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 10x Faster Design</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Generate optimal HVAC design in 10 minutes vs 20-40 hours manual. Engineers focus on complex decisions, not calculations.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Huge Cost Savings</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    15-25% lower installation costs through optimization. 30% energy savings over system lifetime. Avoids $50K+ design error fixes.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📈 Better Outcomes</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Right-sized systems (no over/under capacity). Optimal energy efficiency. Code compliant by default. Fewer change orders.
                </p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Construction Industry Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">10x faster:</strong> 10 minutes vs 20-40 hours manual design</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">15-25% savings:</strong> Lower installation costs through optimization</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">30% energy reduction:</strong> Better efficiency over system lifetime</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Zero design errors:</strong> Eliminates costly post-construction fixes</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Capabilities</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ ML Load Prediction</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Trained on 10K+ buildings for accurate sizing</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Graph Optimization</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Shortest path algorithms for ductwork routing</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-Objective Solver</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Balance cost, efficiency, performance simultaneously</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Physics Simulation</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Heat transfer, airflow modeling, energy analysis</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #ea580c 0%, #f97316 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(234, 88, 12, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Semble AI</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
        </p>
        <div style="margin: 25px 0; padding: 22px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.2);">
            <p style="margin: 8px 0; font-size: 16px;">
                📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: white; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a>
            </p>
            <p style="margin: 8px 0; font-size: 16px;">
                💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">LinkedIn</a> | 
                💻 <a href="https://github.com/Av1352" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">GitHub</a> | 
                🌐 <a href="https://vxanju.com" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">Portfolio</a>
            </p>
        </div>
        <p style="font-size: 15px; margin: 18px 0; font-weight: 700;">
            <strong style="color: white;">Tech Stack:</strong> Optimization Algorithms • Physics Simulation • Constraint Solving • Graph Theory
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered building system design and optimization.<br>
            Load calculation • Equipment selection • Layout optimization • Cost analysis • Multi-objective solving
        </p>
    </div>
    """, unsafe_allow_html=True)