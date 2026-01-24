"""
Noho Labs - Enterprise AI Platform
AI-powered business intelligence and automation
Built for Noho Labs by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from utils.sidebar import render_sidebar
render_sidebar()

st.set_page_config(page_title="Noho Labs - Enterprise AI", layout="wide")

# Initialize session state
if 'dashboard_loaded' not in st.session_state:
    st.session_state.dashboard_loaded = False
if 'recommendation_generated' not in st.session_state:
    st.session_state.recommendation_generated = False

def generate_enterprise_data():
    """Generate sample enterprise AI data"""
    np.random.seed(42)
    
    departments = ['Sales', 'Marketing', 'Operations', 'Finance', 'HR', 'Customer Success']
    ai_tasks = {
        'Sales': ['Lead Scoring', 'Pipeline Forecasting', 'Customer Segmentation'],
        'Marketing': ['Campaign Optimization', 'Content Generation', 'Audience Targeting'],
        'Operations': ['Process Automation', 'Inventory Optimization', 'Quality Control'],
        'Finance': ['Fraud Detection', 'Budget Forecasting', 'Expense Classification'],
        'HR': ['Resume Screening', 'Employee Churn Prediction', 'Performance Analysis'],
        'Customer Success': ['Ticket Routing', 'Churn Prevention', 'Satisfaction Prediction']
    }
    
    data = []
    for dept in departments:
        for task in ai_tasks[dept]:
            accuracy = np.random.uniform(0.88, 0.97)
            time_saved_hours = np.random.uniform(50, 500)
            cost_savings = time_saved_hours * 75
            automation_rate = np.random.uniform(0.70, 0.95)
            
            data.append({
                'department': dept,
                'ai_task': task,
                'accuracy': round(accuracy * 100, 1),
                'time_saved_hours': round(time_saved_hours, 0),
                'cost_savings': round(cost_savings, 0),
                'automation_rate': round(automation_rate * 100, 1),
                'status': 'Active'
            })
    
    return pd.DataFrame(data)

def analyze_enterprise_ai(df):
    """Analyze enterprise AI performance"""
    total_tasks = len(df)
    avg_accuracy = df['accuracy'].mean()
    total_time_saved = df['time_saved_hours'].sum()
    total_cost_savings = df['cost_savings'].sum()
    avg_automation = df['automation_rate'].mean()
    
    by_department = df.groupby('department').agg({
        'accuracy': 'mean',
        'time_saved_hours': 'sum',
        'cost_savings': 'sum',
        'automation_rate': 'mean'
    })
    
    return total_tasks, avg_accuracy, total_time_saved, total_cost_savings, avg_automation, by_department

def create_department_chart(by_department):
    """Create cost savings by department"""
    fig = go.Figure(data=[
        go.Bar(x=by_department.index, y=by_department['cost_savings'],
               marker=dict(color=['#667eea', '#764ba2', '#ec4899', '#10b981', '#f59e0b', '#3b82f6']))
    ])
    fig.update_layout(
        title="Cost Savings by Department",
        xaxis_title="Department",
        yaxis_title="Annual Savings ($)",
        height=400
    )
    return fig

def create_automation_chart(by_department):
    """Create automation rate chart"""
    fig = go.Figure(data=[
        go.Bar(x=by_department.index, y=by_department['automation_rate'],
               marker=dict(color='#10b981'))
    ])
    fig.update_layout(
        title="Automation Rate by Department",
        xaxis_title="Department",
        yaxis_title="Automation Rate (%)",
        height=400
    )
    fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Target: 80%")
    return fig

def generate_ai_recommendation(department, use_case):
    """Generate AI solution recommendation"""
    
    solutions = {
        'Sales': {
            'Lead Scoring': {
                'model': 'Gradient Boosting Classifier',
                'accuracy': 94,
                'time_savings': 320,
                'features': ['Company size', 'Industry', 'Engagement score', 'Budget', 'Timeline']
            },
            'Revenue Forecasting': {
                'model': 'LSTM Time Series',
                'accuracy': 91,
                'time_savings': 180,
                'features': ['Historical revenue', 'Pipeline value', 'Seasonality', 'Market trends']
            },
            'Customer Segmentation': {
                'model': 'K-Means Clustering',
                'accuracy': 88,
                'time_savings': 250,
                'features': ['Purchase history', 'Engagement', 'Demographics', 'Lifetime value']
            }
        },
        'Marketing': {
            'Campaign ROI Prediction': {
                'model': 'Random Forest Regressor',
                'accuracy': 89,
                'time_savings': 200,
                'features': ['Channel mix', 'Budget', 'Audience size', 'Historical ROI', 'Timing']
            },
            'Content Optimization': {
                'model': 'GPT-4 Fine-tuned',
                'accuracy': 93,
                'time_savings': 400,
                'features': ['Topic', 'Audience', 'Tone', 'Format', 'Platform']
            },
            'Churn Prevention': {
                'model': 'XGBoost Classifier',
                'accuracy': 92,
                'time_savings': 280,
                'features': ['Usage patterns', 'Support tickets', 'NPS score', 'Feature adoption']
            }
        },
        'Operations': {
            'Demand Forecasting': {
                'model': 'Prophet Time Series',
                'accuracy': 90,
                'time_savings': 350,
                'features': ['Historical demand', 'Seasonality', 'Promotions', 'External events']
            },
            'Quality Control': {
                'model': 'CNN Image Classification',
                'accuracy': 96,
                'time_savings': 450,
                'features': ['Visual inspection', 'Defect detection', 'Pattern recognition']
            },
            'Process Automation': {
                'model': 'RPA + ML',
                'accuracy': 94,
                'time_savings': 500,
                'features': ['Task frequency', 'Rule complexity', 'Error rate', 'Data quality']
            }
        },
        'Finance': {
            'Fraud Detection': {
                'model': 'Isolation Forest + Neural Network',
                'accuracy': 95,
                'time_savings': 380,
                'features': ['Transaction amount', 'Location', 'Timing', 'Merchant', 'History']
            },
            'Budget Optimization': {
                'model': 'Linear Programming + ML',
                'accuracy': 88,
                'time_savings': 220,
                'features': ['Department budgets', 'Historical spend', 'Goals', 'Constraints']
            },
            'Expense Classification': {
                'model': 'Transformer NLP',
                'accuracy': 93,
                'time_savings': 300,
                'features': ['Receipt text', 'Merchant', 'Amount', 'Category history']
            }
        },
        'HR': {
            'Resume Screening': {
                'model': 'BERT NLP',
                'accuracy': 91,
                'time_savings': 420,
                'features': ['Skills match', 'Experience', 'Education', 'Keywords', 'Culture fit']
            },
            'Attrition Prediction': {
                'model': 'Ensemble Classifier',
                'accuracy': 89,
                'time_savings': 260,
                'features': ['Tenure', 'Satisfaction', 'Promotion history', 'Compensation', 'Manager']
            },
            'Performance Analysis': {
                'model': 'Multi-task Learning',
                'accuracy': 87,
                'time_savings': 190,
                'features': ['KPIs', 'Peer feedback', 'Project outcomes', 'Skills development']
            }
        },
        'Customer Success': {
            'Ticket Prioritization': {
                'model': 'Priority Classifier',
                'accuracy': 92,
                'time_savings': 340,
                'features': ['Issue type', 'Customer tier', 'Urgency', 'Impact', 'SLA']
            },
            'Satisfaction Prediction': {
                'model': 'Sentiment Analysis + Regression',
                'accuracy': 90,
                'time_savings': 210,
                'features': ['Interaction history', 'Resolution time', 'Tone', 'Outcome']
            },
            'Upsell Opportunities': {
                'model': 'Propensity Scoring',
                'accuracy': 88,
                'time_savings': 290,
                'features': ['Usage patterns', 'Feature requests', 'Contract value', 'Engagement']
            }
        }
    }
    
    solution = solutions.get(department, {}).get(use_case, solutions['Sales']['Lead Scoring'])
    
    annual_time_savings = solution['time_savings'] * 12
    annual_cost_savings = annual_time_savings * 75
    implementation_cost = 50000
    annual_roi = ((annual_cost_savings - implementation_cost) / implementation_cost) * 100
    
    feature_cards = []
    for feature in solution['features']:
        card = f'<div style="background: #f0fdf4; border-left: 4px solid #10b981; border-radius: 8px; padding: 10px;"><p style="font-size: 14px; color: #065f46; font-weight: 600; margin: 0;">✓ {feature}</p></div>'
        feature_cards.append(card)
    
    all_features = ''.join(feature_cards)
    
    recommendation_html = f'<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;"><h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🤖 AI Solution Recommendation</h2><div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;"><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Department</p><p style="font-size: 32px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{department}</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Use Case</p><p style="font-size: 24px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{use_case}</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Model Type</p><p style="font-size: 20px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{solution["model"]}</p></div></div></div><div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;"><h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📊 Expected Performance</h3><div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;"><div style="background: white; border-radius: 12px; padding: 18px; text-align: center;"><p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Accuracy</p><p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">{solution["accuracy"]}%</p></div><div style="background: white; border-radius: 12px; padding: 18px; text-align: center;"><p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Time Saved/Month</p><p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">{solution["time_savings"]}h</p></div><div style="background: white; border-radius: 12px; padding: 18px; text-align: center;"><p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Annual Savings</p><p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">${annual_cost_savings/1000:.0f}K</p></div><div style="background: white; border-radius: 12px; padding: 18px; text-align: center;"><p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">ROI</p><p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">{annual_roi:.0f}%</p></div></div><div style="background: white; border-radius: 12px; padding: 20px;"><p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0 0 12px 0;">Key Features Used</p><div style="display: grid; gap: 8px;">{all_features}</div></div></div><div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2);"><h3 style="color: #92400e; font-size: 22px; font-weight: 900; margin: 0 0 15px 0;">💡 Implementation Plan</h3><div style="display: grid; gap: 12px;"><div style="background: white; border-radius: 10px; padding: 16px;"><p style="font-size: 15px; color: #1f2937; font-weight: 700; margin: 0 0 8px 0;">Phase 1: Data Collection (2 weeks)</p><p style="font-size: 13px; color: #6b7280; margin: 0;">Gather historical data, clean and prepare datasets, establish baseline metrics</p></div><div style="background: white; border-radius: 10px; padding: 16px;"><p style="font-size: 15px; color: #1f2937; font-weight: 700; margin: 0 0 8px 0;">Phase 2: Model Training (3 weeks)</p><p style="font-size: 13px; color: #6b7280; margin: 0;">Train and validate ML models, optimize hyperparameters, conduct A/B testing</p></div><div style="background: white; border-radius: 10px; padding: 16px;"><p style="font-size: 15px; color: #1f2937; font-weight: 700; margin: 0 0 8px 0;">Phase 3: Deployment (2 weeks)</p><p style="font-size: 13px; color: #6b7280; margin: 0;">Integrate with existing systems, train team, monitor performance</p></div><div style="background: white; border-radius: 10px; padding: 16px;"><p style="font-size: 15px; color: #1f2937; font-weight: 700; margin: 0 0 8px 0;">Phase 4: Optimization (Ongoing)</p><p style="font-size: 13px; color: #6b7280; margin: 0;">Continuous model retraining, feature enhancement, ROI tracking</p></div></div></div>'
    
    return recommendation_html

def process_dashboard():
    """Generate complete enterprise AI dashboard"""
    df = generate_enterprise_data()
    total_tasks, avg_accuracy, total_time_saved, total_cost_savings, avg_automation, by_department = analyze_enterprise_ai(df)
    
    department_chart = create_department_chart(by_department)
    automation_chart = create_automation_chart(by_department)
    
    return total_tasks, avg_accuracy, total_cost_savings, avg_automation, by_department, department_chart, automation_chart, df

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🏢</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Noho Labs Enterprise AI
        </h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Business Intelligence • AI Automation</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Transform enterprise operations with AI-powered solutions</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">18 AI Tasks</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">92% Accuracy</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">$1.6M Saved</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">82% Automated</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Noho Labs</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📊 Enterprise Dashboard", "🤖 AI Solution Builder"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">AI Performance Overview</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Track 18 AI tasks across 6 departments</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Load Dashboard", type="primary", use_container_width=True):
        st.session_state.dashboard_loaded = True
    
    if st.session_state.dashboard_loaded:
        total_tasks, avg_accuracy, total_cost_savings, avg_automation, by_department, department_chart, automation_chart, df = process_dashboard()
        
        # Build department cards
        colors = ['#667eea', '#10b981', '#ec4899', '#f59e0b', '#3b82f6', '#8b5cf6']
        dept_cards = []
        
        for idx, (dept, row) in enumerate(by_department.iterrows()):
            card = f'<div style="background: white; border-left: 5px solid {colors[idx % len(colors)]}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);"><div style="display: flex; justify-content: space-between; align-items: center;"><div><p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 4px 0;">{dept}</p><p style="font-size: 13px; color: #6b7280; margin: 0;">{row["accuracy"]:.0f}% accuracy • {row["automation_rate"]:.0f}% automated</p></div><div style="text-align: right;"><p style="font-size: 28px; color: {colors[idx % len(colors)]}; font-weight: 900; margin: 0;">${row["cost_savings"]/1000:.0f}K</p><p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">{row["time_saved_hours"]:.0f}h saved</p></div></div></div>'
            dept_cards.append(card)
        
        all_dept_cards = ''.join(dept_cards)
        
        summary_html = f'<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;"><h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🏢 Enterprise AI Dashboard</h2><div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;"><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">AI Tasks</p><p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{total_tasks}</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Active</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Avg Accuracy</p><p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{avg_accuracy:.0f}%</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Performance</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Cost Savings</p><p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">${total_cost_savings/1000:.0f}K</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Annual</p></div><div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);"><p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Automation</p><p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{avg_automation:.0f}%</p><p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Rate</p></div></div></div><div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;"><h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📊 Department Performance</h3><div style="display: grid; gap: 12px;">{all_dept_cards}</div></div>'
        
        st.markdown(summary_html, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(department_chart, use_container_width=True)
        with col2:
            st.plotly_chart(automation_chart, use_container_width=True)
        
        st.dataframe(df, use_container_width=True, height=400)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Get AI Recommendations</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Select your department and use case for custom AI solution</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        dept_input = st.selectbox(
            "Department",
            ['Sales', 'Marketing', 'Operations', 'Finance', 'HR', 'Customer Success'],
            index=0
        )
    
    with col2:
        use_case_input = st.selectbox(
            "Use Case",
            ['Lead Scoring', 'Revenue Forecasting', 'Customer Segmentation',
             'Campaign ROI Prediction', 'Content Optimization', 'Churn Prevention',
             'Demand Forecasting', 'Quality Control', 'Process Automation',
             'Fraud Detection', 'Budget Optimization', 'Expense Classification',
             'Resume Screening', 'Attrition Prediction', 'Performance Analysis',
             'Ticket Prioritization', 'Satisfaction Prediction', 'Upsell Opportunities'],
            index=0
        )
    
    if st.button("🚀 Get AI Solution", type="primary", use_container_width=True):
        st.session_state.recommendation_generated = True
        st.session_state.rec_params = (dept_input, use_case_input)
    
    if st.session_state.recommendation_generated:
        recommendation_html = generate_ai_recommendation(*st.session_state.rec_params)
        st.markdown(recommendation_html, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Noho Labs</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 $1.6M Annual Savings</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    18 AI tasks save 4,900+ hours annually. At $75/hour, that's $1.6M in cost reduction across enterprise.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🎯 92% Accuracy</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Production-ready ML models across sales, marketing, ops, finance, HR, customer success.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 82% Automation</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Free teams from repetitive tasks. Focus on strategic work while AI handles routine operations.
                </p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ 18 AI Use Cases</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Across 6 departments with custom models</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-Model Support</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">XGBoost, LSTM, BERT, CNN, GPT-4</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ ROI Tracking</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Time saved, cost savings, automation rate</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Implementation Plans</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">4-phase deployment roadmap</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Noho Labs</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • XGBoost • BERT • LSTM • CNN • GPT-4 • Streamlit
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing enterprise AI automation platform.<br>
            18 AI use cases • 6 departments • $1.6M savings • 92% accuracy • 82% automation
        </p>
    </div>
    """, unsafe_allow_html=True)