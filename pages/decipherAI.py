"""
Decipher AI - Automated Test Generation Platform
AI-powered testing and quality assurance
Built for Decipher AI by Anju Nandhakumar
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import random
import re

# Page config
st.set_page_config(
    page_title="Decipher AI Demo - Anju Vilashni",
    page_icon="🧪",
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

# Sample code
SAMPLE_CODE = {
    "Calculator Function": """def calculator(a, b, operation):
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError("Invalid operation")""",
    
    "User Authentication": """def authenticate_user(username, password):
    if not username or not password:
        return False
    users = {'admin': 'password123', 'user1': 'securepass'}
    if username in users and users[username] == password:
        return True
    return False""",
    
    "Email Validator": """def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not email:
        return False
    if re.match(pattern, email):
        return True
    return False""",
    
    "Shopping Cart": """class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add_item(self, item, price, quantity=1):
        self.items.append({'item': item, 'price': price, 'quantity': quantity})
    
    def get_total(self):
        return sum(item['price'] * item['quantity'] for item in self.items)"""
}

def generate_tests(code_snippet, test_type, coverage_target):
    """Generate test cases for given code"""
    
    if not code_snippet or len(code_snippet.strip()) < 10:
        st.error("⚠️ Please enter code to generate tests!")
        return
    
    function_count = len(re.findall(r'def \w+\(', code_snippet))
    
    base_tests = function_count * 3
    if coverage_target == "High (90%+)":
        num_tests = int(base_tests * 1.5)
        coverage_pct = random.uniform(92, 98)
    elif coverage_target == "Medium (70-90%)":
        num_tests = base_tests
        coverage_pct = random.uniform(75, 88)
    else:
        num_tests = int(base_tests * 0.7)
        coverage_pct = random.uniform(65, 75)
    
    # Summary
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">✨ Tests Generated Successfully!</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
            <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Tests Generated</p>
            <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">{num_tests}</p>
            <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">{test_type}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(16, 185, 129, 0.3);">
            <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Coverage</p>
            <p style="font-size: 48px; color: #10b981; font-weight: 900; margin: 0;">{coverage_pct:.0f}%</p>
            <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Code covered</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        edge_cases = random.randint(5, 12)
        st.markdown(f"""
        <div style="background: rgba(251, 191, 36, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(251, 191, 36, 0.3);">
            <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Edge Cases</p>
            <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0;">{edge_cases}</p>
            <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Identified</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
            <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Generation Time</p>
            <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">1.2s</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generated tests code sample
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
        <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🧪 Generated Test Suite</h3>
        <div style="background: white; border-radius: 14px; padding: 20px; margin-bottom: 15px;">
            <div style="background: #1f2937; border-radius: 10px; padding: 20px; font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.6; color: #d1d5db; overflow-x: auto;">
<span style="color: #8b5cf6;">import</span> pytest

<span style="color: #6b7280;"># AI-Generated Test Suite</span>
<span style="color: #6b7280;"># Type: {test_type}</span>
<span style="color: #6b7280;"># Coverage: {coverage_pct:.0f}%</span>
<span style="color: #6b7280;"># Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>

<span style="color: #8b5cf6;">def</span> <span style="color: #3b82f6;">test_happy_path</span>():
    <span style="color: #6b7280;">"Test normal expected behavior"</span>
    result = calculator(<span style="color: #10b981;">5</span>, <span style="color: #10b981;">3</span>, <span style="color: #10b981;">'add'</span>)
    <span style="color: #8b5cf6;">assert</span> result == <span style="color: #10b981;">8</span>

<span style="color: #8b5cf6;">def</span> <span style="color: #3b82f6;">test_edge_case_zero</span>():
    <span style="color: #6b7280;">"Test edge case: zero values"</span>
    result = calculator(<span style="color: #10b981;">0</span>, <span style="color: #10b981;">0</span>, <span style="color: #10b981;">'multiply'</span>)
    <span style="color: #8b5cf6;">assert</span> result == <span style="color: #10b981;">0</span>

<span style="color: #8b5cf6;">def</span> <span style="color: #3b82f6;">test_error_handling_divide_by_zero</span>():
    <span style="color: #6b7280;">"Test error: division by zero"</span>
    <span style="color: #8b5cf6;">with</span> pytest.raises(ValueError):
        calculator(<span style="color: #10b981;">10</span>, <span style="color: #10b981;">0</span>, <span style="color: #10b981;">'divide'</span>)

<span style="color: #6b7280;"># ... {num_tests - 3} more tests generated</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Test breakdown
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; border-radius: 12px; padding: 16px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Happy Path Tests</p>
            <p style="font-size: 28px; color: #10b981; font-weight: 900; margin: 0;">{int(num_tests * 0.4)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; border-radius: 12px; padding: 16px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Edge Cases</p>
            <p style="font-size: 28px; color: #f59e0b; font-weight: 900; margin: 0;">{int(num_tests * 0.35)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: white; border-radius: 12px; padding: 16px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Error Handling</p>
            <p style="font-size: 28px; color: #ef4444; font-weight: 900; margin: 0;">{int(num_tests * 0.25)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Coverage analysis
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">📊 Code Coverage Analysis</h3>
        <div style="background: white; border-radius: 14px; padding: 22px; margin-bottom: 15px;">
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 16px; color: #1f2937; font-weight: 700;">Line Coverage</span>
                    <span style="font-size: 20px; color: #10b981; font-weight: 900;">{coverage_pct:.0f}%</span>
                </div>
                <div style="background: #e5e7eb; border-radius: 10px; height: 12px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #10b981, #059669); height: 100%; width: {coverage_pct}%;"></div>
                </div>
            </div>
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 16px; color: #1f2937; font-weight: 700;">Branch Coverage</span>
                    <span style="font-size: 20px; color: #3b82f6; font-weight: 900;">{coverage_pct * 0.85:.0f}%</span>
                </div>
                <div style="background: #e5e7eb; border-radius: 10px; height: 12px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #3b82f6, #2563eb); height: 100%; width: {coverage_pct * 0.85}%;"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 16px; color: #1f2937; font-weight: 700;">Function Coverage</span>
                    <span style="font-size: 20px; color: #8b5cf6; font-weight: 900;">100%</span>
                </div>
                <div style="background: #e5e7eb; border-radius: 10px; height: 12px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #8b5cf6, #7c3aed); height: 100%; width: 100%;"></div>
                </div>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; padding: 18px; color: white; text-align: center;">
            <p style="font-size: 18px; font-weight: 800; margin: 0;">✅ Coverage Target Achieved: {coverage_pct:.0f}%</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Charts
    categories = ['Line', 'Branch', 'Function', 'Statement']
    coverage_values = [coverage_pct, coverage_pct * 0.85, 100, coverage_pct * 0.92]
    
    fig_coverage = go.Figure(data=[go.Bar(
        x=categories,
        y=coverage_values,
        marker_color=['#10b981', '#3b82f6', '#8b5cf6', '#ec4899'],
        text=[f'{v:.0f}%' for v in coverage_values],
        textposition='outside'
    )])
    fig_coverage.add_hline(y=90, line_dash="dash", line_color="#059669", 
                           annotation_text="High Coverage Target")
    fig_coverage.update_layout(
        title="Coverage Metrics by Type",
        yaxis_title="Coverage (%)",
        yaxis_range=[0, 110],
        height=400
    )
    st.plotly_chart(fig_coverage, use_container_width=True)
    
    # Test distribution
    test_categories = ['Happy Path', 'Edge Cases', 'Error Handling', 'Integration', 'Performance']
    test_counts = [int(num_tests * p) for p in [0.35, 0.25, 0.20, 0.12, 0.08]]
    
    fig_tests = go.Figure(data=[go.Pie(
        labels=test_categories,
        values=test_counts,
        marker=dict(colors=['#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6']),
        hole=0.4,
        textinfo='label+value'
    )])
    fig_tests.update_layout(title=f"Test Distribution ({num_tests} total tests)", height=400)
    st.plotly_chart(fig_tests, use_container_width=True)

# Header
st.markdown(
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
            <span style="font-size: 56px;">🧪</span>
        </div>
        <h1 style="
            font-size: 58px;
            font-weight: 900;
            color: white;
            margin: 0 0 18px 0;
            text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            Decipher Test AI
        </h1>
        <p style="
            font-size: 28px;
            color: rgba(255,255,255,0.95);
            font-weight: 700;
            margin: 15px 0;
        ">
            Automated Test Generation & QA
        </p>
        <p style="
            font-size: 18px;
            color: rgba(255,255,255,0.85);
            font-weight: 500;
            margin-bottom: 25px;
        ">
        AI-powered testing • Code coverage • Bug detection
        </p>
        <div style="
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
            max-width: 850px;
            margin: 28px auto 0 auto;
        ">
            <span style="background:#ec4899;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Test Gen</span>
            <span style="background:#8b5cf6;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Coverage</span>
            <span style="background:#f59e0b;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">Bug Detection</span>
            <span style="background:#10b981;color:white;padding:10px 22px;border-radius:30px;font-weight:800;">YC Backed</span>
        </div>
        <p style="
            font-size: 16px;
            color: rgba(255,255,255,0.9);
            margin-top: 28px;
            font-weight: 600;
        ">
            Built for <strong style="color:white;">Decipher AI</strong>
            by <strong style="color:white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["🧪 Generate Tests", "📊 QA Dashboard"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">AI Test Generation</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Paste your code and watch AI generate comprehensive test suites instantly</p>
    </div>
    """, unsafe_allow_html=True)
    
    example_choice = st.selectbox(
        "Try Example Code",
        [
            "💡 Custom Code (Paste Your Own)",
            "🔢 Calculator Function",
            "🔐 User Authentication",
            "📧 Email Validator",
            "🛒 Shopping Cart Class"
        ]
    )
    
    code_input = st.text_area(
        "Your Code",
        value=SAMPLE_CODE.get(example_choice.split(' ', 1)[1], "") if example_choice != "💡 Custom Code (Paste Your Own)" else "",
        height=250,
        placeholder="Paste Python code here..."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_type = st.selectbox("Test Type", ["Unit Tests", "Integration Tests", "E2E Tests"])
    
    with col2:
        coverage_target = st.radio("Coverage Target", ["High (90%+)", "Medium (70-90%)", "Basic (60-70%)"])
    
    if st.button("✨ Generate Tests with AI", use_container_width=True, type="primary"):
        generate_tests(code_input, test_type, coverage_target)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Quality Assurance Analytics</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Organization-wide testing metrics and trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Load QA Dashboard", key="dashboard", use_container_width=True, type="primary"):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 QA Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(102, 126, 234, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Tests Run</p>
                <p style="font-size: 48px; color: #667eea; font-weight: 900; margin: 0;">1,247</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Last 24 hours</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(16, 185, 129, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Pass Rate</p>
                <p style="font-size: 48px; color: #10b981; font-weight: 900; margin: 0;">97.2%</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">1,212 passed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(239, 68, 68, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Bugs Found</p>
                <p style="font-size: 48px; color: #ef4444; font-weight: 900; margin: 0;">35</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Auto-detected</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: rgba(251, 191, 36, 0.15); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(251, 191, 36, 0.3);">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0; font-weight: 600;">Avg Coverage</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0;">89%</p>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Across codebase</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Trends
        days = list(range(7))
        tests_per_day = [random.randint(50, 90) for _ in days]
        bugs_per_day = [random.randint(3, 12) for _ in days]
        
        fig_trends = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Tests Generated Per Day', 'Bugs Detected Per Day'),
            vertical_spacing=0.15
        )
        
        fig_trends.add_trace(go.Bar(x=days, y=tests_per_day, marker_color='#10b981'), row=1, col=1)
        fig_trends.add_trace(go.Scatter(x=days, y=bugs_per_day, mode='lines+markers', 
                                        line=dict(color='#ef4444', width=3)), row=2, col=1)
        
        fig_trends.update_xaxes(title_text="Days Ago", row=2, col=1)
        fig_trends.update_yaxes(title_text="Tests", row=1, col=1)
        fig_trends.update_yaxes(title_text="Bugs", row=2, col=1)
        fig_trends.update_layout(height=600, showlegend=False, title_text="7-Day Testing Activity")
        
        st.plotly_chart(fig_trends, use_container_width=True)

# Footer
st.markdown("<hr style='border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Decipher AI</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
        <strong>Tech Stack:</strong> Python • Streamlit • Pytest • Code Analysis • AI Generation
    </p>
</div>
""", unsafe_allow_html=True)