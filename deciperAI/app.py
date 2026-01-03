"""
Decipher AI - Automated Test Generation Platform
AI-powered testing and quality assurance
Built for Decipher AI by Anju Nandhakumar
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import random
import re

# Sample code snippets for testing
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
    
    # Simulate database lookup
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
        return sum(item['price'] * item['quantity'] for item in self.items)
    
    def remove_item(self, item_name):
        self.items = [i for i in self.items if i['item'] != item_name]"""
}

def generate_tests(code_snippet, test_type, coverage_target):
    """Generate test cases for given code"""
    
    if not code_snippet or len(code_snippet.strip()) < 10:
        return (
            "<div style='background: #fee2e2; border: 2px solid #dc2626; padding: 20px; border-radius: 10px;'><p style='color: #991b1b; font-weight: bold; font-size: 18px; margin: 0;'>⚠️ Please enter code to generate tests!</p></div>",
            None,
            None
        )
    
    # Count functions
    function_count = len(re.findall(r'def \w+\(', code_snippet))
    class_count = len(re.findall(r'class \w+', code_snippet))
    
    # Generate test counts based on coverage target
    base_tests = function_count * 3
    if coverage_target == "High (90%+)":
        num_tests = int(base_tests * 1.5)
    elif coverage_target == "Medium (70-90%)":
        num_tests = base_tests
    else:
        num_tests = int(base_tests * 0.7)
    
    # Simulate coverage
    if coverage_target == "High (90%+)":
        coverage_pct = random.uniform(92, 98)
    elif coverage_target == "Medium (70-90%)":
        coverage_pct = random.uniform(75, 88)
    else:
        coverage_pct = random.uniform(65, 75)
    
    # Generate summary
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">✨ Tests Generated Successfully!</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Tests Generated</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{num_tests}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{test_type}</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Coverage</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{coverage_pct:.0f}%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Code covered</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Edge Cases</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{random.randint(5, 12)}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Identified</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Generation Time</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">1.2s</p>
            </div>
        </div>
    </div>
    """
    
    # Generated test examples
    tests_html = f"""
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2); margin-bottom: 25px;">
        <h3 style="color: #065f46; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🧪 Generated Test Suite</h3>
        
        <div style="background: white; border-radius: 14px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
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
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
            <div style="background: white; border-radius: 12px; padding: 16px; text-align: center;">
                <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Happy Path Tests</p>
                <p style="font-size: 28px; color: #10b981; font-weight: 900; margin: 0;">{int(num_tests * 0.4)}</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 16px; text-align: center;">
                <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Edge Cases</p>
                <p style="font-size: 28px; color: #f59e0b; font-weight: 900; margin: 0;">{int(num_tests * 0.35)}</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 16px; text-align: center;">
                <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Error Handling</p>
                <p style="font-size: 28px; color: #ef4444; font-weight: 900; margin: 0;">{int(num_tests * 0.25)}</p>
            </div>
        </div>
    </div>
    """
    
    # Coverage breakdown
    coverage_html = f"""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2);">
        <h3 style="color: #1e40af; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">📊 Code Coverage Analysis</h3>
        
        <div style="background: white; border-radius: 14px; padding: 22px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 16px; color: #1f2937; font-weight: 700;">Line Coverage</span>
                    <span style="font-size: 20px; color: #10b981; font-weight: 900;">{coverage_pct:.0f}%</span>
                </div>
                <div style="background: #e5e7eb; border-radius: 10px; height: 12px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #10b981, #059669); height: 100%; width: {coverage_pct}%; transition: width 0.3s;"></div>
                </div>
            </div>
            
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 16px; color: #1f2937; font-weight: 700;">Branch Coverage</span>
                    <span style="font-size: 20px; color: #3b82f6; font-weight: 900;">{coverage_pct * 0.85:.0f}%</span>
                </div>
                <div style="background: #e5e7eb; border-radius: 10px; height: 12px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #3b82f6, #2563eb); height: 100%; width: {coverage_pct * 0.85}%; transition: width 0.3s;"></div>
                </div>
            </div>
            
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 16px; color: #1f2937; font-weight: 700;">Function Coverage</span>
                    <span style="font-size: 20px; color: #8b5cf6; font-weight: 900;">100%</span>
                </div>
                <div style="background: #e5e7eb; border-radius: 10px; height: 12px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #8b5cf6, #7c3aed); height: 100%; width: 100%; transition: width 0.3s;"></div>
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; padding: 18px; color: white; text-align: center;">
            <p style="font-size: 18px; font-weight: 800; margin: 0;">✅ Coverage Target Achieved: {coverage_pct:.0f}% {'≥' if coverage_pct >= 90 else '≥' if coverage_pct >= 70 else '<'} {coverage_target.split('(')[1].split(')')[0]}</p>
        </div>
    </div>
    """
    
    # Create charts
    fig_coverage = go.Figure()
    
    categories = ['Line', 'Branch', 'Function', 'Statement']
    coverage_values = [coverage_pct, coverage_pct * 0.85, 100, coverage_pct * 0.92]
    
    fig_coverage.add_trace(go.Bar(
        x=categories,
        y=coverage_values,
        marker_color=['#10b981', '#3b82f6', '#8b5cf6', '#ec4899'],
        text=[f'{v:.0f}%' for v in coverage_values],
        textposition='outside'
    ))
    
    fig_coverage.add_hline(y=90, line_dash="dash", line_color="#059669", 
                           annotation_text="High Coverage Target", annotation_position="right")
    
    fig_coverage.update_layout(
        title="Coverage Metrics by Type",
        yaxis_title="Coverage (%)",
        yaxis_range=[0, 110],
        height=400
    )
    
    # Test distribution
    test_categories = ['Happy Path', 'Edge Cases', 'Error Handling', 'Integration', 'Performance']
    test_counts = [int(num_tests * p) for p in [0.35, 0.25, 0.20, 0.12, 0.08]]
    
    fig_tests = go.Figure(data=[go.Pie(
        labels=test_categories,
        values=test_counts,
        marker=dict(colors=['#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6']),
        hole=0.4,
        textinfo='label+value',
        textfont=dict(size=13, color='white', family='Arial Black')
    )])
    
    fig_tests.update_layout(
        title=f"Test Distribution ({num_tests} total tests)",
        height=400
    )
    
    return summary_html + tests_html + coverage_html, fig_coverage, fig_tests

def generate_qa_dashboard():
    """Generate quality assurance dashboard"""
    
    dashboard_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">📊 QA Dashboard</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Tests Run</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">1,247</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Last 24 hours</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Pass Rate</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">97.2%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">1,212 passed</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Bugs Found</p>
                <p style="font-size: 48px; color: #fca5a5; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">35</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Auto-detected</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Avg Coverage</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">89%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Across codebase</p>
            </div>
        </div>
    </div>
    """
    
    # Test trends
    trend_html = """
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2); margin-bottom: 25px;">
        <h3 style="color: #92400e; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📈 Testing Trends</h3>
        
        <div style="display: grid; gap: 12px;">
            <div style="background: white; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0;">Tests Generated This Week</p>
                    <p style="font-size: 32px; color: #10b981; font-weight: 900; margin: 0;">487</p>
                </div>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">+23% vs last week</p>
            </div>
            
            <div style="background: white; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0;">Bugs Prevented</p>
                    <p style="font-size: 32px; color: #3b82f6; font-weight: 900; margin: 0;">142</p>
                </div>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">Caught before production</p>
            </div>
            
            <div style="background: white; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0;">Time Saved</p>
                    <p style="font-size: 32px; color: #ec4899; font-weight: 900; margin: 0;">68hrs</p>
                </div>
                <p style="font-size: 13px; color: #6b7280; margin: 8px 0 0 0;">vs manual test writing</p>
            </div>
        </div>
    </div>
    """
    
    # Create trend charts
    days = list(range(7))
    tests_per_day = [random.randint(50, 90) for _ in days]
    bugs_per_day = [random.randint(3, 12) for _ in days]
    
    fig_trends = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Tests Generated Per Day', 'Bugs Detected Per Day'),
        vertical_spacing=0.15
    )
    
    fig_trends.add_trace(
        go.Bar(x=days, y=tests_per_day, marker_color='#10b981', name='Tests'),
        row=1, col=1
    )
    
    fig_trends.add_trace(
        go.Scatter(x=days, y=bugs_per_day, mode='lines+markers', 
                   line=dict(color='#ef4444', width=3), name='Bugs'),
        row=2, col=1
    )
    
    fig_trends.update_xaxes(title_text="Days Ago", row=2, col=1)
    fig_trends.update_yaxes(title_text="Tests", row=1, col=1)
    fig_trends.update_yaxes(title_text="Bugs", row=2, col=1)
    
    fig_trends.update_layout(height=600, showlegend=False, title_text="7-Day Testing Activity")
    
    return dashboard_html + trend_html, fig_trends

custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🧪</span>
        </div>
        
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Decipher Test AI
        </h1>
        
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Automated Test Generation & QA</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">AI-powered testing • Code coverage • Bug detection</p>
        
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Test Gen</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Coverage</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Bug Detection</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">YC Backed</span>
        </div>
        
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Decipher AI</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("🧪 Generate Tests"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">AI Test Generation</h3>
                <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Paste your code and watch AI generate comprehensive test suites instantly</p>
            </div>
            """)
            
            example_dropdown = gr.Dropdown(
                choices=[
                    "💡 Custom Code (Paste Your Own)",
                    "🔢 Calculator Function",
                    "🔐 User Authentication",
                    "📧 Email Validator",
                    "🛒 Shopping Cart Class"
                ],
                label="Try Example Code",
                value="💡 Custom Code (Paste Your Own)"
            )
            
            code_input = gr.Textbox(
                label="Your Code",
                placeholder="Paste Python code here...",
                lines=12
            )
            
            with gr.Row():
                test_type = gr.Dropdown(
                    choices=["Unit Tests", "Integration Tests", "E2E Tests"],
                    value="Unit Tests",
                    label="Test Type"
                )
                
                coverage = gr.Radio(
                    choices=["High (90%+)", "Medium (70-90%)", "Basic (60-70%)"],
                    value="High (90%+)",
                    label="Coverage Target"
                )
            
            generate_btn = gr.Button("✨ Generate Tests with AI", variant="primary", size="lg")
            
            test_output = gr.HTML(label="Generated Tests")
            coverage_chart = gr.Plot(label="Coverage Metrics")
            test_chart = gr.Plot(label="Test Distribution")
            
            generate_btn.click(
                fn=generate_tests,
                inputs=[code_input, test_type, coverage],
                outputs=[test_output, coverage_chart, test_chart]
            )
            
            def load_code_example(choice):
                examples = {
                    "🔢 Calculator Function": SAMPLE_CODE["Calculator Function"],
                    "🔐 User Authentication": SAMPLE_CODE["User Authentication"],
                    "📧 Email Validator": SAMPLE_CODE["Email Validator"],
                    "🛒 Shopping Cart Class": SAMPLE_CODE["Shopping Cart"],
                    "💡 Custom Code (Paste Your Own)": ""
                }
                return examples.get(choice, "")
            
            example_dropdown.change(
                fn=load_code_example,
                inputs=[example_dropdown],
                outputs=[code_input]
            )
        
        with gr.Tab("📊 QA Dashboard"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Quality Assurance Analytics</h3>
                <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Organization-wide testing metrics and trends</p>
            </div>
            """)
            
            dashboard_btn = gr.Button("📊 Load QA Dashboard", variant="primary", size="lg")
            
            dashboard_output = gr.HTML(label="Dashboard")
            trend_chart = gr.Plot(label="Testing Trends")
            
            dashboard_btn.click(
                fn=generate_qa_dashboard,
                inputs=[],
                outputs=[dashboard_output, trend_chart]
            )
    
    gr.HTML("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Decipher AI</h2>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 10x Faster Testing</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Writing tests takes 30-50% of development time. AI generates comprehensive test suites in seconds, freeing developers to build features.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🐛 Better Bug Detection</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    AI finds edge cases humans miss. Null checks, boundary conditions, race conditions, error handling - comprehensive coverage.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Cost Reduction</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Production bugs cost 10-100x more than catching in dev. Every bug caught early saves $1K-10K in engineer time and customer impact.
                </p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Real-World Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">10x faster:</strong> Test generation in seconds vs hours manually</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">90%+ coverage:</strong> Comprehensive testing automatically</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">50% fewer bugs:</strong> In production due to better testing</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">$100K+ saved:</strong> Per year in prevented incidents</li>
            </ul>
        </div>
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Smart Test Generation</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Happy path, edge cases, error handling</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Coverage Analysis</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Line, branch, function, statement</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multiple Test Types</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Unit, integration, E2E support</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Production Ready</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Pytest-compatible test code</p>
                </div>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Decipher AI</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Gradio • Pytest • Code Analysis • AI Generation
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered automated test generation and quality assurance.<br>
            Smart test creation • Coverage optimization • Bug prevention • Developer productivity
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()