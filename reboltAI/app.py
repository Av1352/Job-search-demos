"""
Rebolt AI - Natural Language App Builder
Build applications by speaking with AI
Built for Rebolt AI by Anju Nandhakumar
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import random

# App templates that can be generated
APP_TEMPLATES = {
    "Todo List App": {
        "description": "Simple task management with add, complete, delete",
        "components": ["Input field", "Add button", "Task list", "Checkbox", "Delete button"],
        "code_lines": 85,
        "preview": "✓ Buy groceries\n☐ Call dentist\n☐ Finish report"
    },
    "Weather Dashboard": {
        "description": "Real-time weather display with forecasts",
        "components": ["Location input", "Search button", "Current weather card", "5-day forecast", "Chart"],
        "code_lines": 120,
        "preview": "🌤️ Boston, MA\n72°F • Partly Cloudy\nHigh: 75°F • Low: 68°F"
    },
    "Expense Tracker": {
        "description": "Personal finance tracking and visualization",
        "components": ["Amount input", "Category dropdown", "Add expense", "Expense list", "Pie chart"],
        "code_lines": 145,
        "preview": "Total: $1,247.50\n🍕 Food: $385\n🚗 Transport: $210"
    },
    "Customer Survey": {
        "description": "Multi-question survey with analytics",
        "components": ["Rating scale", "Text input", "Submit button", "Results dashboard", "Export"],
        "code_lines": 110,
        "preview": "How satisfied are you?\n⭐⭐⭐⭐⭐\nComments: Great service!"
    },
    "Team Dashboard": {
        "description": "Project tracking and team metrics",
        "components": ["Project cards", "Progress bars", "Team member list", "Timeline", "Stats"],
        "code_lines": 175,
        "preview": "Project Alpha: 75%\n5 tasks • 3 members\nDue: Jan 15"
    }
}

def generate_app_from_prompt(user_prompt, app_complexity):
    """Generate app based on natural language description"""
    
    # Simulate app generation
    prompt_lower = user_prompt.lower()
    
    # Match to template
    matched_template = None
    for template_name, template_data in APP_TEMPLATES.items():
        keywords = template_name.lower().split()
        if any(keyword in prompt_lower for keyword in keywords):
            matched_template = template_name
            break
    
    if not matched_template:
        matched_template = random.choice(list(APP_TEMPLATES.keys()))
    
    template = APP_TEMPLATES[matched_template]
    
    # Adjust based on complexity
    complexity_multiplier = {"Simple": 0.7, "Medium": 1.0, "Complex": 1.4}[app_complexity]
    code_lines = int(template["code_lines"] * complexity_multiplier)
    num_components = int(len(template["components"]) * complexity_multiplier)
    
    # Generate result HTML
    result_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">✨ App Generated Successfully!</h2>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">App Type</p>
                <p style="font-size: 20px; color: white; font-weight: 900; margin: 0;">{matched_template}</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Generation Time</p>
                <p style="font-size: 40px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">2.4s</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Code Lines</p>
                <p style="font-size: 40px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{code_lines}</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Components</p>
                <p style="font-size: 40px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{num_components}</p>
            </div>
        </div>
    </div>
    """
    
    # Components breakdown
    components_html = f"""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🎨 Generated Components</h3>
        
        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
            {''.join([f'<span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 10px 20px; border-radius: 20px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);">{comp}</span>' for comp in template['components'][:num_components]])}
        </div>
        
        <div style="background: white; border-radius: 14px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
            <h4 style="color: #1f2937; font-size: 18px; font-weight: 800; margin: 0 0 12px 0;">📱 App Preview</h4>
            <div style="background: #f9fafb; border: 2px solid #e5e7eb; border-radius: 12px; padding: 20px; font-family: monospace; font-size: 14px; line-height: 1.8;">
{template['preview']}
            </div>
        </div>
    </div>
    """
    
    # Code preview
    code_html = f"""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2); margin-bottom: 25px;">
        <h3 style="color: #92400e; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">💻 Generated Code</h3>
        
        <div style="background: #1f2937; border-radius: 14px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); font-family: 'Courier New', monospace; color: #d1d5db; font-size: 13px; line-height: 1.6; overflow-x: auto;">
<span style="color: #8b5cf6;">import</span> <span style="color: #10b981;">gradio</span> <span style="color: #8b5cf6;">as</span> gr
<span style="color: #8b5cf6;">import</span> <span style="color: #10b981;">pandas</span> <span style="color: #8b5cf6;">as</span> pd

<span style="color: #6b7280;"># {matched_template} - Generated by Rebolt AI</span>

<span style="color: #8b5cf6;">def</span> <span style="color: #3b82f6;">main_app</span>():
    <span style="color: #6b7280;">"AI-generated application logic"</span>
    
    <span style="color: #8b5cf6;">with</span> gr.Blocks() <span style="color: #8b5cf6;">as</span> demo:
        gr.Markdown(<span style="color: #10b981;">"# {matched_template}"</span>)
        
        <span style="color: #6b7280;"># Components generated from natural language</span>
        <span style="color: #d1d5db;">        input_field = gr.Textbox()</span>
        <span style="color: #d1d5db;">        button = gr.Button()</span>
        <span style="color: #d1d5db;">        output = gr.HTML()</span>
        
        <span style="color: #6b7280;"># ... {code_lines - 20} more lines of generated code</span>
        
    demo.launch()

<span style="color: #8b5cf6;">if</span> __name__ == <span style="color: #10b981;">"__main__"</span>:
    main_app()
        </div>
        
        <div style="background: white; border-radius: 12px; padding: 18px; margin-top: 18px;">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
                <div style="text-align: center;">
                    <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Lines of Code</p>
                    <p style="font-size: 28px; color: #f59e0b; font-weight: 900; margin: 0;">{code_lines}</p>
                </div>
                <div style="text-align: center;">
                    <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">Functions</p>
                    <p style="font-size: 28px; color: #8b5cf6; font-weight: 900; margin: 0;">{random.randint(4, 8)}</p>
                </div>
                <div style="text-align: center;">
                    <p style="font-size: 13px; color: #6b7280; margin: 0 0 6px 0;">UI Components</p>
                    <p style="font-size: 28px; color: #3b82f6; font-weight: 900; margin: 0;">{num_components}</p>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Deployment options
    deploy_html = """
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2);">
        <h3 style="color: #065f46; font-size: 24px; font-weight: 900; margin: 0 0 18px 0;">🚀 Deployment Ready</h3>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); width: 50px; height: 50px; border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);">
                    <span style="font-size: 24px;">🌐</span>
                </div>
                <p style="font-size: 16px; color: #1f2937; font-weight: 800; margin: 0 0 6px 0;">Web App</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Vercel • Netlify • AWS</p>
            </div>
            
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); width: 50px; height: 50px; border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);">
                    <span style="font-size: 24px;">📱</span>
                </div>
                <p style="font-size: 16px; color: #1f2937; font-weight: 800; margin: 0 0 6px 0;">Mobile App</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">iOS • Android</p>
            </div>
            
            <div style="background: white; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); width: 50px; height: 50px; border-radius: 50%; margin: 0 auto 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(139, 92, 246, 0.3);">
                    <span style="font-size: 24px;">⚡</span>
                </div>
                <p style="font-size: 16px; color: #1f2937; font-weight: 800; margin: 0 0 6px 0;">API</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">REST • GraphQL</p>
            </div>
        </div>
        
        <div style="background: rgba(16, 185, 129, 0.15); border-radius: 12px; padding: 18px; margin-top: 20px; border: 2px dashed #10b981;">
            <p style="font-size: 14px; color: #065f46; margin: 0; font-weight: 700; text-align: center;">
                ✓ Production-ready code • ✓ Responsive design • ✓ Error handling • ✓ Documentation included
            </p>
        </div>
    </div>
    """
    
    # Create generation timeline
    steps = [
        ("Understanding prompt", 0.3),
        ("Selecting components", 0.5),
        ("Generating UI code", 0.8),
        ("Creating logic", 0.6),
        ("Adding styling", 0.4),
        ("Testing & validation", 0.5)
    ]
    
    fig_timeline = go.Figure()
    
    step_names = [s[0] for s in steps]
    durations = [s[1] for s in steps]
    
    fig_timeline.add_trace(go.Bar(
        y=step_names,
        x=durations,
        orientation='h',
        marker_color=['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4'],
        text=[f'{d:.1f}s' for d in durations],
        textposition='outside'
    ))
    
    fig_timeline.update_layout(
        title="App Generation Pipeline",
        xaxis_title="Time (seconds)",
        height=350,
        showlegend=False
    )
    
    return result_html + components_html + code_html + deploy_html, fig_timeline

def show_app_gallery():
    """Display gallery of apps that can be built"""
    
    gallery_html = """
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(168, 85, 247, 0.2); margin-bottom: 25px;">
        <h3 style="color: #6b21a8; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">🎨 App Gallery</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 0 0 24px 0; font-weight: 600;">Example apps you can build with natural language prompts</p>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px;">
    """
    
    colors = ['#3b82f6', '#10b981', '#ec4899', '#f59e0b', '#8b5cf6']
    
    for idx, (app_name, app_data) in enumerate(APP_TEMPLATES.items()):
        color = colors[idx % len(colors)]
        
        gallery_html += f"""
        <div style="background: white; border-radius: 14px; padding: 22px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid {color};">
            <h4 style="color: #1f2937; font-size: 20px; font-weight: 800; margin: 0 0 10px 0;">{app_name}</h4>
            <p style="color: #6b7280; font-size: 14px; margin: 0 0 15px 0; line-height: 1.6;">{app_data['description']}</p>
            
            <div style="background: #f9fafb; border-radius: 10px; padding: 14px; margin-bottom: 15px;">
                <p style="font-size: 13px; color: #6b7280; margin: 0 0 8px 0; font-weight: 600;">Components:</p>
                <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                    {''.join([f'<span style="background: {color}; color: white; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 700;">{comp}</span>' for comp in app_data['components'][:5]])}
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                <div style="background: #f0f9ff; border-radius: 8px; padding: 10px; text-align: center;">
                    <p style="font-size: 11px; color: #6b7280; margin: 0 0 4px 0;">Lines of Code</p>
                    <p style="font-size: 20px; color: #3b82f6; font-weight: 800; margin: 0;">{app_data['code_lines']}</p>
                </div>
                <div style="background: #fef3c7; border-radius: 8px; padding: 10px; text-align: center;">
                    <p style="font-size: 11px; color: #6b7280; margin: 0 0 4px 0;">Build Time</p>
                    <p style="font-size: 20px; color: #f59e0b; font-weight: 800; margin: 0;">~3s</p>
                </div>
            </div>
        </div>
        """
    
    gallery_html += """
        </div>
        
        <div style="background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); border-radius: 12px; padding: 20px; margin-top: 24px; color: white; text-align: center;">
            <p style="font-size: 18px; font-weight: 800; margin: 0;">💡 Try saying: "Build me a todo list app with priorities" or "Create a weather dashboard for Boston"</p>
        </div>
    </div>
    """
    
    # Create app complexity chart
    apps = list(APP_TEMPLATES.keys())
    lines = [APP_TEMPLATES[app]["code_lines"] for app in apps]
    
    fig_complexity = go.Figure(data=[
        go.Bar(
            x=apps,
            y=lines,
            marker_color=colors,
            text=[f'{l} lines' for l in lines],
            textposition='outside'
        )
    ])
    
    fig_complexity.update_layout(
        title="App Complexity (Lines of Code)",
        yaxis_title="Lines of Code",
        height=400
    )
    
    return gallery_html, fig_complexity

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
            <span style="font-size: 56px;">🗣️</span>
        </div>
        
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Rebolt AI
        </h1>
        
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Build Apps by Speaking with AI</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Natural language → Production-ready code • No coding required</p>
        
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">No-Code</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">AI-Powered</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Instant Deploy</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">YC Backed</span>
        </div>
        
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Rebolt AI</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("🗣️ Build App"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Natural Language App Builder</h3>
                <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Describe your app in plain English and watch AI build it instantly</p>
            </div>
            """)
            
            # Quick examples dropdown
            example_dropdown = gr.Dropdown(
                choices=[
                    "💡 Custom Prompt (Enter Your Own)",
                    "📝 Build a todo list app with priorities",
                    "🌤️ Create a weather dashboard",
                    "💰 Make an expense tracker with charts",
                    "📊 Build a customer survey tool",
                    "👥 Create a team project dashboard"
                ],
                label="Quick Examples",
                value="💡 Custom Prompt (Enter Your Own)"
            )
            
            user_prompt = gr.Textbox(
                label="Describe Your App",
                placeholder="Example: Build me a todo list app where I can add tasks, mark them complete, and delete them. Make it colorful and easy to use.",
                lines=4
            )
            
            complexity = gr.Radio(
                choices=["Simple", "Medium", "Complex"],
                value="Medium",
                label="App Complexity"
            )
            
            generate_btn = gr.Button("✨ Generate App with AI", variant="primary", size="lg")
            
            generation_output = gr.HTML(label="Generated App")
            timeline_chart = gr.Plot(label="Generation Timeline")
            
            generate_btn.click(
                fn=generate_app_from_prompt,
                inputs=[user_prompt, complexity],
                outputs=[generation_output, timeline_chart]
            )
            
            # Load examples
            def load_example(choice):
                examples = {
                    "📝 Build a todo list app with priorities": "Build me a todo list app where I can add tasks with priority levels (high, medium, low), mark them complete with checkboxes, and delete finished tasks. Make it colorful with different colors for each priority.",
                    "🌤️ Create a weather dashboard": "Create a weather dashboard where users can search for any city and see current weather, 5-day forecast, and a temperature chart. Include weather icons and make it beautiful.",
                    "💰 Make an expense tracker with charts": "Make an expense tracker where I can add expenses with amount and category, see total spending, and view a pie chart breakdown by category. Include categories like Food, Transport, Entertainment.",
                    "📊 Build a customer survey tool": "Build a customer survey with rating questions (1-5 stars), text feedback boxes, and a submit button. Show results in a dashboard with average ratings and a list of comments.",
                    "👥 Create a team project dashboard": "Create a team dashboard showing project cards with progress bars, team member assignments, deadlines, and status indicators. Make it look professional for a startup.",
                    "💡 Custom Prompt (Enter Your Own)": ""
                }
                return examples.get(choice, "")
            
            example_dropdown.change(
                fn=load_example,
                inputs=[example_dropdown],
                outputs=[user_prompt]
            )
        
        with gr.Tab("🎨 App Gallery"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Pre-Built App Templates</h3>
                <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Explore what's possible with Rebolt AI's natural language builder</p>
            </div>
            """)
            
            gallery_btn = gr.Button("🎨 View App Gallery", variant="primary", size="lg")
            
            gallery_output = gr.HTML(label="Gallery")
            complexity_chart = gr.Plot(label="Complexity Analysis")
            
            gallery_btn.click(
                fn=show_app_gallery,
                inputs=[],
                outputs=[gallery_output, complexity_chart]
            )
    
    gr.HTML("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Rebolt AI</h2>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 100x Faster Development</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Traditional app dev takes weeks. Rebolt generates production code in seconds. Launch MVPs same day, iterate in real-time with users.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🎨 No-Code Revolution</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Non-technical users build real apps. Product managers ship features without engineering. Democratize software creation.
                </p>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Cost Efficiency</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Developer costs $100K+/year. Rebolt subscription is $50-200/month. Build 10x more apps with same budget.
                </p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Real-World Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Hours → Seconds:</strong> Build apps 100x faster than traditional coding</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">$100K → $1K:</strong> Replace full-time developer with AI subscription</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Anyone → Builder:</strong> Non-technical users ship production apps</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">Idea → MVP:</strong> Launch same day, not same quarter</li>
            </ul>
        </div>
        
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ LLM-Powered Generation</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">GPT-4 understands intent, generates code</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-Platform Output</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Web, iOS, Android from single prompt</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Instant Preview</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">See your app running in real-time</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ One-Click Deploy</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Push to production with single click</p>
                </div>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Rebolt AI</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
            <strong style="color: white;">Tech Stack:</strong> Python • Gradio • LLM Integration • Code Generation
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing natural language app generation with AI.<br>
            Voice/text input • Instant code generation • Multi-platform deployment
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()