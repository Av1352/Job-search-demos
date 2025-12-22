"""
Adobe AEP Agent Orchestrator Demo
Multi-Agent Marketing Campaign Builder
"""

import gradio as gr
import time

async def orchestrate_campaign(product, audience, goal, context, progress=gr.Progress()):
    """
    Orchestrate multiple AI agents to build a marketing campaign
    """
    if not product or not audience:
        return None, None, None, None, "⚠️ Please fill in Product and Target Audience"
    
    # Build campaign brief
    brief = f"Product: {product}\nAudience: {audience}\nGoal: {goal}"
    if context:
        brief += f"\nContext: {context}"
    
    orchestration_log = ""
    
    def add_log(msg):
        nonlocal orchestration_log
        orchestration_log += f"→ {msg}\n"
        return orchestration_log
    
    # Initialize orchestration
    progress(0.0, desc="Initializing Agent Orchestrator...")
    add_log("🎯 Agent Orchestrator initialized")
    add_log("📋 Analyzing campaign requirements...")
    time.sleep(0.5)
    
    # AGENT 1: Audience Targeting Agent
    progress(0.2, desc="Activating Audience Agent...")
    add_log("🔵 Activating Audience Agent...")
    time.sleep(0.3)
    
    audience_prompt = f"""You are an Audience Targeting Agent for Adobe Experience Platform.

Analyze this campaign brief and provide:
1. Target audience segments (3-4 specific segments)
2. Key demographics and psychographics
3. Recommended channels to reach them

Campaign Brief:
{brief}

Respond in a concise, structured format."""

    try:
        # Using Anthropic API (built into Claude artifacts)
        audience_response = await fetch('https://api.anthropic.com/v1/messages', {
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            },
            'body': {
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 1000,
                'messages': [{'role': 'user', 'content': audience_prompt}]
            }
        })
        
        audience_result = audience_response['content'][0]['text']
        add_log("✅ Audience segments identified")
        progress(0.4, desc="Audience analysis complete")
        time.sleep(0.5)
        
        # AGENT 2: Content Creator Agent
        progress(0.5, desc="Activating Content Agent...")
        add_log("🟣 Activating Content Agent...")
        time.sleep(0.3)
        
        content_prompt = f"""You are a Content Creator Agent for Adobe Experience Platform.

Based on this campaign brief and audience analysis, create:
1. Compelling headline (10 words max)
2. Email subject line (8 words max)
3. Social media post (280 chars max)
4. Call-to-action text

Campaign Brief:
{brief}

Audience Analysis:
{audience_result}

Make it creative, engaging, and conversion-focused."""

        content_response = await fetch('https://api.anthropic.com/v1/messages', {
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            },
            'body': {
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 800,
                'messages': [{'role': 'user', 'content': content_prompt}]
            }
        })
        
        content_result = content_response['content'][0]['text']
        add_log("✅ Marketing content generated")
        progress(0.7, desc="Content creation complete")
        time.sleep(0.5)
        
        # AGENT 3: Campaign Optimizer Agent
        progress(0.8, desc="Activating Optimizer Agent...")
        add_log("🟢 Activating Optimizer Agent...")
        time.sleep(0.3)
        
        optimizer_prompt = f"""You are a Campaign Optimization Agent for Adobe Experience Platform.

Analyze this campaign and provide:
1. A/B test recommendations (2-3 variations to test)
2. Success metrics to track
3. Predicted performance estimate
4. Optimization suggestions

Campaign Brief:
{brief}

Audience Strategy:
{audience_result}

Content Created:
{content_result}

Provide actionable recommendations."""

        optimizer_response = await fetch('https://api.anthropic.com/v1/messages', {
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            },
            'body': {
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 1000,
                'messages': [{'role': 'user', 'content': optimizer_prompt}]
            }
        })
        
        optimizer_result = optimizer_response['content'][0]['text']
        add_log("✅ Campaign optimizations ready")
        progress(0.9, desc="Optimization complete")
        time.sleep(0.5)
        
        # Final Orchestration
        progress(0.95, desc="Synthesizing insights...")
        add_log("🎯 Synthesizing multi-agent insights...")
        
        summary_prompt = f"""You are the Agent Orchestrator for Adobe Experience Platform.

Three specialized agents have analyzed this campaign. Create a concise executive summary (4-5 sentences) that synthesizes their insights:

AUDIENCE AGENT FINDINGS:
{audience_result}

CONTENT AGENT CREATIVES:
{content_result}

OPTIMIZER AGENT RECOMMENDATIONS:
{optimizer_result}

Provide an executive summary highlighting the coordinated campaign strategy."""

        summary_response = await fetch('https://api.anthropic.com/v1/messages', {
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            },
            'body': {
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 500,
                'messages': [{'role': 'user', 'content': summary_prompt}]
            }
        })
        
        summary = summary_response['content'][0]['text']
        add_log("✅ Campaign orchestration complete!")
        progress(1.0, desc="Done!")
        
        return audience_result, content_result, optimizer_result, summary, orchestration_log
        
    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="Adobe AEP Agent Orchestrator Demo") as demo:
    gr.Markdown("""
    # 🚀 Adobe AEP Agent Orchestrator
    
    ### Multi-Agent Marketing Campaign Builder
    
    Experience how Adobe's Agent Orchestrator coordinates multiple specialized AI agents to build 
    complete marketing campaigns. Enter your campaign brief and watch three agents collaborate in real-time.
    
    **Inspired by Adobe Experience Platform's enterprise multi-agent system.**
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Campaign Brief")
            
            product = gr.Textbox(
                label="Product/Service",
                placeholder="e.g., AI-powered fitness app",
                lines=1
            )
            
            audience = gr.Textbox(
                label="Target Audience",
                placeholder="e.g., Health-conscious millennials, 25-40",
                lines=1
            )
            
            goal = gr.Radio(
                choices=["Brand Awareness", "Conversion", "Engagement", "Customer Retention"],
                value="Conversion",
                label="Campaign Goal"
            )
            
            context = gr.Textbox(
                label="Additional Context (Optional)",
                placeholder="e.g., Launching in Q1, competitor is Peloton, budget is $50k",
                lines=3
            )
            
            orchestrate_btn = gr.Button(
                "🚀 Orchestrate Campaign",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("""
            ---
            ### 🤖 The Three Agents:
            
            **🔵 Audience Agent**  
            Analyzes target segments and channels
            
            **🟣 Content Agent**  
            Creates marketing copy and creatives
            
            **🟢 Optimizer Agent**  
            Suggests tests and optimizations
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 🎯 Agent Outputs")
            
            with gr.Accordion("📊 Orchestration Log", open=True):
                orch_log = gr.Textbox(
                    label="Real-time Agent Activity",
                    lines=8,
                    interactive=False
                )
            
            with gr.Tabs():
                with gr.Tab("🔵 Audience Agent"):
                    audience_output = gr.Textbox(
                        label="Target Segments & Channels",
                        lines=8,
                        interactive=False
                    )
                
                with gr.Tab("🟣 Content Agent"):
                    content_output = gr.Textbox(
                        label="Marketing Creatives",
                        lines=8,
                        interactive=False
                    )
                
                with gr.Tab("🟢 Optimizer Agent"):
                    optimizer_output = gr.Textbox(
                        label="A/B Tests & Metrics",
                        lines=8,
                        interactive=False
                    )
                
                with gr.Tab("🎯 Executive Summary"):
                    summary_output = gr.Textbox(
                        label="Synthesized Campaign Strategy",
                        lines=8,
                        interactive=False
                    )
    
    gr.Markdown("""
    ---
    
    ## 🏢 How This Relates to Adobe AEP
    
    **Adobe Experience Platform Agent Orchestrator** coordinates specialized agents across:
    - Real-Time Customer Data Platform
    - Adobe Journey Optimizer  
    - Adobe Experience Manager
    - Customer Journey Analytics
    
    This demo showcases the core concept: **multiple specialized agents collaborating** to accomplish 
    complex marketing tasks faster than any single system could.
    
    ### Key Capabilities Demonstrated:
    ✅ Multi-agent coordination  
    ✅ Sequential task execution  
    ✅ Context sharing between agents  
    ✅ Real-time orchestration  
    ✅ Goal-oriented automation  
    
    ---
    
    **Built by:** Anju Vilashni Nandhakumar | nandhakumar.anju@gmail.com  
    **LinkedIn:** [linkedin.com/in/anju-vilashni](https://linkedin.com/in/anju-vilashni/)  
    **GitHub:** [github.com/Av1352](https://github.com/Av1352)
    
    *Demonstrating understanding of Adobe's AEP Agent Orchestrator architecture*
    """)
    
    # Wire up
    orchestrate_btn.click(
        fn=orchestrate_campaign,
        inputs=[product, audience, goal, context],
        outputs=[audience_output, content_output, optimizer_output, summary_output, orch_log]
    )

if __name__ == "__main__":
    demo.launch()