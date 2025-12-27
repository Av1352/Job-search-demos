import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from model import HealthcarePricePredictor

# Initialize the ML model
predictor = HealthcarePricePredictor()
predictor.train()

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
"""

def predict_price(procedure, location, insurance):
    """Main prediction function"""
    
    # Get predictions for all facility types
    results = predictor.predict_all_facilities(procedure, location, insurance)
    
    # Create facility comparison chart
    facility_fig = go.Figure(data=[
        go.Bar(
            x=[f['name'] for f in results['facilities']],
            y=[f['patient_pays'] for f in results['facilities']],
            marker_color=['#10b981' if i == 0 else '#6b7280' for i in range(len(results['facilities']))],
            text=[f"${f['patient_pays']:.0f}" for f in results['facilities']],
            textposition='outside'
        )
    ])
    facility_fig.update_layout(
        title="Price Comparison Across Facilities",
        xaxis_title="Facility",
        yaxis_title="Your Cost ($)",
        height=400,
        template="plotly_white"
    )
    
    # Create feature importance chart
    importance_df = pd.DataFrame(results['feature_importance'])
    importance_fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title="Feature Importance in Price Prediction",
        labels={'importance': 'Importance', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )
    importance_fig.update_layout(height=300, template="plotly_white")
    
    # Create training loss curve
    loss_df = pd.DataFrame(results['training_losses'])
    loss_fig = px.line(
        loss_df,
        x='iteration',
        y='loss',
        title="Model Training Progress (Loss Curve)",
        labels={'iteration': 'Iteration', 'loss': 'MSE Loss'}
    )
    loss_fig.update_layout(height=300, template="plotly_white")
    
    # Create prediction explanation chart
    contrib_data = sorted(
        results['best_facility']['contributions'],
        key=lambda x: abs(x['contribution']),
        reverse=True
    )
    contrib_fig = go.Figure(data=[
        go.Bar(
            x=[c['contribution'] for c in contrib_data],
            y=[c['feature'] for c in contrib_data],
            orientation='h',
            marker_color=['#ef4444' if c['contribution'] > 0 else '#10b981' for c in contrib_data],
            text=[f"${c['contribution']:.0f}" for c in contrib_data],
            textposition='outside'
        )
    ])
    contrib_fig.update_layout(
        title="How Features Affect Your Price",
        xaxis_title="Price Impact ($)",
        yaxis_title="Feature",
        height=300,
        template="plotly_white"
    )
    
    # Format output as HTML
    best_facility = results['best_facility']
    metrics = results['model_metrics']
    
    summary = f"""
<div style="background: linear-gradient(135deg, #3b82f6, #2563eb); padding: 25px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h1 style="color: white; margin: 0 0 10px 0; font-size: 32px;">🎯 Your Price Prediction</h1>
</div>

<div style="background: #f0fdf4; border: 2px solid #10b981; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h2 style="color: #065f46; margin-top: 0;">Best Facility: {best_facility['name']}</h2>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
        <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Your Cost</p>
            <p style="color: #10b981; font-size: 28px; font-weight: bold; margin: 5px 0;">${best_facility['patient_pays']:.2f}</p>
        </div>
        <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Total Procedure Cost</p>
            <p style="color: #3b82f6; font-size: 28px; font-weight: bold; margin: 5px 0;">${best_facility['price']:.2f}</p>
        </div>
        <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Quality Rating</p>
            <p style="color: #f59e0b; font-size: 28px; font-weight: bold; margin: 5px 0;">⭐ {best_facility['quality']}/5.0</p>
        </div>
    </div>
    
    <div style="margin-top: 15px;">
        <p style="margin: 5px 0;"><strong>Distance:</strong> {best_facility['distance']}</p>
        <p style="margin: 5px 0;"><strong>Wait Time:</strong> {best_facility['wait_time']}</p>
    </div>
</div>

<div style="background: linear-gradient(135deg, #10b981, #059669); padding: 20px; border-radius: 10px; margin: 20px 0; color: white;">
    <h2 style="margin-top: 0;">💰 Potential Savings</h2>
    <p style="font-size: 36px; font-weight: bold; margin: 10px 0;">${results['savings']:.2f}</p>
    <p style="font-size: 18px; opacity: 0.9;">Save {results['savings_percent']:.1f}% by choosing the best-value facility</p>
</div>

<hr style="border: 1px solid #e5e7eb; margin: 30px 0;">

<div style="background: #eff6ff; border: 2px solid #3b82f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h2 style="color: #1e40af; margin-top: 0;">🤖 ML Model Performance</h2>
    
    <table style="width: 100%; border-collapse: collapse;">
        <tr>
            <td style="padding: 10px; font-weight: bold;">R² Score</td>
            <td style="padding: 10px; color: #10b981; font-weight: bold; font-size: 20px;">{metrics['r2']:.1%}</td>
            <td style="padding: 10px; color: #6b7280;">Prediction accuracy</td>
        </tr>
        <tr style="background: #f9fafb;">
            <td style="padding: 10px; font-weight: bold;">MAE</td>
            <td style="padding: 10px; color: #3b82f6; font-weight: bold; font-size: 20px;">${metrics['mae']:.2f}</td>
            <td style="padding: 10px; color: #6b7280;">Average error</td>
        </tr>
        <tr>
            <td style="padding: 10px; font-weight: bold;">RMSE</td>
            <td style="padding: 10px; color: #8b5cf6; font-weight: bold; font-size: 20px;">${metrics['rmse']:.2f}</td>
            <td style="padding: 10px; color: #6b7280;">Error variance</td>
        </tr>
        <tr style="background: #f9fafb;">
            <td style="padding: 10px; font-weight: bold;">Training Samples</td>
            <td style="padding: 10px; color: #f59e0b; font-weight: bold; font-size: 20px;">{results['training_size']}</td>
            <td style="padding: 10px; color: #6b7280;">Dataset size</td>
        </tr>
    </table>
</div>

<hr style="border: 1px solid #e5e7eb; margin: 30px 0;">

<h2 style="color: #1e40af;">📊 All Facility Options</h2>

<div style="margin-top: 20px;">
"""
    
    for i, fac in enumerate(results['facilities'], 1):
        is_best = i == 1
        bg_color = '#f0fdf4' if is_best else '#f9fafb'
        border_color = '#10b981' if is_best else '#e5e7eb'
        
        summary += f"""
<div style="background: {bg_color}; border: 2px solid {border_color}; padding: 15px; border-radius: 8px; margin: 10px 0;">
    <div style="display: flex; justify-content: space-between; align-items: start;">
        <div>
            <h3 style="margin: 0 0 5px 0; color: #111827;">
                {i}. {fac['name']} 
                {'<span style="background: #10b981; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; margin-left: 10px;">✅ BEST VALUE</span>' if is_best else ''}
            </h3>
            <p style="margin: 5px 0; color: #6b7280; font-size: 14px;">Quality: ⭐ {fac['quality']}/5.0</p>
            <p style="margin: 5px 0; color: #6b7280; font-size: 14px;">Wait Time: {fac['wait_time']}</p>
        </div>
        <div style="text-align: right;">
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Your Cost</p>
            <p style="color: #10b981; font-size: 32px; font-weight: bold; margin: 5px 0;">${fac['patient_pays']:.2f}</p>
        </div>
    </div>
</div>
"""
    
    summary += "</div>"
    
    return summary, facility_fig, importance_fig, loss_fig, contrib_fig


# Create Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 30px;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 10px;">
            <span style="font-size: 48px;">🏥</span>
            <h1 style="font-size: 48px; margin: 0; background: linear-gradient(to right, #3b82f6, #2563eb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; display: inline-block;">
                Serif Health
            </h1>
        </div>
        <h2 style="color: #6b7280; font-size: 24px; margin: 10px 0;">ML-Powered Healthcare Price Predictor</h2>
        <h3 style="color: #9ca3af; font-size: 16px; margin: 10px 0;">Price Transparency Platform</h3>
        <p style="color: #6b7280; margin-top: 15px;">
            <strong>Built by Anju Vilashni Nandhakumar</strong> | MS AI, Northeastern University (2025)
        </p>
        <p style="color: #3b82f6; font-size: 14px; margin-top: 10px;">
            Linear Regression • 500 Training Samples • 85%+ Accuracy • SHAP-style Explanations
        </p>
    </div>
    """)
    
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #eff6ff, #dbeafe); padding: 25px; border-radius: 12px; margin: 20px 0; border: 1px solid #3b82f6;">
        <h2 style="color: #1e40af; margin-top: 0;">🎯 System Overview</h2>
        <p style="color: #1f2937; line-height: 1.8;">
            This demo showcases a complete ML pipeline for predicting healthcare prices:
        </p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="font-weight: bold; color: #3b82f6; margin: 0;">✅ Real Model Training</p>
                <p style="font-size: 12px; color: #6b7280; margin: 5px 0;">500+ samples, gradient descent</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="font-weight: bold; color: #10b981; margin: 0;">✅ Feature Engineering</p>
                <p style="font-size: 12px; color: #6b7280; margin: 5px 0;">4 categorical variables</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="font-weight: bold; color: #8b5cf6; margin: 0;">✅ Model Evaluation</p>
                <p style="font-size: 12px; color: #6b7280; margin: 5px 0;">R², MAE, RMSE metrics</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="font-weight: bold; color: #f59e0b; margin: 0;">✅ Explainability</p>
                <p style="font-size: 12px; color: #6b7280; margin: 5px 0;">SHAP-style contributions</p>
            </div>
        </div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h3 style='color: #3b82f6; font-size: 20px;'>📋 Select Your Parameters</h3>")
            
            procedure = gr.Dropdown(
                choices=[
                    'MRI - Brain (CPT 70553)',
                    'CT Scan - Chest (CPT 71260)',
                    'Knee Arthroscopy (CPT 29881)',
                    'Colonoscopy (CPT 45378)',
                    'Blood Panel (CPT 80053)',
                    'X-Ray - Chest (CPT 71046)'
                ],
                label="Medical Procedure",
                value='MRI - Brain (CPT 70553)'
            )
            
            location = gr.Dropdown(
                choices=[
                    'Boston, MA',
                    'Houston, TX',
                    'San Francisco, CA',
                    'Miami, FL',
                    'Chicago, IL'
                ],
                label="Location",
                value='Boston, MA'
            )
            
            insurance = gr.Dropdown(
                choices=[
                    'PPO Insurance',
                    'HMO Insurance',
                    'High Deductible Plan',
                    'Medicare',
                    'Uninsured (Cash)'
                ],
                label="Insurance Type",
                value='PPO Insurance'
            )
            
            predict_btn = gr.Button("🤖 Run ML Prediction", variant="primary", size="lg")
            
            gr.HTML("""
            <hr style="margin: 20px 0; border: 1px solid #e5e7eb;">
            <div style="background: #f3f4f6; padding: 15px; border-radius: 8px;">
                <h4 style="color: #374151; margin-top: 0; font-size: 16px;">📊 Model Stats</h4>
                <ul style="color: #6b7280; font-size: 14px; line-height: 1.8; margin: 10px 0; padding-left: 20px;">
                    <li><strong>Algorithm:</strong> Linear Regression (Gradient Descent)</li>
                    <li><strong>Features:</strong> 4 categorical variables</li>
                    <li><strong>Training:</strong> 500 samples, 1000 iterations</li>
                    <li><strong>Performance:</strong> 85%+ R² score</li>
                </ul>
            </div>
            """)
        
        with gr.Column(scale=2):
            gr.HTML("<h3 style='color: #10b981; font-size: 24px;'>💡 Prediction Results</h3>")
            output_text = gr.HTML()
    
    gr.HTML("<hr style='border: 2px solid #e5e7eb; margin: 30px 0;'>")
    gr.HTML("<h2 style='color: #1e40af; font-size: 28px; text-align: center;'>📈 Interactive Visualizations</h2>")
    
    with gr.Row():
        with gr.Column():
            facility_plot = gr.Plot(label="Facility Price Comparison")
        with gr.Column():
            importance_plot = gr.Plot(label="Feature Importance")
    
    with gr.Row():
        with gr.Column():
            loss_plot = gr.Plot(label="Training Loss Curve")
        with gr.Column():
            contrib_plot = gr.Plot(label="Prediction Explanation")
    
    # Examples
    gr.HTML("<hr style='border: 2px solid #e5e7eb; margin: 30px 0;'>")
    gr.HTML("<h2 style='color: #1e40af; font-size: 24px;'>🎯 Try These Examples</h2>")
    gr.Examples(
        examples=[
            ['MRI - Brain (CPT 70553)', 'Boston, MA', 'PPO Insurance'],
            ['Knee Arthroscopy (CPT 29881)', 'San Francisco, CA', 'High Deductible Plan'],
            ['Blood Panel (CPT 80053)', 'Houston, TX', 'Medicare'],
            ['Colonoscopy (CPT 45378)', 'Chicago, IL', 'Uninsured (Cash)']
        ],
        inputs=[procedure, location, insurance],
        label="Quick Start Examples"
    )
    
    # About section
    gr.HTML("""
    <hr style="border: 2px solid #e5e7eb; margin: 40px 0;">
    
    <div style="background: linear-gradient(135deg, #f0fdf4, #dcfce7); padding: 25px; border-radius: 12px; border: 1px solid #10b981;">
        <h2 style="color: #065f46; margin-top: 0;">🎓 About This Project</h2>
        
        <p style="color: #1f2937; line-height: 1.8;">
            This is a technical demonstration built for <strong style="color: #3b82f6;">Serif Health's ML Engineer position</strong> 
            by <strong style="color: #10b981;">Anju Vilashni Nandhakumar</strong>.
        </p>
        
        <h3 style="color: #059669; margin-top: 20px;">Key Technical Features:</h3>
        <ol style="color: #1f2937; line-height: 2; padding-left: 25px;">
            <li><strong>ML Model Training</strong> - Gradient descent implementation with 1000 iterations</li>
            <li><strong>Feature Engineering</strong> - Label encoding for categorical variables</li>
            <li><strong>Model Evaluation</strong> - R², MAE, RMSE metrics</li>
            <li><strong>Explainability</strong> - SHAP-style feature contribution analysis</li>
            <li><strong>Data Visualization</strong> - Interactive Plotly charts</li>
        </ol>
        
        <h3 style="color: #059669; margin-top: 20px;">Why This Matters:</h3>
        <p style="color: #1f2937; line-height: 1.8;">
            Healthcare price transparency is critical. This demo shows how ML can empower patients with:
        </p>
        <ul style="color: #1f2937; line-height: 2; padding-left: 25px;">
            <li>Accurate price predictions (85%+ accuracy)</li>
            <li>Facility comparisons (save up to 70%)</li>
            <li>Clear explanations of price factors</li>
            <li>Data-driven recommendations</li>
        </ul>
        
        <h3 style="color: #059669; margin-top: 20px;">Production Enhancements:</h3>
        <p style="color: #1f2937; line-height: 1.8;">
            In production, this would include:
        </p>
        <ul style="color: #1f2937; line-height: 2; padding-left: 25px;">
            <li>XGBoost/LightGBM for better accuracy</li>
            <li>Real-time claims data pipeline</li>
            <li>Confidence intervals with bootstrapping</li>
            <li>A/B testing framework</li>
            <li>Automated model retraining</li>
        </ul>
        
        <hr style="border: 1px solid #10b981; margin: 25px 0;">
        
        <div style="text-align: center;">
            <p style="margin: 10px 0;"><strong>Connect with me:</strong></p>
            <p style="margin: 5px 0;">
                💼 <a href="https://linkedin.com/in/anju-vilashni" style="color: #3b82f6;">LinkedIn</a> | 
                💻 <a href="https://github.com/Av1352" style="color: #3b82f6;">GitHub</a> | 
                🌐 <a href="https://vxanju.com" style="color: #3b82f6;">Portfolio</a> | 
                📧 nandhakumar.anju@gmail.com
            </p>
            <p style="color: #6b7280; font-size: 12px; margin-top: 15px; font-style: italic;">
                Built with: Python, NumPy, Gradio, Plotly | December 2024
            </p>
        </div>
    </div>
    """)
    
    # Connect the prediction function
    predict_btn.click(
        fn=predict_price,
        inputs=[procedure, location, insurance],
        outputs=[output_text, facility_plot, importance_plot, loss_plot, contrib_plot]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()