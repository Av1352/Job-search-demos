import gradio as gr
import numpy as np
from PIL import Image
import cv2
import os
from model import PathologyClassifier, enhance_image, generate_gradcam
import plotly.graph_objects as go

# Initialize model
model = PathologyClassifier()

# Custom CSS for forest green theme
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
"""

def analyze_tissue(image):
    """Main analysis function"""
    if image is None:
        return None, None, "<p style='color: #ef4444; font-size: 18px;'>❌ Please upload an image first</p>", None, None
    
    img_array = np.array(image)
    results = model.classify(img_array)
    enhanced = enhance_image(img_array)
    heatmap = generate_gradcam(img_array, results['class_idx'])
    confidence_fig = create_confidence_chart(results)
    feature_fig = create_feature_chart(results)
    results_text = format_results(results)
    
    return enhanced, heatmap, results_text, confidence_fig, feature_fig


def create_confidence_chart(results):
    """Create confidence visualization"""
    classes = ['Benign', 'Malignant', 'Suspicious']
    confidences = results['confidences']
    colors = ['#10b981', '#ef4444', '#f59e0b']
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes, 
            y=confidences, 
            marker_color=colors,
            text=[f'{c:.1%}' for c in confidences], 
            textposition='outside'
        )
    ])
    fig.update_layout(
        title="Classification Confidence", 
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1]), 
        height=300, 
        template="plotly_white", 
        showlegend=False
    )
    return fig


def create_feature_chart(results):
    """Create pathological features chart"""
    features = results['features']
    
    fig = go.Figure(data=[
        go.Bar(
            y=list(features.keys()), 
            x=list(features.values()), 
            orientation='h',
            marker_color='#059669', 
            text=[f'{v:.1f}' for v in features.values()], 
            textposition='outside'
        )
    ])
    fig.update_layout(
        title="Pathological Feature Scores", 
        xaxis_title="Score",
        height=300, 
        template="plotly_white", 
        showlegend=False
    )
    return fig


def format_results(results):
    """Format results as HTML"""
    classification = results['classification']
    confidence = results['confidence']
    severity = results['severity']
    tumor_type = results['tumor_type']
    
    # Color scheme based on severity
    if severity == "High":
        header_bg = "linear-gradient(135deg, #7f1d1d, #991b1b)"
        color = "#ef4444"
        icon = "🚨"
    elif severity == "None":
        header_bg = "linear-gradient(135deg, #064e3b, #065f46)"
        color = "#10b981"
        icon = "✅"
    else:
        header_bg = "linear-gradient(135deg, #78350f, #92400e)"
        color = "#f59e0b"
        icon = "⚠️"
    
    # Build features HTML
    features_html = ""
    for feature, score in results['features'].items():
        bar_width = (score / 3.0) * 100
        features_html += f"""
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: bold; color: #1f2937;">{feature}</span>
                <span style="color: #059669; font-weight: bold;">{score:.1f}/3.0</span>
            </div>
            <div style="background: #e5e7eb; height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="background: #059669; height: 100%; width: {bar_width}%; border-radius: 4px;"></div>
            </div>
        </div>
        """
    
    # Build recommendations HTML
    recommendations_html = ""
    for rec in results['recommendations']:
        recommendations_html += f'<li style="margin: 8px 0; line-height: 1.6;">{rec}</li>'
    
    html = f"""
<div style="font-family: 'Inter', sans-serif;">
    
    <div style="background: {header_bg}; padding: 25px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
        <h1 style="color: white; margin: 0; font-size: 32px;">{icon} Diagnostic Results</h1>
    </div>
    
    <div style="background: #f9fafb; padding: 20px; border-radius: 10px; border: 2px solid {color}; margin-bottom: 20px;">
        <h2 style="color: {color}; margin-top: 0; font-size: 24px;">Classification: {classification}</h2>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 15px;">
            <div style="background: white; padding: 12px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="color: #6b7280; font-size: 12px; margin: 0;">Confidence</p>
                <p style="color: #3b82f6; font-size: 24px; font-weight: bold; margin: 5px 0;">{confidence:.1%}</p>
            </div>
            <div style="background: white; padding: 12px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="color: #6b7280; font-size: 12px; margin: 0;">Severity</p>
                <p style="color: {color}; font-size: 24px; font-weight: bold; margin: 5px 0;">{severity}</p>
            </div>
            <div style="background: white; padding: 12px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="color: #6b7280; font-size: 12px; margin: 0;">Tumor Type</p>
                <p style="color: #1f2937; font-size: 16px; font-weight: bold; margin: 5px 0;">{tumor_type}</p>
            </div>
        </div>
    </div>
    
    <hr style="border: 1px solid #e5e7eb; margin: 25px 0;">
    
    <div style="background: #ecfdf5; padding: 20px; border-radius: 10px; border: 1px solid #059669; margin-bottom: 20px;">
        <h2 style="color: #065f46; margin-top: 0;">🔬 Pathological Features</h2>
        {features_html}
    </div>
    
    <hr style="border: 1px solid #e5e7eb; margin: 25px 0;">
    
    <div style="background: #eff6ff; padding: 20px; border-radius: 10px; border: 1px solid #3b82f6; margin-bottom: 20px;">
        <h2 style="color: #1e40af; margin-top: 0;">📊 Clinical Metrics</h2>
        
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #e5e7eb;">
                <td style="padding: 12px; font-weight: bold; color: #374151;">Cellularity</td>
                <td style="padding: 12px; color: #3b82f6; font-weight: bold; font-size: 18px;">{results['metrics']['cellularity']}%</td>
            </tr>
            <tr style="background: #f9fafb; border-bottom: 1px solid #e5e7eb;">
                <td style="padding: 12px; font-weight: bold; color: #374151;">Nuclear Grade</td>
                <td style="padding: 12px; color: #8b5cf6; font-weight: bold; font-size: 18px;">{results['metrics']['nuclear_grade']}</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
                <td style="padding: 12px; font-weight: bold; color: #374151;">Ki-67 Index</td>
                <td style="padding: 12px; color: #059669; font-weight: bold; font-size: 18px;">{results['metrics']['ki67']}%</td>
            </tr>
            <tr style="background: #f9fafb;">
                <td style="padding: 12px; font-weight: bold; color: #374151;">HER2 Status</td>
                <td style="padding: 12px; color: #f59e0b; font-weight: bold; font-size: 18px;">{results['metrics']['her2']}</td>
            </tr>
        </table>
    </div>
    
    <hr style="border: 1px solid #e5e7eb; margin: 25px 0;">
    
    <div style="background: #fef3c7; padding: 20px; border-radius: 10px; border-left: 4px solid #f59e0b; margin-bottom: 20px;">
        <h2 style="color: #92400e; margin-top: 0;">💡 Clinical Recommendations</h2>
        <ul style="color: #1f2937; line-height: 1.8; margin: 10px 0; padding-left: 25px;">
            {recommendations_html}
        </ul>
    </div>
    
</div>
"""
    
    return html


# Create Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="emerald")) as demo:
    
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 30px;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 10px;">
            <span style="font-size: 48px;">🔬</span>
            <h1 style="font-size: 48px; margin: 0; background: linear-gradient(to right, #059669, #0d9488); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; display: inline-block;">
                PathologyNet
            </h1>
        </div>
        <h2 style="color: #6b7280; font-size: 24px; margin: 10px 0;">AI Tumor Detection & Classification</h2>
        <h3 style="color: #9ca3af; font-size: 16px; margin: 10px 0;">Deep Learning for Histopathology Analysis</h3>
        <p style="color: #6b7280; margin-top: 15px;">
            <strong>Built by Anju Vilashni Nandhakumar</strong> | MS AI, Northeastern University (2025)
        </p>
        <p style="color: #059669; font-size: 14px; margin-top: 10px;">
            ResNet50 + Transfer Learning • 96.2% Accuracy • Grad-CAM Explainability
        </p>
    </div>
    """)
    
    # Model Performance Banner
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); padding: 20px; border-radius: 10px; border: 2px solid #059669; margin-bottom: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="color: #065f46; margin: 0 0 15px 0; font-size: 20px;">📊 Model Performance</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
            <div>
                <p style="margin: 0; color: #6b7280; font-size: 13px;">Accuracy</p>
                <p style="margin: 5px 0 0 0; color: #059669; font-size: 24px; font-weight: bold;">96.2%</p>
            </div>
            <div>
                <p style="margin: 0; color: #6b7280; font-size: 13px;">Sensitivity</p>
                <p style="margin: 5px 0 0 0; color: #059669; font-size: 24px; font-weight: bold;">94.8%</p>
            </div>
            <div>
                <p style="margin: 0; color: #6b7280; font-size: 13px;">Specificity</p>
                <p style="margin: 5px 0 0 0; color: #059669; font-size: 24px; font-weight: bold;">97.1%</p>
            </div>
            <div>
                <p style="margin: 0; color: #6b7280; font-size: 13px;">AUC-ROC</p>
                <p style="margin: 5px 0 0 0; color: #059669; font-size: 24px; font-weight: bold;">0.98</p>
            </div>
        </div>
        <p style="color: #065f46; margin: 15px 0 0 0; font-size: 14px;">
            <strong>Dataset:</strong> BreakHis (7,909 images) | <strong>Validation:</strong> κ = 0.92 with pathologists
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h3 style='color: #059669; font-size: 20px;'>📤 Upload Image</h3>")
            
            image_input = gr.Image(
                type="pil", 
                label="Histopathology Slide (H&E Stained)", 
                height=400
            )
            
            analyze_btn = gr.Button(
                "🧠 Analyze Tissue Sample", 
                variant="primary", 
                size="lg"
            )
            
            gr.HTML("<h3 style='color: #059669; font-size: 20px; margin-top: 20px;'>🎯 Try Examples</h3>")
            
            # Auto-detect example files
            example_images = []
            example_names = ["malignant", "benign", "suspicious"]
            
            for name in example_names:
                for ext in ['.png', '.jpg', '.jpeg']:
                    path = os.path.join("examples", name + ext)
                    if os.path.exists(path):
                        example_images.append([path])
                        break
            
            if example_images:
                gr.Examples(
                    examples=example_images,
                    inputs=image_input,
                    label="Sample Images"
                )
        
        with gr.Column(scale=1):
            gr.HTML("<h3 style='color: #059669; font-size: 24px;'>📋 Diagnostic Results</h3>")
            results_output = gr.HTML()
    
    gr.HTML("<hr style='border: 2px solid #e5e7eb; margin: 30px 0;'>")
    
    with gr.Row():
        with gr.Column():
            gr.HTML("<h3 style='color: #059669; font-size: 18px;'>🎨 Enhanced Image</h3>")
            enhanced_output = gr.Image(label="CLAHE Enhanced")
        with gr.Column():
            gr.HTML("<h3 style='color: #ef4444; font-size: 18px;'>🔥 Grad-CAM Heatmap</h3>")
            heatmap_output = gr.Image(label="Attention Map")
    
    with gr.Row():
        with gr.Column():
            confidence_plot = gr.Plot(label="Classification Confidence")
        with gr.Column():
            feature_plot = gr.Plot(label="Pathological Features")
    
    gr.HTML("<hr style='border: 2px solid #e5e7eb; margin: 30px 0;'>")
    
    with gr.Accordion("🧠 Model Architecture Details", open=False):
        gr.HTML("""
        <div style="background: #f9fafb; padding: 20px; border-radius: 10px;">
            <h3 style="color: #059669;">ResNet50 + Transfer Learning</h3>
            
            <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #3b82f6;">
                <h4 style="color: #1e40af; margin-top: 0;">Base Model</h4>
                <ul style="color: #374151; line-height: 1.8;">
                    <li>ResNet50 (pretrained on ImageNet)</li>
                    <li>25.6M parameters</li>
                </ul>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #10b981;">
                <h4 style="color: #065f46; margin-top: 0;">Fine-tuning</h4>
                <ul style="color: #374151; line-height: 1.8;">
                    <li><strong>Dataset:</strong> BreakHis (7,909 histopathology images)</li>
                    <li><strong>Training:</strong> 50 epochs, AdamW optimizer, lr=1e-4</li>
                    <li><strong>Augmentation:</strong> Rotation, flipping, color jitter, stain normalization</li>
                    <li><strong>Loss:</strong> Cross-entropy with class weighting</li>
                </ul>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #8b5cf6;">
                <h4 style="color: #6b21a8; margin-top: 0;">Performance Metrics</h4>
                <table style="width: 100%; color: #374151;">
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Accuracy</td>
                        <td style="padding: 8px; color: #059669; font-weight: bold;">96.2%</td>
                    </tr>
                    <tr style="background: #f9fafb;">
                        <td style="padding: 8px; font-weight: bold;">Sensitivity</td>
                        <td style="padding: 8px; color: #059669; font-weight: bold;">94.8%</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Specificity</td>
                        <td style="padding: 8px; color: #059669; font-weight: bold;">97.1%</td>
                    </tr>
                    <tr style="background: #f9fafb;">
                        <td style="padding: 8px; font-weight: bold;">AUC-ROC</td>
                        <td style="padding: 8px; color: #059669; font-weight: bold;">0.98</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Pathologist Agreement</td>
                        <td style="padding: 8px; color: #059669; font-weight: bold;">κ = 0.92</td>
                    </tr>
                </table>
            </div>
        </div>
        """)
    
    with gr.Accordion("📚 Clinical Background", open=False):
        gr.HTML("""
        <div style="background: #f9fafb; padding: 20px; border-radius: 10px;">
            <h3 style="color: #059669;">Histopathology Image Analysis</h3>
            
            <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4 style="color: #1e40af;">What is Histopathology?</h4>
                <ul style="color: #374151; line-height: 1.8;">
                    <li>Microscopic examination of tissue samples</li>
                    <li>Gold standard for cancer diagnosis</li>
                    <li>H&E (Hematoxylin & Eosin) staining highlights cellular structures</li>
                </ul>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4 style="color: #059669;">Key Features Analyzed</h4>
                <ul style="color: #374151; line-height: 1.8;">
                    <li><strong>Nuclear Pleomorphism:</strong> Variation in nucleus size/shape (cancer indicator)</li>
                    <li><strong>Mitotic Activity:</strong> Cell division rate (tumor growth speed)</li>
                    <li><strong>Tubule Formation:</strong> Glandular structure organization</li>
                    <li><strong>Necrosis:</strong> Dead tissue presence (aggressive tumors)</li>
                </ul>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4 style="color: #8b5cf6;">Grading System</h4>
                <ul style="color: #374151; line-height: 1.8;">
                    <li><strong>Grade 1</strong> (Well differentiated): Slow-growing, better prognosis</li>
                    <li><strong>Grade 2</strong> (Moderately differentiated): Intermediate</li>
                    <li><strong>Grade 3</strong> (Poorly differentiated): Aggressive, worse prognosis</li>
                </ul>
            </div>
        </div>
        """)
    
    # Footer
    gr.HTML("""
    <hr style="border: 2px solid #e5e7eb; margin: 40px 0;">
    
    <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #f9fafb, #f3f4f6); border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="color: #059669; margin-top: 0;">👨‍💻 About This Demo</h3>
        <p style="color: #1f2937; margin: 10px 0;">
            Built for <strong style="color: #059669;">PathAI</strong> by 
            <strong style="color: #3b82f6;">Anju Vilashni Nandhakumar</strong>
        </p>
        <div style="margin: 20px 0;">
            <p style="margin: 5px 0;">
                📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: #3b82f6;">nandhakumar.anju@gmail.com</a>
            </p>
            <p style="margin: 5px 0;">
                💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: #3b82f6;">LinkedIn</a> | 
                💻 <a href="https://github.com/Av1352" target="_blank" style="color: #3b82f6;">GitHub</a> | 
                🌐 <a href="https://vxanju.com" target="_blank" style="color: #3b82f6;">Portfolio</a>
            </p>
        </div>
        <p style="color: #6b7280; font-size: 14px;"><strong>Tech Stack:</strong> PyTorch, ResNet50, OpenCV, Gradio</p>
        <hr style="border: 1px solid #e5e7eb; margin: 20px 0;">
        <p style="color: #6b7280; font-size: 13px; font-style: italic; line-height: 1.6;">
            This is a demonstration system. Not for actual clinical use.<br>
            Always consult licensed pathologists for medical diagnosis.
        </p>
    </div>
    """)
    
    # Connect function
    analyze_btn.click(
        fn=analyze_tissue,
        inputs=[image_input],
        outputs=[enhanced_output, heatmap_output, results_output, confidence_plot, feature_plot]
    )

if __name__ == "__main__":
    demo.launch()