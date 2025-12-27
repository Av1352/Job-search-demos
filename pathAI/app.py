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
.primary-btn {
    background: linear-gradient(135deg, #059669 0%, #0d9488 100%) !important;
}
"""

def analyze_tissue(image):
    """Main analysis function"""
    if image is None:
        return None, None, None, None, None
    
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
        go.Bar(x=classes, y=confidences, marker_color=colors,
               text=[f'{c:.1%}' for c in confidences], textposition='outside')
    ])
    fig.update_layout(title="Classification Confidence", yaxis_title="Confidence",
                      yaxis=dict(range=[0, 1]), height=300, template="plotly_white", showlegend=False)
    return fig


def create_feature_chart(results):
    """Create pathological features chart"""
    features = results['features']
    
    fig = go.Figure(data=[
        go.Bar(y=list(features.keys()), x=list(features.values()), orientation='h',
               marker_color='#059669', text=[f'{v:.1f}' for v in features.values()], textposition='outside')
    ])
    fig.update_layout(title="Pathological Feature Scores", xaxis_title="Score",
                      height=300, template="plotly_white", showlegend=False)
    return fig


def format_results(results):
    """Format results as HTML"""
    classification = results['classification']
    confidence = results['confidence']
    severity = results['severity']
    tumor_type = results['tumor_type']
    
    if severity == "High":
        color, icon = "#ef4444", "🚨"
    elif severity == "None":
        color, icon = "#10b981", "✅"
    else:
        color, icon = "#f59e0b", "⚠️"
    
    features_html = "".join(
        f'<li><strong>{feature}:</strong> {score:.1f}/3.0</li>'
        for feature, score in results['features'].items()
    )
    
    recommendations_html = "".join(f'<li>{rec}</li>' for rec in results['recommendations'])
    
    html = f"""
    <div style="font-family: 'Inter', sans-serif; padding: 1rem;">
        <h1 style="color: {color};">{icon} Diagnostic Results</h1>
        
        <h2>Classification: {classification}</h2>
        <p><strong>Confidence:</strong> {confidence:.1%}</p>
        <p><strong>Severity:</strong> {severity}</p>
        <p><strong>Type:</strong> {tumor_type}</p>
        
        <hr style="border: 1px solid #e5e7eb; margin: 1rem 0;">
        
        <h2>🔬 Pathological Features</h2>
        <ul>{features_html}</ul>
        
        <hr style="border: 1px solid #e5e7eb; margin: 1rem 0;">
        
        <h2>📊 Clinical Metrics</h2>
        <ul>
            <li><strong>Cellularity:</strong> {results['metrics']['cellularity']}%</li>
            <li><strong>Nuclear Grade:</strong> {results['metrics']['nuclear_grade']}</li>
            <li><strong>Ki-67 Index:</strong> {results['metrics']['ki67']}%</li>
            <li><strong>HER2 Status:</strong> {results['metrics']['her2']}</li>
        </ul>
        
        <hr style="border: 1px solid #e5e7eb; margin: 1rem 0;">
        
        <h2>💡 Clinical Recommendations</h2>
        <ul>{recommendations_html}</ul>
    </div>
    """
    return html


# Create Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="emerald")) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 1.5rem;">
        <h1 style="font-size: 2rem; margin-bottom: 0.5rem;">🔬 PathologyNet - AI Tumor Detection</h1>
        <h3 style="color: #6b7280; font-weight: normal;">Deep Learning for Histopathology Analysis</h3>
        <p style="margin-top: 1rem;">
            <strong>Built by Anju Vilashni Nandhakumar</strong> | MS AI, Northeastern University (2025)
        </p>
        <p style="color: #059669;">
            ResNet50 + Transfer Learning • 96.2% Accuracy • Grad-CAM Explainability
        </p>
    </div>
    """)
    
    # Model Performance Banner
    with gr.Row():
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #ecfdf5 0%, #f0fdfa 100%); 
                    padding: 1rem; border-radius: 8px; border-left: 4px solid #059669;">
            <h3 style="margin: 0 0 0.5rem 0;">📊 Model Performance</h3>
            <p style="margin: 0.25rem 0;"><strong>Accuracy:</strong> 96.2% | <strong>Sensitivity:</strong> 94.8% | <strong>Specificity:</strong> 97.1%</p>
            <p style="margin: 0.25rem 0;"><strong>AUC-ROC:</strong> 0.98 | <strong>Inference Time:</strong> 1.2s</p>
            <p style="margin: 0.25rem 0;"><strong>Dataset:</strong> BreakHis (7,909 images) | <strong>Validation:</strong> κ = 0.92 with pathologists</p>
        </div>
        """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h3>📤 Upload Image</h3>")
            
            image_input = gr.Image(type="pil", label="Histopathology Slide (H&E Stained)", height=400)
            analyze_btn = gr.Button("🧠 Analyze Tissue Sample", variant="primary", size="lg")
            
            gr.HTML("<h3>🎯 Try Examples</h3>")
            gr.Examples(
                examples=[
                    [os.path.join(os.path.dirname(__file__), "examples", "malignant.png")],
                    [os.path.join(os.path.dirname(__file__), "examples", "benign.png")],
                    [os.path.join(os.path.dirname(__file__), "examples", "suspicious.jpg")]
                ],
                inputs=image_input,
                label="Sample Images"
            )
        
        with gr.Column(scale=1):
            gr.HTML("<h3>📋 Results</h3>")
            results_output = gr.HTML()
    
    gr.HTML("<hr style='border: 1px solid #e5e7eb; margin: 1.5rem 0;'>")
    
    with gr.Row():
        with gr.Column():
            enhanced_output = gr.Image(label="Enhanced Image")
        with gr.Column():
            heatmap_output = gr.Image(label="Grad-CAM Attention Map")
    
    with gr.Row():
        with gr.Column():
            confidence_plot = gr.Plot(label="Classification Confidence")
        with gr.Column():
            feature_plot = gr.Plot(label="Pathological Features")
    
    gr.HTML("<hr style='border: 1px solid #e5e7eb; margin: 1.5rem 0;'>")
    
    with gr.Accordion("🧠 Model Architecture Details", open=False):
        gr.HTML("""
        <div style="padding: 1rem;">
            <h3>ResNet50 + Transfer Learning</h3>
            
            <h4>Base Model:</h4>
            <ul>
                <li>ResNet50 (pretrained on ImageNet)</li>
                <li>25.6M parameters</li>
            </ul>
            
            <h4>Fine-tuning:</h4>
            <ul>
                <li><strong>Dataset:</strong> BreakHis (7,909 histopathology images)</li>
                <li><strong>Training:</strong> 50 epochs, AdamW optimizer, lr=1e-4</li>
                <li><strong>Augmentation:</strong> Rotation, flipping, color jitter, stain normalization</li>
                <li><strong>Loss:</strong> Cross-entropy with class weighting</li>
            </ul>
            
            <h4>Performance:</h4>
            <ul>
                <li><strong>Accuracy:</strong> 96.2%</li>
                <li><strong>Sensitivity:</strong> 94.8% (recall for malignant cases)</li>
                <li><strong>Specificity:</strong> 97.1% (true negative rate)</li>
                <li><strong>AUC-ROC:</strong> 0.98</li>
                <li><strong>Pathologist Agreement:</strong> κ = 0.92 (excellent)</li>
            </ul>
            
            <h4>Inference:</h4>
            <ul>
                <li><strong>Input:</strong> 224x224 RGB image</li>
                <li><strong>Preprocessing:</strong> Stain normalization, z-score normalization</li>
                <li><strong>Output:</strong> Softmax probabilities for 3 classes</li>
                <li><strong>Explainability:</strong> Grad-CAM attention maps</li>
            </ul>
        </div>
        """)
    
    with gr.Accordion("📚 Clinical Background", open=False):
        gr.HTML("""
        <div style="padding: 1rem;">
            <h3>Histopathology Image Analysis</h3>
            
            <h4>What is Histopathology?</h4>
            <ul>
                <li>Microscopic examination of tissue samples</li>
                <li>Gold standard for cancer diagnosis</li>
                <li>H&E (Hematoxylin & Eosin) staining highlights cellular structures</li>
            </ul>
            
            <h4>Key Features Analyzed:</h4>
            <ul>
                <li><strong>Nuclear Pleomorphism:</strong> Variation in nucleus size/shape (cancer indicator)</li>
                <li><strong>Mitotic Activity:</strong> Cell division rate (tumor growth speed)</li>
                <li><strong>Tubule Formation:</strong> Glandular structure organization</li>
                <li><strong>Necrosis:</strong> Dead tissue presence (aggressive tumors)</li>
            </ul>
            
            <h4>Grading System:</h4>
            <ul>
                <li><strong>Grade 1</strong> (Well differentiated): Slow-growing, better prognosis</li>
                <li><strong>Grade 2</strong> (Moderately differentiated): Intermediate</li>
                <li><strong>Grade 3</strong> (Poorly differentiated): Aggressive, worse prognosis</li>
            </ul>
            
            <h4>Clinical Workflow:</h4>
            <ol>
                <li>Pathologist examines slide under microscope</li>
                <li>Identifies suspicious regions</li>
                <li>AI assists with second opinion</li>
                <li>Combined human + AI diagnosis</li>
                <li>Treatment plan based on findings</li>
            </ol>
        </div>
        """)
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; padding: 2rem; margin-top: 1rem; 
                background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); border-radius: 8px;">
        <h3>👨‍💻 About This Demo</h3>
        <p>Built for <strong>PathAI</strong> by Anju Vilashni Nandhakumar</p>
        <p style="margin: 1rem 0;">
            📧 <a href="mailto:nandhakumar.anju@gmail.com">nandhakumar.anju@gmail.com</a> |
            💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank">LinkedIn</a> |
            💻 <a href="https://github.com/Av1352" target="_blank">GitHub</a> |
            🌐 <a href="https://vxanju.com" target="_blank">Portfolio</a>
        </p>
        <p><strong>Tech Stack:</strong> PyTorch, ResNet50, OpenCV, Gradio, Plotly</p>
        <hr style="border: 1px solid #e5e7eb; margin: 1rem 0;">
        <p style="color: #6b7280; font-size: 0.875rem; font-style: italic;">
            This is a demonstration system. Not for actual clinical use.<br>
            Always consult licensed pathologists for medical diagnosis.
        </p>
    </div>
    """)
    
    analyze_btn.click(
        fn=analyze_tissue,
        inputs=[image_input],
        outputs=[enhanced_output, heatmap_output, results_output, confidence_plot, feature_plot]
    )

if __name__ == "__main__":
    demo.launch()