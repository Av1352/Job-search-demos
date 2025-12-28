"""
ClearML MLOps Demo - Interactive Training Dashboard
----------------------------------------------------
Train ML models with real-time tracking via beautiful Gradio interface
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from clearml import Task
import matplotlib.pyplot as plt

# Model definition
class MNISTNet(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Global data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

def create_plot(train_metrics, test_metrics):
    """Create training curves plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_metrics['loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, test_metrics['loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Test Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_metrics['acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, test_metrics['acc'], 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def train_model(batch_size, learning_rate, hidden_size, dropout, epochs, progress=gr.Progress()):
    """Train model with ClearML tracking"""
    
    # Initialize ClearML Task
    task = Task.init(
        project_name='MNIST Classification',
        task_name=f'Training_bs{int(batch_size)}_lr{learning_rate}_h{int(hidden_size)}',
        tags=['gradio-demo', 'interactive']
    )
    
    # Log hyperparameters
    task.connect({
        'batch_size': int(batch_size),
        'learning_rate': learning_rate,
        'hidden_size': int(hidden_size),
        'dropout': dropout,
        'epochs': int(epochs)
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False)
    
    # Initialize model
    model = MNISTNet(int(hidden_size), dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Metrics storage
    train_metrics = {'loss': [], 'acc': []}
    test_metrics = {'loss': [], 'acc': []}
    
    output_log = ""
    
    # Training loop
    for epoch in range(1, int(epochs) + 1):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                progress((epoch - 1 + batch_idx / len(train_loader)) / int(epochs), 
                        desc=f"Epoch {epoch}/{int(epochs)}")
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_metrics['loss'].append(train_loss)
        train_metrics['acc'].append(train_acc)
        
        # Test
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / len(test_loader.dataset)
        test_metrics['loss'].append(test_loss)
        test_metrics['acc'].append(test_acc)
        
        # Report to ClearML
        from clearml import Logger
        logger = Logger.current_logger()
        logger.report_scalar("train", "loss", iteration=epoch, value=train_loss)
        logger.report_scalar("train", "accuracy", iteration=epoch, value=train_acc)
        logger.report_scalar("test", "loss", iteration=epoch, value=test_loss)
        logger.report_scalar("test", "accuracy", iteration=epoch, value=test_acc)
        
        # Update log
        epoch_log = f"Epoch {epoch}/{int(epochs)} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n"
        output_log += epoch_log
    
    # Save model
    torch.save(model.state_dict(), 'best_model.pth')
    
    # Create plots
    plot_fig = create_plot(train_metrics, test_metrics)
    
    # Get ClearML experiment URL
    clearml_url = f"https://app.clear.ml/projects/{task.project}/experiments/{task.id}"
    
    # Final summary HTML
    summary = f"""
    <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 14px; padding: 28px; box-shadow: 0 8px 16px rgba(59, 130, 246, 0.2);">
        <h2 style="color: #1e40af; font-size: 28px; font-weight: 900; margin: 0 0 24px 0; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 32px;">🎉</span> Training Complete!
        </h2>
        
        <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <h3 style="color: #1f2937; font-weight: 700; margin: 0 0 15px 0; font-size: 18px;">📊 Final Results</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px;">
                <div style="background: #f0fdf4; padding: 15px; border-radius: 8px; border-left: 4px solid #10b981;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Best Test Accuracy</p>
                    <p style="color: #10b981; font-size: 32px; font-weight: 800; margin: 5px 0;">{max(test_metrics['acc']):.2f}%</p>
                </div>
                <div style="background: #eff6ff; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Final Train Accuracy</p>
                    <p style="color: #3b82f6; font-size: 32px; font-weight: 800; margin: 5px 0;">{train_metrics['acc'][-1]:.2f}%</p>
                </div>
                <div style="background: #fef3c7; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b;">
                    <p style="color: #6b7280; font-size: 12px; margin: 0;">Total Epochs</p>
                    <p style="color: #f59e0b; font-size: 32px; font-weight: 800; margin: 5px 0;">{int(epochs)}</p>
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; padding: 20px; color: white; margin-bottom: 20px;">
            <h3 style="margin: 0 0 12px 0; font-size: 18px; font-weight: 700;">🔗 View in ClearML Dashboard</h3>
            <a href="{clearml_url}" target="_blank" style="color: white; font-size: 14px; text-decoration: underline; word-break: break-all;">
                {clearml_url}
            </a>
        </div>
        
        <div style="background: white; border-radius: 12px; padding: 20px;">
            <h3 style="color: #1f2937; font-weight: 700; margin: 0 0 15px 0; font-size: 16px;">✅ Automatically Logged to ClearML:</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                <div style="background: #f9fafb; padding: 12px; border-radius: 6px;">
                    <p style="margin: 0; color: #374151; font-size: 14px;">✓ All hyperparameters</p>
                </div>
                <div style="background: #f9fafb; padding: 12px; border-radius: 6px;">
                    <p style="margin: 0; color: #374151; font-size: 14px;">✓ Training/test metrics per epoch</p>
                </div>
                <div style="background: #f9fafb; padding: 12px; border-radius: 6px;">
                    <p style="margin: 0; color: #374151; font-size: 14px;">✓ Model weights and artifacts</p>
                </div>
                <div style="background: #f9fafb; padding: 12px; border-radius: 6px;">
                    <p style="margin: 0; color: #374151; font-size: 14px;">✓ Git commit information</p>
                </div>
                <div style="background: #f9fafb; padding: 12px; border-radius: 6px;">
                    <p style="margin: 0; color: #374151; font-size: 14px;">✓ Python environment</p>
                </div>
                <div style="background: #f9fafb; padding: 12px; border-radius: 6px;">
                    <p style="margin: 0; color: #374151; font-size: 14px;">✓ Console output logs</p>
                </div>
            </div>
        </div>
        
        <div style="background: rgba(59, 130, 246, 0.1); padding: 16px; border-radius: 8px; margin-top: 20px;">
            <h4 style="color: #1e40af; font-weight: 700; margin: 0 0 10px 0; font-size: 14px;">📈 Next Steps:</h4>
            <ul style="margin: 0; padding-left: 24px; color: #3b82f6; font-size: 13px; line-height: 2;">
                <li>Compare with other experiments in ClearML dashboard</li>
                <li>Clone this configuration for hyperparameter tuning</li>
                <li>Download the trained model for deployment</li>
                <li>Reproduce this exact experiment on any machine</li>
            </ul>
        </div>
    </div>
    """
    
    return plot_fig, output_log, summary, clearml_url

# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
"""

# Gradio Interface
with gr.Blocks(
    title="ClearML MLOps Demo",
    css=custom_css,
    theme=gr.themes.Soft(primary_hue="blue")
) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 6px 16px rgba(59, 130, 246, 0.2);">
        <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4); margin: 0 auto 20px auto;">
            <span style="font-size: 44px;">🚀</span>
        </div>
        
        <h1 style="font-size: 52px; font-weight: 900; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0 0 15px 0;">
            ClearML
        </h1>
        
        <p style="font-size: 26px; color: #1f2937; font-weight: 700; margin: 12px 0;">Experiment Tracking Dashboard</p>
        <p style="font-size: 16px; color: #6b7280; font-weight: 500; margin-bottom: 24px;">Interactive ML Training with Auto-Magical Experiment Management</p>
        
        <div style="display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; max-width: 700px; margin: 0 auto;">
            <span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);">Auto-Logging</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(139, 92, 246, 0.3);">MLOps</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(16, 185, 129, 0.3);">Open Source</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 8px 18px; border-radius: 25px; font-size: 14px; font-weight: 700; box-shadow: 0 2px 6px rgba(249, 115, 22, 0.3);">PyTorch</span>
        </div>
    </div>
    """)
    
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 2px solid #a855f7; border-radius: 14px; padding: 24px; margin-bottom: 25px;">
        <h3 style="color: #6b21a8; font-size: 20px; font-weight: 700; margin: 0 0 15px 0;">✨ The Magic: Just 2 Lines of Code</h3>
        <div style="background: #1f2937; border-radius: 10px; padding: 20px; font-family: 'Courier New', monospace;">
            <pre style="margin: 0; color: #10b981; font-size: 14px; line-height: 1.8;"><span style="color: #8b5cf6;">from</span> clearml <span style="color: #8b5cf6;">import</span> Task
task = Task.init(project_name=<span style="color: #fbbf24;">'MNIST'</span>, task_name=<span style="color: #fbbf24;">'Training'</span>)

<span style="color: #6b7280;"># That's it! Everything below is now tracked automatically ✨</span></pre>
        </div>
        <p style="color: #7c3aed; font-size: 13px; margin: 15px 0 0 0; font-weight: 600;">
            🎯 No manual logging, no boilerplate code - ClearML handles everything!
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h3 style='color: #3b82f6; font-size: 22px; font-weight: 700; margin-bottom: 15px;'>🎛️ Training Configuration</h3>")
            
            batch_size = gr.Slider(
                minimum=16, maximum=256, value=64, step=16,
                label="Batch Size",
                info="Number of samples per batch"
            )
            
            learning_rate = gr.Slider(
                minimum=0.0001, maximum=0.01, value=0.001, step=0.0001,
                label="Learning Rate",
                info="Step size for optimizer"
            )
            
            hidden_size = gr.Slider(
                minimum=64, maximum=512, value=128, step=64,
                label="Hidden Layer Size",
                info="Number of neurons in hidden layer"
            )
            
            dropout = gr.Slider(
                minimum=0.0, maximum=0.5, value=0.25, step=0.05,
                label="Dropout Rate",
                info="Regularization to prevent overfitting"
            )
            
            epochs = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Number of Epochs",
                info="Training iterations over full dataset"
            )
            
            train_btn = gr.Button(
                "🚀 Start Training with ClearML",
                variant="primary",
                size="lg"
            )
            
            gr.HTML("""
            <hr style="margin: 25px 0; border: 1px solid #e5e7eb;">
            <div style="background: #f0fdf4; border: 2px solid #10b981; border-radius: 10px; padding: 20px;">
                <h4 style="color: #065f46; margin: 0 0 12px 0; font-size: 16px; font-weight: 700;">📊 Auto-Tracked by ClearML:</h4>
                <ul style="margin: 0; padding-left: 20px; color: #047857; font-size: 14px; line-height: 2;">
                    <li><strong>Hyperparameters</strong> - All config values</li>
                    <li><strong>Metrics</strong> - Loss, accuracy per epoch</li>
                    <li><strong>Model</strong> - Saved PyTorch weights</li>
                    <li><strong>Code</strong> - Git commit + changes</li>
                    <li><strong>Environment</strong> - Packages, versions</li>
                    <li><strong>Console</strong> - All training output</li>
                    <li><strong>System</strong> - GPU/CPU/RAM usage</li>
                </ul>
                <div style="background: #d1fae5; padding: 12px; border-radius: 6px; margin-top: 15px; text-align: center;">
                    <p style="color: #065f46; font-weight: 700; margin: 0; font-size: 14px;">✨ Zero extra code required!</p>
                </div>
            </div>
            """)
        
        with gr.Column(scale=2):
            gr.HTML("<h3 style='color: #10b981; font-size: 22px; font-weight: 700; margin-bottom: 15px;'>📈 Training Results</h3>")
            
            plot_output = gr.Plot(label="Training Curves")
            
            with gr.Accordion("📝 Training Log", open=False):
                log_output = gr.Textbox(
                    label="Console Output",
                    lines=10,
                    max_lines=20
                )
            
            summary_output = gr.HTML(label="Experiment Summary")
            
            clearml_link = gr.Textbox(
                label="🔗 ClearML Experiment URL",
                interactive=False
            )
    
    gr.HTML("""
    <hr style="border: 2px solid #e5e7eb; margin: 40px 0;">
    
    <div style="text-align: center; padding: 28px; background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); border-radius: 16px; box-shadow: 0 4px 8px rgba(0,0,0,0.08);">
        <h3 style="color: #3b82f6; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">👨‍💻 About This Demo</h3>
        <p style="color: #1f2937; margin: 10px 0; font-size: 16px; line-height: 1.6;">
            Built for <strong style="color: #3b82f6;">ClearML</strong> by 
            <strong style="color: #10b981;">Anju Vilashni Nandhakumar</strong>
        </p>
        <div style="margin: 20px 0; padding: 18px; background: white; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <p style="margin: 6px 0; font-size: 14px;">
                📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: #3b82f6; font-weight: 600;">nandhakumar.anju@gmail.com</a>
            </p>
            <p style="margin: 6px 0; font-size: 14px;">
                💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: #3b82f6; font-weight: 600;">LinkedIn</a> | 
                💻 <a href="https://github.com/Av1352" target="_blank" style="color: #3b82f6; font-weight: 600;">GitHub</a> | 
                🌐 <a href="https://vxanju.com" target="_blank" style="color: #3b82f6; font-weight: 600;">Portfolio</a>
            </p>
        </div>
        <p style="color: #6b7280; font-size: 14px; margin: 12px 0; font-weight: 600;">
            <strong style="color: #3b82f6;">Tech Stack:</strong> ClearML, PyTorch, MNIST, Gradio, Matplotlib
        </p>
        <hr style="border: 1px solid #e5e7eb; margin: 20px 0;">
        <p style="color: #9ca3af; font-size: 13px; font-style: italic; line-height: 1.6;">
            Demonstrating ClearML's MLOps capabilities through practical CNN training.<br>
            Shows automatic experiment tracking, hyperparameter logging, and model versioning.
        </p>
    </div>
    """)
    
    # Wire up the training
    train_btn.click(
        fn=train_model,
        inputs=[batch_size, learning_rate, hidden_size, dropout, epochs],
        outputs=[plot_output, log_output, summary_output, clearml_link]
    )

if __name__ == "__main__":
    demo.launch()