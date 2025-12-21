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

# Global data loaders (load once)
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
    """
    Train model with ClearML tracking
    """
    # Initialize ClearML Task
    task = Task.init(
        project_name='MNIST Classification',
        task_name=f'Training_bs{int(batch_size)}_lr{learning_rate}_h{int(hidden_size)}',
        tags=['gradio-demo', 'interactive']
    )
    
    # Log hyperparameters to ClearML
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
            
            # Update progress
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
    
    # Final summary
    summary = f"""
🎉 Training Complete!

📊 Final Results:
- Best Test Accuracy: {max(test_metrics['acc']):.2f}%
- Final Train Accuracy: {train_metrics['acc'][-1]:.2f}%
- Total Epochs: {int(epochs)}

🔗 View in ClearML Dashboard:
{clearml_url}

✅ Automatically Logged:
- All hyperparameters
- Training/test metrics per epoch
- Model weights and artifacts
- Git commit information
- Python environment
- Console output

📈 You can now:
- Compare with other experiments
- Clone this configuration
- Download the trained model
- Reproduce this exact experiment
"""
    
    return plot_fig, output_log, summary, clearml_url

# Gradio Interface
with gr.Blocks(title="ClearML MLOps Demo") as demo:
    gr.Markdown("""
    # 🚀 ClearML Experiment Tracking Dashboard
    
    ### Interactive ML Training with Auto-Magical Experiment Management
    
    Train a CNN on MNIST and watch as ClearML automatically tracks **everything** - hyperparameters, 
    metrics, models, code, environment, and more!
    
    **What makes ClearML special:** Just add `Task.init()` and everything is tracked automatically. 
    No manual logging required! 🎯
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Training Configuration")
            
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
            
            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            
            ### 📊 What Gets Tracked Automatically:
            
            ✅ **Hyperparameters** - All config values  
            ✅ **Metrics** - Loss, accuracy per epoch  
            ✅ **Model** - Saved PyTorch weights  
            ✅ **Code** - Git commit + uncommitted changes  
            ✅ **Environment** - Python packages, versions  
            ✅ **Console** - All training output  
            ✅ **System** - GPU/CPU/RAM usage  
            
            **Zero extra code required!** 🎉
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 📈 Training Results")
            
            plot_output = gr.Plot(label="Training Curves")
            
            with gr.Accordion("📝 Training Log", open=False):
                log_output = gr.Textbox(
                    label="Console Output",
                    lines=10,
                    max_lines=20
                )
            
            summary_output = gr.Textbox(
                label="Experiment Summary",
                lines=15
            )
            
            clearml_link = gr.Textbox(
                label="🔗 ClearML Experiment URL",
                interactive=False
            )
    
    gr.Markdown("""
    ---
    
    ## 🎯 How This Showcases ClearML
    
    ### The Magic: Just 2 Lines of Code
```python
    from clearml import Task
    task = Task.init(project_name='MNIST', task_name='Training')
    # That's it! Everything below is now tracked automatically
```
    
    ### What Happens Behind the Scenes:
    
    1. **Auto-detects** your Git repository and captures commit hash
    2. **Captures** all Python packages and versions (`pip freeze`)
    3. **Logs** all arguments and hyperparameters
    4. **Records** all console output (prints, warnings, errors)
    5. **Tracks** scalars, plots, and custom metrics
    6. **Uploads** model files and artifacts
    7. **Monitors** system resources (GPU, CPU, RAM)
    
    ### ClearML vs Competitors:
    
    | Feature | ClearML | MLflow | W&B |
    |---------|---------|--------|-----|
    | Auto-logging | ✅ Zero code | ❌ Manual calls | ❌ Manual calls |
    | Open Source | ✅ Apache 2.0 | ✅ Apache 2.0 | ❌ Proprietary |
    | Self-hostable | ✅ Free | ✅ Free | ❌ Enterprise only |
    | Pipelines | ✅ Built-in | ⚠️ Limited | ⚠️ Limited |
    | Remote Execution | ✅ Yes | ❌ No | ❌ No |
    | HPO | ✅ Built-in | ❌ External | ✅ Paid tier |
    
    ---
    
    ## 🏢 Production Use Cases
    
    ### Medical Imaging Pipeline (Example)
    
    ClearML would track:
    - **Data versioning** - DICOM datasets per hospital/scanner
    - **Model experiments** - ResNet vs EfficientNet vs ViT
    - **Preprocessing** - Different normalization/augmentation strategies
    - **Performance** - Accuracy per pathology type
    - **Deployment** - Which model version is in production
    - **Compliance** - Full audit trail for FDA validation
    
    ### Why ClearML for Production:
    
    ✅ **Reproducibility** - Recreate any experiment exactly  
    ✅ **Collaboration** - Team shares experiments seamlessly  
    ✅ **Resource Management** - Track GPU costs per experiment  
    ✅ **Model Registry** - Centralized model storage  
    ✅ **CI/CD Integration** - Automated training pipelines  
    
    ---
    
    **Built by:** Anju Vilashni Nandhakumar  
    **Contact:** nandhakumar.anju@gmail.com  
    **LinkedIn:** [linkedin.com/in/anju-vilashni](https://www.linkedin.com/in/anju-vilashni/)  
    
    *Demonstrating ClearML's MLOps capabilities through practical implementation*
    """)
    
    # Wire up the training
    train_btn.click(
        fn=train_model,
        inputs=[batch_size, learning_rate, hidden_size, dropout, epochs],
        outputs=[plot_output, log_output, summary_output, clearml_link]
    )

if __name__ == "__main__":
    demo.launch(share=True)
    