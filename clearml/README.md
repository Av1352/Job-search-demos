# 🔬 ClearML Experiment Tracking Demo

> Production-ready MLOps demonstration showcasing ClearML's auto-magical experiment tracking

**Project:** MNIST CNN Classification with Complete Experiment Management

---

## 🎯 What This Demonstrates

This demo showcases ClearML's core MLOps capabilities through a complete training pipeline:

### ClearML Features Highlighted
- **✨ Auto-Magical Tracking** - Just 2 lines of code to track everything
- **📊 Real-time Metrics** - Loss, accuracy, and custom metrics
- **🔧 Hyperparameter Logging** - Automatic capture of all parameters
- **🎨 Model Versioning** - Automatic model storage and retrieval
- **🔄 Experiment Comparison** - Compare multiple runs side-by-side
- **📝 Git Integration** - Track code versions and uncommitted changes
- **🚀 Reproducibility** - Clone any experiment with one click

---

## 🚀 Quick Start

### 1. Setup ClearML Account
```bash
# Sign up for free at https://app.clear.ml
# Get your API credentials from Settings > Workspace

# Initialize ClearML
clearml-init
```

### 2. Install Dependencies
```bash
cd Job-search-demos/clearml
pip install -r requirements.txt
```

### 3. Run Single Experiment
```bash
# Train with default hyperparameters
python train_mnist.py

# Or customize hyperparameters
python train_mnist.py --batch_size 128 --learning_rate 0.01 --epochs 10
```

### 4. Run Multiple Experiments
```bash
# Run 4 different configurations automatically
python train_experiments.py
```

### 5. View Results

Go to [https://app.clear.ml](https://app.clear.ml) and see:
- Real-time training metrics
- Experiment comparison dashboard
- Model artifacts and code
- GPU/CPU utilization
- Hyperparameter analysis

---

## 📊 Architecture
```
train_mnist.py
    ↓
Task.init()  ← Just 2 lines to enable tracking!
    ↓
[Automatic Logging]
    ├── Hyperparameters (from argparse)
    ├── Git info (commit hash, branch, diffs)
    ├── Environment (Python packages, versions)
    ├── Console output (all prints)
    ├── Scalars (loss, accuracy per epoch)
    ├── Models (PyTorch state_dicts)
    └── Artifacts (any files you save)
    ↓
ClearML Server
    ↓
Beautiful Web Dashboard!
```

---

## 🔬 What Gets Tracked Automatically

### Without Any Extra Code:
- ✅ All hyperparameters from argparse
- ✅ Git commit hash and branch
- ✅ Uncommitted code changes
- ✅ Python environment (pip freeze)
- ✅ Console output (stdout/stderr)
- ✅ System metrics (CPU/GPU/RAM)

### With Simple Logger Calls:
- ✅ Training/validation metrics
- ✅ Custom scalars and plots
- ✅ Images and debug samples
- ✅ Confusion matrices
- ✅ Model weights

---

## 🎨 Example Experiments

### Experiment 1: Baseline
```bash
python train_mnist.py \
  --batch_size 64 \
  --learning_rate 0.001 \
  --hidden_size 128 \
  --dropout 0.25 \
  --epochs 5
```

### Experiment 2: Higher Learning Rate
```bash
python train_mnist.py \
  --learning_rate 0.01
```

### Experiment 3: Larger Network
```bash
python train_mnist.py \
  --hidden_size 256
```

### Experiment 4: Less Regularization
```bash
python train_mnist.py \
  --dropout 0.1
```

**Result:** Compare all 4 in ClearML dashboard to find optimal config! 📈

---

## 💡 Key ClearML Advantages

### vs Manual Tracking (Spreadsheets)
- ❌ Manual: Copy-paste metrics to Excel
- ✅ ClearML: Automatic, real-time dashboard

### vs MLflow
- ❌ MLflow: Requires explicit logging calls everywhere
- ✅ ClearML: Auto-logs everything with Task.init()

### vs Weights & Biases
- ❌ W&B: SaaS-only, per-seat pricing
- ✅ ClearML: Open-source, self-hostable, free tier

---

## 🏆 ClearML's Production Advantages

### For Teams:
- **Experiment Reproducibility** - Clone any experiment with one click
- **Collaboration** - Share experiments, compare results
- **Resource Management** - See GPU utilization across team
- **Model Registry** - Centralized model storage
- **Pipeline Automation** - Chain experiments together

### For Production:
- **Remote Execution** - Run experiments on cloud GPUs
- **Hyperparameter Optimization** - Automated tuning
- **Model Deployment** - Serve models from registry
- **A/B Testing** - Track model performance in production
- **Cost Tracking** - Monitor compute costs per experiment

---

## 📈 Results You'll See

After running experiments, you'll see in ClearML:

### 1. Experiment Table
| Experiment | LR | Hidden Size | Dropout | Test Acc | Train Time |
|-----------|----|----|-------|----------|------------|
| Baseline | 0.001 | 128 | 0.25 | 98.5% | 3m 45s |
| Higher LR | 0.01 | 128 | 0.25 | 98.2% | 3m 42s |
| Larger Hidden | 0.001 | 256 | 0.25 | 98.8% | 5m 12s |
| Less Dropout | 0.001 | 128 | 0.1 | 98.9% | 3m 48s |

### 2. Training Curves
- Loss over time (train vs test)
- Accuracy over time (train vs test)
- Side-by-side comparison of all experiments

### 3. System Metrics
- GPU utilization %
- Memory usage
- CPU usage

### 4. Model Artifacts
- Download best_model.pth
- See exact code used
- View full environment

---

## 🛠️ Advanced Features

### Remote Execution
```bash
# Clone experiment #123 and run on remote GPU
clearml-task --clone 123 --queue gpu_queue
```

### Hyperparameter Optimization
```python
from clearml.automation import HyperParameterOptimizer

optimizer = HyperParameterOptimizer(
    base_task_id='your_task_id',
    hyper_parameters=[
        UniformParameterRange('learning_rate', min_value=0.0001, max_value=0.1),
        UniformIntegerParameterRange('hidden_size', min_value=64, max_value=512)
    ],
    objective_metric_title='test',
    objective_metric_series='accuracy',
    objective_metric_sign='max'
)

optimizer.start()
```

---

## 📚 Technical Details

### Model Architecture
```
Conv2D(1→32, 3x3) → ReLU → MaxPool → Dropout
Conv2D(32→64, 3x3) → ReLU → MaxPool → Dropout
Flatten → FC(3136→128) → ReLU
FC(128→10) → Output
```

**Parameters:** ~1.2M  
**Input:** 28x28 grayscale images  
**Output:** 10 classes (digits 0-9)

### Training Details
- **Dataset:** MNIST (60K train, 10K test)
- **Optimizer:** Adam
- **Loss:** CrossEntropyLoss
- **Batch Size:** 64 (configurable)
- **Epochs:** 5 (configurable)
- **Hardware:** CPU or CUDA GPU

---

## 🎯 Why ClearML?

### The Problem ClearML Solves

**Before ClearML:**
```python
# Manually track everything 😭
results = {
    'lr': 0.001,
    'batch_size': 64,
    'train_acc': 98.5,
    'test_acc': 97.2
}
with open('results.json', 'w') as f:
    json.dump(results, f)  # Hope you don't lose this file!
```

**With ClearML:**
```python
# Just add 2 lines 🎉
task = Task.init(project_name='My Project', task_name='Experiment 1')
# Everything else is automatic!
```

---

## 👨‍💻 About This Demo

**Built by:** Anju Vilashni Nandhakumar  
**Purpose:** Application to ClearML  
**Contact:** nandhakumar.anju@gmail.com  
**LinkedIn:** [linkedin.com/in/anju-vilashni](https://www.linkedin.com/in/anju-vilashni/)  
**GitHub:** [github.com/Av1352](https://github.com/Av1352)

---

### Why ClearML?

I'm passionate about MLOps and building production-ready ML systems. ClearML's approach to "auto-magical" experiment tracking is exactly the kind of developer experience that accelerates ML teams. The platform's ability to track everything without code changes, combined with powerful orchestration and deployment features, makes it an ideal solution for scaling ML operations.

My background in medical imaging and deep learning has given me firsthand experience with the challenges of experiment tracking and model management. ClearML's comprehensive platform addresses these pain points elegantly.

---

**⭐ Star this repo if you found this demo useful!**

*This is a technical demonstration project and is not affiliated with or endorsed by ClearML.*