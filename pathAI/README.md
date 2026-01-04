# 🔬 PathologyNet – AI Tumor Detection & Classification

**Deep learning pipeline for histopathology tumor detection with explainable AI**

Built for **PathAI** by **Anju Vilashni Nandhakumar** · MS in AI, Northeastern University (2025)  

🔗 **[Live Demo](https://huggingface.co/spaces/Av1352/pathai-tumor-detection)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)**  

---

## What This Does

End-to-end CNN system for classifying H&E histopathology images and visualizing model focus.  

**Features:**
- ResNet50 transfer learning tumor classifier (Benign / Malignant / Suspicious)  
- ~96% classification accuracy with sensitivity and specificity >94%  
- Grad-CAM heatmaps to highlight regions driving the prediction  
- Optional CLAHE enhancement for clearer tissue visualization  

**Example Flow:**  
Upload H&E patch → enhance image → classify tumor status with confidence → view Grad-CAM overlay + key pathological features.  

---

## Model & Performance

**Architecture:**
- Backbone: ResNet50 pretrained on ImageNet  
- Training data: BreakHis breast histopathology images  
- Augmentation: rotation, flip, color jitter, stain-aware normalization  

**Metrics:**
- Accuracy: ~96%  
- Sensitivity (malignant): ~95%  
- Specificity (benign): ~97%  
- AUC-ROC: ~0.98  

---

## Why It Matters

Manual histopathology is the diagnostic gold standard but limited by pathologist workload and global shortages.  

This model illustrates how AI can:  
- Speed up screening and second reads  
- Provide consistent, fatigue-free assistance  
- Offer transparent visual explanations pathologists can inspect  

---

## Tech Stack

PyTorch • ResNet50 • Grad-CAM • OpenCV (CLAHE) • Gradio UI • Hugging Face Spaces  

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)  

*Built with ❤️ for PathAI*  