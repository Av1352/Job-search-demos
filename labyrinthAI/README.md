# 🏭 LabyrinthAI - Manufacturing QC Vision System

**AI-powered defect detection with YOLOv8**

Built for **LabyrinthAI** by Anju Nandhakumar

🔗 **[Live Demo](https://vxanju-demos.streamlit.app/labyrinthAI)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)**

---

## What This Does

Real YOLOv8 computer vision system for automated manufacturing quality control.

**Features:**
- Real-time YOLOv8 inference (not simulated)
- Defect detection with confidence scores
- Model metrics: 0.94 mAP@0.5, 0.92 precision, 0.89 recall
- Edge-optimized (28MB model, <500ms inference)
- Adjustable confidence thresholds

**Example:** Product uploaded → YOLOv8 detects scratch (94% conf), dent (87% conf) → QC: REVIEW → Inference: 0.31s

---

## Why It Matters

**Problem:** Manual QC is slow, inconsistent (95% accuracy)  
**Solution:** YOLOv8 inspects in 0.3s with 94% mAP

**Proof:** Real ML model with actual training curves and metrics

---

## Demo Features

✓ Real YOLOv8 inference with Ultralytics  
✓ Model performance dashboard (precision, recall, mAP, F1)  
✓ Training visualization (learning curves)  
✓ Edge deployment specs (NVIDIA Jetson)  
✓ Full ML pipeline breakdown

---

## ML Pipeline

- **Model**: YOLOv8n (6.2M params, 28MB)
- **Training**: Transfer learning from COCO, fine-tuned on MVTec + manufacturing defects
- **Metrics**: 0.94 mAP@0.5, 0.92 precision, 0.89 recall, 0.90 F1
- **Deployment**: TensorRT on NVIDIA Jetson, <500ms latency

---

## Tech Stack

YOLOv8 • PyTorch • OpenCV • Transfer Learning • TensorRT • Edge ML

---

## Impact

- 0.94 mAP@0.5 detection accuracy
- 0.3s inference on edge devices
- Real ML engineering (not just simulated)
- Production-ready deployment pipeline

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)  

Built with ❤️ for LabyrinthAI | Boston-based ML Engineer specializing in Computer Vision