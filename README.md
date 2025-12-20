# 🚀 AI/ML Engineering Portfolio

> Advanced machine learning and computer vision systems across healthcare, computational photography, and infrastructure domains

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents
- [Featured Projects](#featured-projects)
- [Technical Capabilities](#technical-capabilities)
- [About](#about)
- [Contact](#contact)

---

## 🎯 Featured Projects

### 🔬 Computational Photography

#### Low-Light Image Enhancement System
**[Live Demo](https://huggingface.co/spaces/av1352/Glass-imaging)** | **[Source Code](./glass-imaging)**

Advanced image enhancement system for low-light photography using adaptive algorithms and multi-stage processing.

**Technical Implementation:**
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for intelligent local contrast enhancement
- LAB color space processing for luminance-chrominance separation
- Non-local means denoising with edge preservation
- Multi-stage pipeline: brightness, contrast, saturation, and sharpness optimization

**Technologies:** Python, OpenCV, Gradio, NumPy, PIL

**Results:**
- 200% average brightness improvement in low-light scenarios
- 150% contrast enhancement while maintaining natural appearance
- Real-time processing capability via web interface

---

### 🏥 Medical AI Systems

#### Histopathology Tumor Classification
Deep learning classifier for cancer detection in histopathology images.

**Technical Implementation:**
- ResNet-50 architecture with transfer learning
- Custom data augmentation pipeline for medical imaging
- Class imbalance handling with weighted loss functions
- Grad-CAM visualization for model interpretability

**Technologies:** PyTorch, OpenCV, scikit-learn

**Performance:**
- 96% accuracy on test set
- 94% sensitivity, 97% specificity
- Sub-second inference time for clinical deployment

---

#### Deepfake Video Detection
Authentication system for detecting manipulated video content.

**Technical Implementation:**
- Spatiotemporal CNN architecture for frame-sequence analysis
- Optical flow analysis for motion artifacts
- Facial landmark tracking for consistency verification
- Real-time processing pipeline

**Technologies:** TensorFlow, OpenCV, dlib

**Performance:**
- 95% accuracy on benchmark datasets (FaceForensics++, DFDC)
- 30 FPS processing on standard hardware
- Robust to compression and resolution variations

---

### 🤖 AI Infrastructure & MLOps

#### AI Agent Framework
Serverless AI infrastructure demonstrating modern agent orchestration patterns.

**Technical Implementation:**
- Modular agent architecture with role-based specialization
- Multi-turn conversation management with context preservation
- RESTful API with FastAPI
- Gradio interface for rapid prototyping

**Technologies:** Python, FastAPI, Gradio, OpenAI API

**Features:**
- Three specialized agents: Research Summarizer, Code Reviewer, Product Critic
- Temperature and format controls for output customization
- Extensible architecture for additional agent types

---

### 👁️ Computer Vision Applications

#### Real-Time Sign Language Recognition
ML system for translating Indian Sign Language to text in real-time.

**Technical Implementation:**
- MediaPipe for hand landmark detection
- LSTM network for gesture sequence classification
- Custom dataset of 5,000+ gesture samples
- Real-time webcam processing pipeline

**Technologies:** TensorFlow, MediaPipe, OpenCV

**Performance:**
- 92% accuracy across 26 gesture classes
- 15 FPS real-time processing
- Low-latency inference (<100ms)

---

#### Intelligent Recommendation System
Collaborative filtering system for personalized content recommendations.

**Technical Implementation:**
- Matrix factorization with SVD
- Hybrid approach combining collaborative and content-based filtering
- Cold-start problem mitigation strategies
- A/B testing framework for validation

**Technologies:** Python, scikit-learn, pandas, NumPy

**Business Impact:**
- 25% increase in user engagement
- 18% improvement in conversion rates
- Deployed for 10,000+ active users

---

## 💻 Technical Capabilities

### Machine Learning & Deep Learning
- **Frameworks:** PyTorch, TensorFlow, Keras, scikit-learn
- **Computer Vision:** OpenCV, PIL, MediaPipe, albumentations
- **NLP:** Transformers, spaCy, NLTK
- **Model Optimization:** ONNX, TensorRT, quantization, pruning

### Software Engineering
- **Languages:** Python, JavaScript, SQL, C++
- **Web Frameworks:** FastAPI, Flask, Streamlit, Gradio
- **Databases:** PostgreSQL, MongoDB, Redis
- **Cloud & DevOps:** AWS (EC2, S3, Lambda), Docker, Git, CI/CD

### Specialized Domains
- **Medical Imaging:** DICOM processing, image segmentation, classification
- **Computational Photography:** HDR, low-light enhancement, denoising
- **MLOps:** Experiment tracking, model versioning, deployment pipelines
- **Edge AI:** Model optimization, mobile deployment, real-time inference

---

## 📊 Project Metrics

| Category | Projects | Avg. Accuracy | Deployments |
|----------|----------|---------------|-------------|
| Medical AI | 3 | 94% | 2 |
| Computer Vision | 4 | 91% | 3 |
| NLP | 2 | 89% | 1 |
| MLOps | 2 | N/A | 2 |

**Total:** 11 production-ready ML systems | **Code Coverage:** 85%+ | **Documentation:** Comprehensive

---

## 🎓 About

Machine Learning Engineer with expertise in medical imaging, computer vision, and production ML systems. Specialized in building end-to-end AI solutions from research to deployment.

**Core Competencies:**
- Deep learning architecture design and optimization
- Medical image analysis and diagnostic AI
- Real-time computer vision systems
- MLOps and production deployment
- Cross-functional collaboration with clinical and business teams

**Research Interests:**
- Computational photography and neural image processing
- Few-shot learning for medical applications
- Efficient neural architectures for edge deployment
- Explainable AI for healthcare

---

## 📫 Contact

**Anju Vilashni Nandhakumar**  
Machine Learning Engineer

- 🌐 **Portfolio:** [vxanju.com](https://vxanju.com)
- 📧 **Email:** nandhakumar.anju@gmail.com
- 💼 **LinkedIn:** [linkedin.com/in/anju-vilashni](https://www.linkedin.com/in/anju-vilashni/)
- 🐙 **GitHub:** [github.com/Av1352](https://github.com/Av1352)

**Status:** Open to ML Engineer opportunities | F1 OPT (Seeking H1B sponsorship)

---

## 📄 License

This repository is MIT licensed. See [LICENSE](LICENSE) for details.

---

## ⭐ Support

If you find these projects useful, please consider:
- ⭐ Starring this repository
- 🔄 Sharing with others in the ML community
- 💬 Providing feedback or suggestions

---

*Last Updated: December 2025*
```

---

## 🗂️ Complete Folder Structure
Job-search-demos/
├── README.md                    ← Use the above
├── LICENSE                      ← Add MIT license
├── .gitignore                   ← Python/Node ignores
│
├── glass-imaging/
│   ├── README.md
│   ├── app.py
│   ├── requirements.txt
│   └── assets/
│       └── demo-screenshot.png
│
├── langbase/
│   ├── README.md
│   ├── app.py
│   ├── requirements.txt
│   └── static/
│       └── index.html
│
├── medical-imaging/
│   ├── tumor-classification/
│   │   ├── README.md
│   │   ├── train.py
│   │   └── requirements.txt
│   └── deepfake-detection/
│       ├── README.md
│       └── model.py
│
├── computer-vision/
│   ├── sign-language/
│   └── recommendation-system/
│
└── docs/
    ├── technical-overview.md
    └── deployment-guide.md
```
