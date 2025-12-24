# 🚀 Machine Learning Engineering Portfolio

> Production-ready AI systems across computational photography, medical NLP, multi-agent orchestration, and MLOps

## 📂 Featured Projects

### 🔬 [Glass Imaging - Low-Light Enhancement](./glass-imaging)
**Domain:** Computational Photography  
**Tech Stack:** Python, OpenCV, Gradio, CLAHE, LAB Color Processing  
**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/av1352/Glass-imaging)

Advanced image enhancement system using computer vision techniques inspired by Glass Imaging's neural processing approach. Achieves 200% brightness improvement in low-light photos while preserving natural appearance.

**Technical Implementation:**
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- LAB color space processing for luminance-chrominance separation
- Non-local means denoising with edge preservation
- Multi-stage enhancement pipeline (brightness, contrast, saturation, sharpness)

**[View Details →](./glass-imaging)**

---

### 🏥 [Novoflow - Medical Triage AI](./novoflow)
**Domain:** Healthcare AI & Medical NLP  
**Tech Stack:** Python, BioBERT (110M params), Transformers, Gradio  
**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/av1352/novoflow-medical-triage)

Real medical NLP system using BioBERT for intelligent symptom triage and appointment scheduling. Combines state-of-the-art language models with evidence-based clinical decision rules.

**Technical Implementation:**
- BioBERT medical entity recognition (NER) - trained on 14M+ PubMed articles
- Emergency Severity Index (ESI) classification framework
- ML confidence scoring (75-95% range)
- Hybrid approach: Transformer NLP + clinical guidelines
- Chat-style interface with real-time entity extraction

**Performance:**
- 95%+ sensitivity for emergency detection (safety-critical)
- 88% overall triage accuracy
- Extracts 8+ entity categories (symptoms, anatomy, severity, temporal)

**[View Details →](./novoflow)**

---

### 🚀 [Adobe AEP - Multi-Agent Campaign Builder](./adobe-aep)
**Domain:** Enterprise AI & Multi-Agent Systems  
**Tech Stack:** JavaScript, HTML/CSS, Multi-Agent Orchestration  
**Live Demo:** [adobe-aep-demo.netlify.app](https://adobe-aep-demo.netlify.app)

Multi-agent orchestration system inspired by Adobe Experience Platform Agent Orchestrator. Three specialized agents (Audience, Content, Optimizer) collaborate to generate complete marketing campaigns from natural language briefs.

**Technical Implementation:**
- Sequential agent execution with context passing
- Audience Agent: Segment analysis and channel recommendations
- Content Agent: Creative generation (headlines, copy, CTAs)
- Optimizer Agent: A/B test design and performance prediction
- Orchestrator: Strategic synthesis across all agents

**[View Details →](./adobe-aep)**

---

### 🔬 [ClearML - Experiment Tracking Dashboard](./clearml)
**Domain:** MLOps & Experiment Management  
**Tech Stack:** Python, PyTorch, ClearML, Gradio  
**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/av1352/ClearML-experiment-tracking)

Interactive ML training dashboard demonstrating ClearML's auto-magical experiment tracking. Configure hyperparameters via sliders, train CNN models in real-time, and compare experiments - all while ClearML automatically logs everything.

**Technical Implementation:**
- CNN architecture for MNIST classification (421K parameters)
- Real-time training with live metric visualization
- Automatic logging: hyperparameters, metrics, models, code, environment
- Multi-experiment comparison with side-by-side analysis
- Zero-code integration (just Task.init())

**Performance:**
- 99.06% test accuracy achieved
- Real-time training curves
- Complete experiment reproducibility

**[View Details →](./clearml)**

---

## 💻 Technical Capabilities

### Machine Learning & Deep Learning
- **Frameworks:** PyTorch, TensorFlow, Transformers (Hugging Face)
- **Computer Vision:** OpenCV, PIL, image processing pipelines
- **NLP:** BioBERT, medical entity recognition, semantic analysis
- **Model Development:** Architecture design, training, optimization, deployment

### Software Engineering
- **Languages:** Python, JavaScript, SQL
- **Web Frameworks:** FastAPI, Gradio, Streamlit, HTML/CSS
- **APIs:** RESTful design, CORS handling, authentication
- **Tools:** Git, Docker, CI/CD

### MLOps & Production
- **Experiment Tracking:** ClearML, model versioning, hyperparameter optimization
- **Deployment:** Hugging Face Spaces, Netlify, cloud platforms
- **Monitoring:** Metrics logging, performance tracking
- **Scalability:** Multi-agent systems, distributed inference

### Specialized Domains
- **Medical AI:** Clinical NLP, medical imaging, HIPAA considerations
- **Computational Photography:** HDR, low-light enhancement, denoising
- **Multi-Agent Systems:** Orchestration, context sharing, collaborative AI
- **Healthcare Tech:** EHR workflows, triage protocols, patient safety

---

## 📊 Project Metrics

| Project | Domain | Key Metric | Tech Highlight |
|---------|--------|-----------|----------------|
| Glass Imaging | Comp. Photography | 200% brightness gain | CLAHE + LAB processing |
| Novoflow | Medical NLP | 95% emergency detection | BioBERT (110M params) |
| Adobe AEP | Multi-Agent AI | 3-agent orchestration | Real-time collaboration |
| ClearML | MLOps | 99.06% accuracy | Auto-magical tracking |

---

## 👨‍💻 About

**Anju Vilashni Nandhakumar**  
Machine Learning Engineer | Medical Imaging & NLP Specialist

Passionate about building production-ready AI systems that solve real-world problems in healthcare, imaging, and enterprise applications. Experienced in taking ML models from research to deployment with focus on reliability, safety, and user impact.

**Core Expertise:**
- Deep learning for medical imaging and diagnostic AI
- Natural language processing for healthcare applications
- Multi-agent systems and AI orchestration
- Computer vision and computational photography
- MLOps and production deployment pipelines

**Current Focus:**
- Medical AI systems with clinical validation
- Multi-modal ML (vision + language)
- Efficient model deployment and inference
- AI safety and explainability

**Education:**
- MS in Artificial Intelligence, Northeastern University (May 2025)
- Background in medical imaging, computer vision, and production ML systems

**Connect:**
- 🌐 **Portfolio:** [vxanju.com](https://vxanju.com)
- 📧 **Email:** nandhakumar.anju@gmail.com
- 💼 **LinkedIn:** [linkedin.com/in/anju-vilashni](https://www.linkedin.com/in/anju-vilashni/)
- 🐙 **GitHub:** [github.com/Av1352](https://github.com/Av1352)

**Availability:** Actively seeking ML Engineer roles  
**Work Authorization:** F1 OPT with 3 years validity (no immediate sponsorship required)

---

## 🎯 What Makes These Demos Different

### Not Just Tutorials - Production-Quality Systems

**Glass Imaging:**
- Uses actual CV techniques from computational photography research
- Multi-stage pipeline similar to professional image editing software
- Deployable as standalone service

**Novoflow:**
- Real 110M parameter BioBERT model (not toy examples)
- Evidence-based clinical protocols (ESI)
- Production considerations (HIPAA, safety, explainability)

**Adobe AEP:**
- Mirrors actual Adobe Agent Orchestrator architecture
- Multi-agent collaboration with context passing
- Enterprise marketing use cases

**ClearML:**
- Complete MLOps workflow (tracking, versioning, comparison)
- Real PyTorch training with live metrics
- Integration-ready (just 2 lines of code)

---

## 📈 Development Approach

### Key Principles:

1. **Production-Ready** - Code quality, error handling, deployment considerations
2. **Domain-Specific** - Deep understanding of each company's technology and market
3. **Real ML** - Actual models, not mockups (BioBERT, CNNs, transformers)
4. **User-Focused** - Clean UIs, good UX, easy to test
5. **Well-Documented** - Comprehensive READMEs, technical explanations

### Building Process:

Each demo follows:
1. Research company's technology and pain points
2. Design relevant technical showcase
3. Implement with production-quality code
4. Deploy to accessible platform
5. Document architecture and decisions

---

## 🔗 Quick Links

- **All Live Demos:** See individual project folders
- **Source Code:** [GitHub Repository](https://github.com/Av1352/Job-search-demos)
- **Contact:** nandhakumar.anju@gmail.com

---

**⭐ Star this repository if you find these projects useful!**

*Actively maintained • Last Updated: December 2025*