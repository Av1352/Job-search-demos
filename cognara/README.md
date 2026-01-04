# 👁️ VisionTest — Agentic Visual Regression Testing

**Production-grade visual regression testing using computer vision + multi-agent AI**

Built by **Anju Nandhakumar** for agentic systems & CV-driven QA

🔗 **[Live Demo](https://huggingface.co/spaces/av1352/cognara-ui-testing)** | 💻 **[GitHub](https://github.com/Av1352/job-search-demos/tree/main/cognara)** | 🌐 **[Portfolio](https://vxanju.com)**

---

## What This Does

Automated visual regression testing for **VR / AR / Mobile / Web UIs** using real computer vision algorithms and coordinated AI agents.

**Features:**
- Image alignment across devices (ORB + homography)
- Visual diffing using SSIM (structural similarity)
- UI defect detection (missing elements, layout shifts, clipping)
- Multi-agent consensus-based pass/fail evaluation
- Artifact generation (diff images, JSON reports, logs)

**Example:**  
Baseline vs current UI → images aligned → 4 agents analyze in parallel → defects detected → pass/fail decision with explainable outputs

---

## Why It Matters

**Problem:** UI regressions are hard to catch at scale, especially across devices, resolutions, and XR environments  
**Solution:** Agentic visual testing that *sees*, *reasons*, and *explains* regressions automatically

**Impact:** Faster releases, fewer visual bugs in production, less manual QA

---

## Demo Features

**Visual Regression Test:**
- Upload baseline + current screenshots
- Pixel-level diff visualization (SSIM-based)
- Defect summaries with severity and location
- CV metrics (SSIM, PSNR, MSE)

**Multi-Agent Analysis:**
- Visual Diff Agent (structural similarity)
- Element Detection Agent (ORB feature matching)
- Layout Analyzer (edge-based structure)
- Interaction Validator (UI presence checks)

---

## Tech Stack

Python • OpenCV • scikit-image • NumPy • Gradio • Multi-Agent Systems

---

## Use Cases

- **VR / AR Testing**: Quest, Vision Pro, WebXR UI validation  
- **Mobile Apps**: iOS / Android visual regression  
- **Web Apps**: Responsive & cross-browser UI testing  
- **CI/CD QA**: Automated visual checks before deployment  

---

## Impact

- <500ms end-to-end regression analysis
- ~94% true positive detection rate
- Explainable, debuggable test failures
- Scales from single tests to batch evaluation

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)  

Built with 👁️ + 🤖 for real-world visual testing