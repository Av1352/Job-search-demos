# 🔬 Glass Imaging - Low-Light Enhancement Demo

**Classical low-light enhancement inspired by Glass Imaging’s AI-first camera pipeline**

Built for **Glass Imaging** by Anju Nandhakumar  

🔗 **[Live Demo](https://huggingface.co/spaces/av1352/Glass-imaging)** | 💼 **[LinkedIn](https://linkedin.com/in/anju-vilashni)** | 🌐 **[Portfolio](https://vxanju.com)**

---

## What This Does

AI-powered enhancement for dark, noisy photos using classical computer vision.

**Features:**
- CLAHE-based local contrast enhancement  
- LAB color space processing to preserve realistic colors  
- Optional non-local means denoising for very noisy images  
- Tuned boosts to brightness, contrast, saturation, and sharpness

**Example Flow:**  
Upload low-light image → (optional) denoise → RGB → LAB → CLAHE on L channel → multi-stage enhancement → Download brighter, clearer image.

---

## Why It Matters

**Problem:** Low-light smartphone photos are often dark, noisy, and washed out.  
**Solution:** This pipeline recovers detail and contrast while keeping colors natural, making low-light images more usable for everyday photography, security, and medical contexts.

**Inspiration:** Glass Imaging uses neural ISPs and co-designed optics + AI to deliver DSLR-level quality from ultra-thin smartphone cameras.
---

## Demo Features

✓ Upload image + toggle denoising  
✓ Side-by-side before/after visualization  
✓ Adjustable enhancement strength presets (subtle → strong)  
✓ Download enhanced output image   

---

## Tech Stack

Python • OpenCV • scikit-image • NumPy • Gradio UI 

---

## Future Direction

- Swap classical pipeline for CNN/GAN-based low-light enhancement  
- Move earlier in the chain to operate on RAW sensor data  
- Optimize for real-time, on-device processing similar to Glass Imaging’s Neural Night stack  

---

**Contact:** [nandhakumar.anju@gmail.com](mailto:nandhakumar.anju@gmail.com)  

Built with ❤️ for Glass Imaging