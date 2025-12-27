"""
Visual diffing using SSIM and learned embeddings
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

class VisualDiffEngine:
    """Multi-method visual comparison"""
    
    def __init__(self):
        self.ssim_threshold = 0.95
        self.pixel_threshold = 2.0  # 2% pixel change threshold
        
    def compute_diff(self, img1: np.ndarray, img2: np.ndarray):
        """
        Compute visual difference using multiple metrics
        
        Returns:
            dict with SSIM, PSNR, MSE, pixel diff percentage, and diff map
        """
        # Ensure same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Compute SSIM
        ssim_score, ssim_diff = ssim(gray1, gray2, full=True)
        
        # Compute PSNR
        psnr_score = psnr(img1, img2)
        
        # Compute MSE
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        
        # Pixel-level difference
        diff_abs = cv2.absdiff(img1, img2)
        diff_gray = cv2.cvtColor(diff_abs, cv2.COLOR_RGB2GRAY)
        
        # Threshold to find significant changes
        _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        changed_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        change_percent = (changed_pixels / total_pixels) * 100
        
        # Generate diff heatmap
        diff_map = (ssim_diff * 255).astype("uint8")
        diff_map = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        # Find contours of changed regions
        contours, _ = cv2.findContours(diff_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes for changed regions
        changed_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Ignore tiny changes
                x, y, w, h = cv2.boundingRect(cnt)
                changed_regions.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': int(area)
                })
        
        return {
            'ssim': float(ssim_score),
            'psnr': float(psnr_score),
            'mse': float(mse),
            'change_percent': float(change_percent),
            'changed_pixels': int(changed_pixels),
            'total_pixels': int(total_pixels),
            'diff_map': diff_map,
            'changed_regions': changed_regions,
            'passed': ssim_score >= self.ssim_threshold and change_percent <= self.pixel_threshold
        }
    
    def generate_diff_visualization(self, img1: np.ndarray, img2: np.ndarray, diff_map: np.ndarray):
        """
        Generate visual diff with red highlighting
        """
        # Ensure same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Create colored overlay for differences
        diff_colored = np.zeros_like(img2)
        diff_colored[diff_map == 255] = [255, 0, 0]  # Red for differences
        
        # Blend with current image
        overlay = cv2.addWeighted(img2, 0.7, diff_colored, 0.3, 0)
        
        # Draw bounding boxes around changed regions
        contours, _ = cv2.findContours(diff_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        return overlay