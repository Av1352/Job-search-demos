"""
Screenshot capture utilities
"""
import numpy as np
from PIL import Image
import io

class ScreenshotCapture:
    """Utilities for capturing and processing screenshots"""
    
    @staticmethod
    def normalize_screenshot(img: np.ndarray, target_size: tuple = (1920, 1080)) -> np.ndarray:
        """
        Normalize screenshot to standard size
        
        Handles different device resolutions
        """
        import cv2
        h, w = img.shape[:2]
        
        # Calculate aspect ratio
        aspect = w / h
        target_aspect = target_size[0] / target_size[1]
        
        if abs(aspect - target_aspect) < 0.01:
            # Same aspect ratio, simple resize
            return cv2.resize(img, target_size)
        else:
            # Different aspect ratio, pad to maintain aspect
            if aspect > target_aspect:
                # Wider image, pad top/bottom
                new_w = target_size[0]
                new_h = int(new_w / aspect)
                resized = cv2.resize(img, (new_w, new_h))
                
                # Pad
                pad_top = (target_size[1] - new_h) // 2
                pad_bottom = target_size[1] - new_h - pad_top
                padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                # Taller image, pad left/right
                new_h = target_size[1]
                new_w = int(new_h * aspect)
                resized = cv2.resize(img, (new_w, new_h))
                
                # Pad
                pad_left = (target_size[0] - new_w) // 2
                pad_right = target_size[0] - new_w - pad_left
                padded = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            return padded
    
    @staticmethod
    def preprocess_for_diff(img: np.ndarray) -> np.ndarray:
        """
        Preprocess screenshot for robust diffing
        
        - Noise reduction
        - Color normalization
        - Edge enhancement
        """
        import cv2
        
        # Bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Normalize brightness
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_normalized = clahe.apply(l)
        
        # Merge back
        normalized_lab = cv2.merge([l_normalized, a, b])
        normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)
        
        return normalized_rgb