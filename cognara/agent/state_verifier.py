"""
State verification using perception before actions
"""
import cv2
import numpy as np

class StateVerifier:
    """Verify UI state before agent acts"""
    
    def __init__(self, visual_diff_engine):
        self.diff_engine = visual_diff_engine
        
    def verify_state(self, expected_screenshot: np.ndarray, actual_screenshot: np.ndarray, threshold: float = 0.90):
        """
        Verify current UI matches expected state
        
        Returns:
            dict with match status and confidence
        """
        # Compute similarity
        diff_result = self.diff_engine.compute_diff(expected_screenshot, actual_screenshot)
        
        # State matches if SSIM > threshold
        matches = diff_result['ssim'] >= threshold
        
        return {
            'matches': matches,
            'confidence': diff_result['ssim'],
            'details': diff_result
        }
    
    def find_element(self, screenshot: np.ndarray, template: np.ndarray, method=cv2.TM_CCOEFF_NORMED):
        """
        Find UI element in screenshot using template matching
        
        Returns:
            Location and confidence if found
        """
        # Convert to grayscale
        gray_screen = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        
        # Template matching
        result = cv2.matchTemplate(gray_screen, gray_template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Check if match is strong enough
        if max_val > 0.8:
            h, w = gray_template.shape
            return {
                'found': True,
                'confidence': float(max_val),
                'location': {
                    'x': int(max_loc[0]),
                    'y': int(max_loc[1]),
                    'width': int(w),
                    'height': int(h)
                }
            }
        else:
            return {
                'found': False,
                'confidence': float(max_val),
                'location': None
            }