"""
Defect detection: missing elements, overlaps, clipping
"""
import cv2
import numpy as np

class DefectDetector:
    """Detect UI defects using computer vision"""
    
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000)
        
    def detect_missing_elements(self, img1: np.ndarray, img2: np.ndarray):
        """
        Detect missing UI elements using feature matching
        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Detect ORB keypoints
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return []
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Find unmatched keypoints in baseline (potentially missing elements)
        matched_kp1_indices = set(m.queryIdx for m in matches)
        unmatched_kp1 = [kp for i, kp in enumerate(kp1) if i not in matched_kp1_indices]
        
        defects = []
        
        # Group nearby unmatched keypoints as missing elements
        if len(unmatched_kp1) > 10:  # Significant number of unmatched features
            # Use simple clustering to find missing regions
            points = np.array([kp.pt for kp in unmatched_kp1])
            
            if len(points) > 0:
                # Find concentrated missing regions
                # Simple approach: look for clusters of unmatched points
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                
                if len(x_coords) > 20:
                    defects.append({
                        'type': 'Missing Element',
                        'severity': 'medium',
                        'confidence': min(0.95, len(unmatched_kp1) / len(kp1)),
                        'location': f'Region around ({int(np.mean(x_coords))}, {int(np.mean(y_coords))})',
                        'description': f'{len(unmatched_kp1)} features not found in current image',
                        'expected': f'{len(kp1)} features',
                        'actual': f'{len(kp2)} features, {len(matches)} matched',
                        'agent': 'Element Detection Agent'
                    })
        
        return defects
    
    def detect_layout_shifts(self, img1: np.ndarray, img2: np.ndarray, changed_regions: list):
        """
        Detect layout shifts from changed regions
        """
        defects = []
        
        # Look for large changed regions (likely layout shifts)
        for region in changed_regions:
            if region['area'] > 5000:  # Significant change
                defects.append({
                    'type': 'Layout Shift',
                    'severity': 'high' if region['area'] > 20000 else 'medium',
                    'confidence': 0.92,
                    'location': f"({region['x']}, {region['y']})",
                    'description': f"Large region changed: {region['width']}x{region['height']}px",
                    'expected': 'Stable position',
                    'actual': f'Shifted by ~{region["width"]}px',
                    'agent': 'Visual Diff Agent',
                    'coordinates': region
                })
        
        return defects
    
    def detect_clipping_issues(self, img: np.ndarray):
        """
        Detect potential clipping or overflow issues
        """
        defects = []
        
        # Check edges for cut-off elements
        # Look for high-contrast regions near image borders
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        h, w = edges.shape
        border_width = 5
        
        # Check each border
        borders = {
            'top': edges[:border_width, :],
            'bottom': edges[-border_width:, :],
            'left': edges[:, :border_width],
            'right': edges[:, -border_width:]
        }
        
        for border_name, border_region in borders.items():
            edge_density = np.sum(border_region > 0) / border_region.size
            
            if edge_density > 0.1:  # Significant edges near border
                defects.append({
                    'type': 'Potential Clipping',
                    'severity': 'low',
                    'confidence': 0.75,
                    'location': f'{border_name.title()} edge',
                    'description': f'High edge density near {border_name} border may indicate clipped content',
                    'expected': 'Clean borders',
                    'actual': f'{edge_density:.1%} edge density',
                    'agent': 'Layout Analyzer'
                })
        
        return defects