"""
Image alignment to handle resolution and device variance
"""
import cv2
import numpy as np

class ImageAligner:
    """Align images from different devices/resolutions"""
    
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=5000)
        
    def align_images(self, img1: np.ndarray, img2: np.ndarray, max_features: int = 5000):
        """
        Align img2 to match img1 using feature-based registration
        
        Handles:
        - Different resolutions
        - Slight rotations
        - Device viewport differences
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Detect ORB features
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            # Fallback: simple resize
            return cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Match features using FLANN
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Filter good matches (Lowe's ratio test)
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            # Not enough matches, fallback to resize
            return cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Extract matched keypoint locations
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Warp img2 to align with img1
        h, w = img1.shape[:2]
        aligned = cv2.warpPerspective(img2, M, (w, h))
        
        return aligned
    
    def simple_resize_align(self, img1: np.ndarray, img2: np.ndarray):
        """Simple alignment by resizing to same dimensions"""
        return cv2.resize(img2, (img1.shape[1], img1.shape[0]))