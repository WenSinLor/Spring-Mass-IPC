import cv2
import numpy as np
from typing import List, Tuple

class MarkerTracker:
    """
    Identifies red markers, cleans noise, and calculates centroids for all found markers.
    """
    def __init__(self, lower_hsv1, upper_hsv1, lower_hsv2, upper_hsv2, min_area=50, max_area=50):
        self.min_area = min_area
        self.max_area = max_area 
        self.lower_hsv1 = lower_hsv1
        self.upper_hsv1 = upper_hsv1
        self.lower_hsv2 = lower_hsv2
        self.upper_hsv2 = upper_hsv2
        self.min_area = min_area # Threshold to reject small noise contours
        
        # Define a kernel for morphological operations. 
        # A 5x5 square is a good starting point. Increase size for larger noise.
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        # 1. HSV Thresholding with two ranges
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_hsv1, self.upper_hsv1)
        mask2 = cv2.inRange(hsv, self.lower_hsv2, self.upper_hsv2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 2. Morphological "Opening" to remove noise
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        
        # 3. Find centroids of all valid markers
        centroids = self._find_marker_centroids(mask_clean)
        return mask_clean, centroids

    def _find_marker_centroids(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Finds contours and filters them by Area AND Shape.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found_centroids = []
        for c in contours:
            area = cv2.contourArea(c)
            
            # Get bounding box to check shape
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            
            # --- FILTERING LOGIC ---
            
            # 1. Size Filter: Keep only objects within a reasonable size range
            # Adjust these values based on your specific video resolution!
            # The 'big noise' on the right will be > max_area
            if area < self.min_area or area > self.max_area:
                continue

            # 2. Shape Filter: Markers are squares, so aspect ratio should be close to 1.
            # Allow for some perspective distortion (e.g., 0.5 to 2.0)
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # --- END FILTERING ---

            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                found_centroids.append((cX, cY))
        
        return found_centroids