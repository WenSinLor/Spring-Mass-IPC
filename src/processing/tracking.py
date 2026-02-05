import cv2
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
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

class VideoLoader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

    def stream_frames(self):
        """Generator that yields frames one by one, ensuring it stops at EOF."""
        if not self.cap.isOpened():
            return

        # robustly get total frames to prevent infinite looping
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while self.cap.isOpened():
            # Safety Check 1: logical limit
            if current_frame >= total_frames:
                break
                
            ret, frame = self.cap.read()
            
            # Safety Check 2: OpenCV signal
            if not ret:
                break
            
            yield frame
            current_frame += 1

class VideoProcessor:
    def __init__(self, video_path, xml_path, tracker):
        self.video_path = Path(video_path)
        self.xml_path = Path(xml_path)
        self.tracker = tracker
        self.fps = 29.97
        self.start_timestamp = 0.0

    def get_start_timestamp(self):
        try:
            tree = ET.parse(self.xml_path)
            for elem in tree.getroot().iter():
                if 'CreationDate' in elem.tag:
                    dt = datetime.fromisoformat(elem.attrib['value'])
                    return dt.timestamp()
        except Exception as e:
            print(f"Warning: XML Timestamp error ({e}). Defaulting to 0.0")
        return 0.0

    def process_video(self, expected_markers=9, visualize=False):
        print(f"Processing Video: {self.video_path.name}")
        self.start_timestamp = self.get_start_timestamp()
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened(): raise IOError(f"Cannot open video: {self.video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
        raw_trajectories, frame_indices = [], []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        window_name = f"2x2 View: {self.video_path.name}"
        if visualize:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                clean_frame = frame.copy()
                frame_with_dots = frame.copy()
                mask, centroids = self.tracker.process_frame(clean_frame)
                sorted_centroids = sorted(centroids, key=lambda p: (p[1] // 50, p[0]))

                if len(sorted_centroids) == expected_markers:
                    block = [[c[0], c[1], 0] for c in sorted_centroids]
                    raw_trajectories.append(block)
                    frame_indices.append(frame_count)
                
                # --- VISUALIZATION ---
                if visualize:
                    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    black_with_dots = np.zeros_like(clean_frame)

                    for i, centroid in enumerate(sorted_centroids):
                        cv2.circle(frame_with_dots, centroid, 15, (0, 255, 0), -1)
                        cv2.circle(black_with_dots, centroid, 15, (0, 255, 0), -1)
                        # Text on black background
                        text_pos = (centroid[0] - 15, centroid[1] - 20)
                        cv2.putText(black_with_dots, str(i), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4)
                        cv2.putText(black_with_dots, str(i), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
                        # Text on video frame
                        cv2.putText(frame_with_dots, str(i), (centroid[0]-10, centroid[1]-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    grid = np.vstack([np.hstack([clean_frame, mask_bgr]), 
                                      np.hstack([frame_with_dots, black_with_dots])])
                    cv2.imshow(window_name, grid)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break

                if frame_count % 100 == 0: print(f"  Frame {frame_count}/{total_frames}...", end='\r')
                frame_count += 1
        finally:
            cap.release()
            if visualize: cv2.destroyAllWindows()

        print(f"\nTracking Complete. Valid Frames: {len(raw_trajectories)}")
        return np.array(raw_trajectories), np.array(frame_indices)
