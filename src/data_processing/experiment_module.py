import cv2
import numpy as np
import pandas as pd
import h5py
import xml.etree.ElementTree as ET
from scipy import signal
from datetime import datetime
from pathlib import Path

# --- 1. VIDEO PROCESSING MODULE ---
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

# --- 2. DATA MANAGEMENT MODULE ---
class DataManager:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_sensor_csv(self, csv_path):
        """Robust CSV loader that handles # in headers."""
        print(f"Loading Sensor Data: {Path(csv_path).name}")
        
        # 1. Find Header Line (contains "Epoch")
        header_idx = 0
        with open(csv_path, 'r') as f:
            for i, line in enumerate(f):
                if "Epoch" in line:
                    header_idx = i
                    break
        
        # 2. Read CSV (Do NOT use comment='#')
        df = pd.read_csv(csv_path, header=header_idx)
        
        # 3. Cleanup Column Names
        df.columns = [c.replace('#', '').replace('"', '').strip() for c in df.columns]
        
        # 4. Extract Data
        epoch_col = next((c for c in df.columns if 'Epoch' in c), None)
        if not epoch_col: raise ValueError(f"No 'Epoch' column found. Columns: {df.columns}")
        
        t_sensor = pd.to_numeric(df[epoch_col], errors='coerce').values / 1000.0
        
        # Assume signal is column 3 (Distance1)
        signal_col = df.columns[3] 
        y_sensor = pd.to_numeric(df[signal_col], errors='coerce').values
        y_sensor = y_sensor - y_sensor[0] # Zero it
        
        return t_sensor, y_sensor

    def save_to_h5(self, filename, trajectories, time_array, metadata=None):
        filepath = self.output_dir / filename
        print(f"Saving to {filepath.name}...")
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset('trajectories', data=trajectories, compression="gzip")
            hf.create_dataset('time', data=time_array, compression="gzip")
            if metadata:
                for k, v in metadata.items(): hf.attrs[k] = v
        print("Save Successful.")

# --- 3. SYNCHRONIZATION MODULE ---
class TimeSynchronizer:
    @staticmethod
    def calculate_offset(ref_t, ref_y, target_t, target_y):
        print("Synchronizing Signals...")
        t_start, t_end = max(ref_t.min(), target_t.min()), min(ref_t.max(), target_t.max())
        if t_start >= t_end:
            t_start, t_end = min(ref_t.min(), target_t.min()), max(ref_t.max(), target_t.max())
        
        dt = 0.001
        t_common = np.arange(t_start, t_end, dt)
        
        ref_interp = np.interp(t_common, ref_t, ref_y) - np.mean(ref_y)
        tgt_interp = np.interp(t_common, target_t, target_y) - np.mean(target_y)
        
        lags = signal.correlation_lags(len(ref_interp), len(tgt_interp), mode='full')
        corr = signal.correlate(ref_interp, tgt_interp, mode='full')
        
        offset = lags[np.argmax(corr)] * dt
        print(f"  -> Calculated Offset: {offset:.4f} seconds")
        return offset