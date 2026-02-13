import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import re

# Ensure the package root is in the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import your modules
from processing.tracking import MarkerTracker, VideoProcessor
from data_io.writer import DataWriter
from data_io.loader import SensorLoader


def extract_amplitude(filename):
    """
    Extracts 'amp=X' from a filename string. 
    Example: 'spring-mass-2D-3by3_amp=1_2026-02-04.csv' -> 'amp=1'
    """
    match = re.search(r"(amp=[\d\.]+)", filename)
    if match:
        return match.group(1)
    return "unknown_amp"

def main():
    # --- GRID CONFIGURATION ---
    GRID_ROWS = 4
    GRID_COLS = 4
    # --------------------------

    # --- PATH CONFIGURATION ---
    current_script_dir = Path(__file__).parent.resolve()
    DATA_DIR = current_script_dir.parent.parent / "data"
    
    # Input Directories
    CAMERA_DATA_DIR = DATA_DIR / "camera_data" / "topology_5"
    SENSOR_DATA_DIR = DATA_DIR / "vibrometer_data" / "topology_5"
    
    # Base Output Directory
    EXPERIMENT_DATA_DIR = DATA_DIR / "experiment_data" / "topology_5"

    # --- 1. SCAN AND PAIR FILES ---
    # Get sorted lists
    camera_files = sorted([f for f in os.listdir(CAMERA_DATA_DIR) if f.endswith('.MP4')])
    sensor_files = sorted([f for f in os.listdir(SENSOR_DATA_DIR) if f.endswith('.csv')], reverse=True)

    # Pair files (Assumes 1-to-1 matching based on sort order)
    file_pairs = list(zip(camera_files, sensor_files))

    print(f"Found {len(file_pairs)} pairs to process in {CAMERA_DATA_DIR.name}:")
    for v, s in file_pairs:
        print(f"  {v} <-> {s}")
    print("-" * 40)

    # --- 2. PROCESSING LOOP ---
    for video_file_name, sensor_file_name in file_pairs:
        print(f"\nProcessing: {video_file_name}")
        
        # Define Input Paths
        VIDEO_FILE = CAMERA_DATA_DIR / video_file_name
        # XML usually has same stem but ends in M01.XML.
        xml_name = f"{VIDEO_FILE.stem.split('-')[0]}M01.XML"
        XML_FILE = CAMERA_DATA_DIR / xml_name

        # --- DYNAMIC FOLDER CREATION ---
        # 1. Extract Amplitude Tag (e.g., "amp=1")
        amp_tag = extract_amplitude(sensor_file_name)
        
        # 2. Create Output Folder: .../experiment_data/topology_0/amp=1/
        output_folder = EXPERIMENT_DATA_DIR / amp_tag
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"   Target Folder: {output_folder}")

        # 3. Define Output File Names (Standardized)
        # We can use generic names now because they are in separate folders
        output_filename = "tracking_data.h5"
        
        # Red Color Ranges (HSV)
        LOWER_RED1, UPPER_RED1 = np.array([0, 120, 70]), np.array([10, 255, 255])
        LOWER_RED2, UPPER_RED2 = np.array([170, 120, 70]), np.array([180, 255, 255])

        # ==========================================
        # STAGE 1: VIDEO PROCESSING
        # ==========================================
        print(f"   Expecting a {GRID_ROWS}x{GRID_COLS} grid.")
        tracker = MarkerTracker(LOWER_RED1, UPPER_RED1, LOWER_RED2, UPPER_RED2, min_area=10000, max_area=20000)
        video_proc = VideoProcessor(VIDEO_FILE, XML_FILE, tracker)
        
        # Initialize DataWriter with the specific amplitude folder
        data_writer = DataWriter(output_dir=output_folder)

        # Run Tracking
        raw_trajectories, frame_indices = video_proc.process_video(
            grid_rows=GRID_ROWS,
            grid_cols=GRID_COLS,
            visualize=False
        )

        # Rough Timing
        xml_start_ts = video_proc.start_timestamp
        fps = video_proc.fps
        video_time = xml_start_ts + (frame_indices / fps)

        # ==========================================
        # STAGE 2: FINAL SAVE
        # ==========================================
        final_meta = {
            'fps': fps,
            'xml_start_time': xml_start_ts,
            'source_video': video_file_name,
            'source_sensor': sensor_file_name,
            'amplitude_group': amp_tag
        }
        data_writer.save_to_h5(output_filename, raw_trajectories, video_time, final_meta)
        print("   -> Pair Complete.")

    print("\nAll pairs processed successfully.")

if __name__ == "__main__":
    main()
