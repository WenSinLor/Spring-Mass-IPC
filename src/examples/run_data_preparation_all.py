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
from processing.synchronization import TimeSynchronizer

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
    # --- CONFIGURATION ---
    current_script_dir = Path(__file__).parent.resolve()
    DATA_DIR = current_script_dir.parent.parent / "data"
    
    # Input Directories
    CAMERA_DATA_DIR = DATA_DIR / "camera_data" / "topology_1"
    SENSOR_DATA_DIR = DATA_DIR / "vibrometer_data" / "topology_1"
    
    # Base Output Directory
    EXPERIMENT_DATA_DIR = DATA_DIR / "experiment_data" / "topology_1"

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
        SENSOR_FILE = SENSOR_DATA_DIR / sensor_file_name

        # --- DYNAMIC FOLDER CREATION ---
        # 1. Extract Amplitude Tag (e.g., "amp=1")
        amp_tag = extract_amplitude(sensor_file_name)
        
        # 2. Create Output Folder: .../experiment_data/topology_0/amp=1/
        output_folder = EXPERIMENT_DATA_DIR / amp_tag
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"   Target Folder: {output_folder}")

        # 3. Define Output File Names (Standardized)
        # We can use generic names now because they are in separate folders
        raw_h5_name = "raw_tracking_data.h5"
        final_h5_name = "calibrated_tracking_data.h5"
        plot_name = "time_alignment_displacement.svg"
        
        plot_path = output_folder / plot_name

        # Red Color Ranges (HSV)
        LOWER_RED1, UPPER_RED1 = np.array([0, 120, 70]), np.array([10, 255, 255])
        LOWER_RED2, UPPER_RED2 = np.array([170, 120, 70]), np.array([180, 255, 255])

        # ==========================================
        # STAGE 1: VIDEO PROCESSING
        # ==========================================
        tracker = MarkerTracker(LOWER_RED1, UPPER_RED1, LOWER_RED2, UPPER_RED2, min_area=10000, max_area=20000)
        video_proc = VideoProcessor(VIDEO_FILE, XML_FILE, tracker)
        
        # Initialize DataWriter with the specific amplitude folder
        data_writer = DataWriter(output_dir=output_folder)

        # Run Tracking
        raw_trajectories, frame_indices = video_proc.process_video(visualize=False)

        # Rough Timing
        xml_start_ts = video_proc.start_timestamp
        fps = video_proc.fps
        rough_video_time = xml_start_ts + (frame_indices / fps)

        # Save Raw Backup
        raw_meta = {'fps': fps, 'xml_start_time': xml_start_ts, 'type': 'raw', 'original_video': video_file_name}
        data_writer.save_to_h5(raw_h5_name, raw_trajectories, rough_video_time, raw_meta)

        # ==========================================
        # STAGE 2: CALIBRATION
        # ==========================================
        # Load Sensor
        sensor_loader = SensorLoader(SENSOR_FILE)
        t_sensor, y_sensor = sensor_loader.get_data()

        # Prepare Video Signal (Node 0, X-Axis, Inverted)
        vid_signal_x = raw_trajectories[:, 0, 0]
        vid_signal_x = -(vid_signal_x - vid_signal_x[0]) 

        # Calculate & Apply Offset
        offset = TimeSynchronizer.calculate_offset(t_sensor, y_sensor, rough_video_time, vid_signal_x)
        calibrated_video_time = rough_video_time + offset

        # ==========================================
        # STAGE 3: VISUALIZATION
        # ==========================================
        # Create 3 subplots sharing the X-axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Row 1: Sensor Only
        ax1.plot(t_sensor, y_sensor, color='tab:red', label='Sensor Data')
        ax1.set_ylabel('Sensor (mm)', color='tab:red')
        ax1.set_title(f"1. Sensor Data ({amp_tag})")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # Row 2: Camera Only
        ax2.plot(calibrated_video_time, vid_signal_x, color='tab:blue', label='Camera Data')
        ax2.set_ylabel('Camera (pixels)', color='tab:blue')
        ax2.set_title("2. Camera Data")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        # Row 3: Overlay (Dual Axis)
        ax3.set_title(f"3. Overlay Alignment (Offset: {offset:.4f}s)")
        ax3.set_xlabel('Epoch Time (s)')
        ax3.grid(True, alpha=0.3)

        ax3.set_ylabel('Sensor (mm)', color='tab:red')
        ax3.plot(t_sensor, y_sensor, color='tab:red', alpha=0.6, label='Sensor', linewidth=1.5)
        ax3.tick_params(axis='y', labelcolor='tab:red')

        ax3_right = ax3.twinx()
        ax3_right.set_ylabel('Camera (pixels)', color='tab:blue')
        ax3_right.plot(calibrated_video_time, vid_signal_x, color='tab:blue', alpha=0.6, label='Camera', linewidth=1.5)
        ax3_right.tick_params(axis='y', labelcolor='tab:blue')

        plt.tight_layout()
        plt.savefig(plot_path, format='svg')
        plt.close()
        print(f"   Saved plot: {plot_name}")

        # ==========================================
        # STAGE 4: FINAL SAVE
        # ==========================================
        final_meta = {
            'fps': fps,
            'xml_start_time': xml_start_ts,
            'sync_offset_applied': offset,
            'source_video': video_file_name,
            'source_sensor': sensor_file_name, # Critical for the next step (slicing)
            'amplitude_group': amp_tag,
            'type': 'calibrated'
        }
        data_writer.save_to_h5(final_h5_name, raw_trajectories, calibrated_video_time, final_meta)
        print("   -> Pair Complete.")

    print("\nAll pairs processed successfully.")

if __name__ == "__main__":
    main()