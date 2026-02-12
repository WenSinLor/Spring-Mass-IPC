import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure the package root is in the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Import your modules
from processing.tracking import MarkerTracker, VideoProcessor
from data_io.writer import DataWriter
from data_io.loader import SensorLoader


def main():
    # --- CONFIGURATION ---
    current_script_dir = Path(__file__).parent.resolve()
    DATA_DIR = current_script_dir.parent.parent / "data"
    VIDEO_FILE = DATA_DIR / "camera_data" / "topology_0" / "C1050.MP4"
    XML_FILE = DATA_DIR / "camera_data" / "topology_0" / "C1050M01.XML"
    SENSOR_FILE = DATA_DIR / "vibrometer_data" / "topology_0" / "spring-mass-2D-3by3_amp=1_2026-02-12.csv"
    
    EXPERIMENT_FILE = DATA_DIR / "experiment_data" / "topology_0" / "tracking_data.h5"
    EXPERIMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    PLOT_SVG = DATA_DIR / "experiment_data" / "topology_0" / "time_alignment_displacement.svg"

    LOWER_RED1, UPPER_RED1 = np.array([0, 120, 70]), np.array([10, 255, 255])
    LOWER_RED2, UPPER_RED2 = np.array([170, 120, 70]), np.array([180, 255, 255])

    # ==========================================
    # STAGE 1: VIDEO PROCESSING
    # ==========================================
    tracker = MarkerTracker(LOWER_RED1, UPPER_RED1, LOWER_RED2, UPPER_RED2, min_area=10000, max_area=20000)
    video_proc = VideoProcessor(VIDEO_FILE, XML_FILE, tracker)
    data_writer = DataWriter(output_dir=DATA_DIR / "experiment_data" / "topology_0")

    # Run with 4-window visualization
    trajectories, frame_indices = video_proc.process_video(visualize=False)

    # Rough Timing
    xml_start_ts = video_proc.start_timestamp
    print(xml_start_ts, flush=True)
    fps = video_proc.fps
    video_time = xml_start_ts + (frame_indices / fps)
    print(video_time, flush=True)

    # ==========================================
    # STAGE 2: Load Data
    # ==========================================
    # Load Sensor (Now using the robust loader)
    sensor_loader = SensorLoader(SENSOR_FILE)
    t_sensor, x_sensor = sensor_loader.get_data()

    # Prepare Video Signal (Node 0, X-Axis, Inverted)
    vid_signal_x = trajectories[:, 0, 0]
    vid_signal_x = -(vid_signal_x - vid_signal_x[0]) 

    # ==========================================
    # STAGE 3: VISUALIZATION (3 Rows)
    # ==========================================
    print("Generating 3-Row Alignment Plot...")
    
    # Create 3 subplots sharing the X-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Row 1: Sensor Only
    ax1.plot(t_sensor, x_sensor, color='tab:red', label='Sensor Data')
    ax1.set_ylabel('Sensor (mm)', color='tab:red')
    ax1.set_title("1. Sensor Data")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Row 2: Camera Only
    ax2.plot(video_time, vid_signal_x, color='tab:blue', label='Camera Data')
    ax2.set_ylabel('Camera (pixels)', color='tab:blue')
    ax2.set_title("2. Camera Data")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Row 3: Overlay (Dual Axis)
    ax3.set_title(f"3. Overlay Alignment")
    ax3.set_xlabel('Epoch Time (s)')
    ax3.grid(True, alpha=0.3)

    # Left Axis (Sensor)
    ax3.set_ylabel('Sensor (mm)', color='tab:red')
    ax3.plot(t_sensor, x_sensor, color='tab:red', alpha=0.6, label='Sensor', linewidth=1.5)
    ax3.tick_params(axis='y', labelcolor='tab:red')

    # Right Axis (Camera)
    ax3_right = ax3.twinx()
    ax3_right.set_ylabel('Camera (pixels)', color='tab:blue')
    ax3_right.plot(video_time, vid_signal_x, color='tab:blue', alpha=0.6, label='Camera', linewidth=1.5)
    ax3_right.tick_params(axis='y', labelcolor='tab:blue')

    plt.tight_layout()
    plt.savefig(PLOT_SVG, format='svg')
    plt.show()
    print(f"Saved plot to {PLOT_SVG}")

    # ==========================================
    # STAGE 4: FINAL SAVE
    # ==========================================
    final_meta = {
        'fps': fps,
        'xml_start_time': xml_start_ts,
        'source_video': VIDEO_FILE.name,
        'source_sensor': SENSOR_FILE.name
    }
    data_writer.save_to_h5(EXPERIMENT_FILE, trajectories, video_time, final_meta)
    print("\nWorkflow Complete.")

if __name__ == "__main__":
    main()