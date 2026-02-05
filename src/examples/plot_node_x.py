import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from scipy import signal

# Import your framework modules
from data_processing.trajectory_analyzer import TrajectoryAnalyzer
from data_processing.sensor_loader import SensorLoader

def get_video_start_timestamp(xml_path):
    """Parses Sony XML for CreationDate -> Unix Timestamp."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for elem in root.iter():
            if 'CreationDate' in elem.tag:
                dt = datetime.fromisoformat(elem.attrib['value'])
                return dt.timestamp()
    except Exception as e:
        print(f"XML Error: {e}")
    return 0.0

def synchronize_signals(ref_time, ref_data, target_time, target_data):
    """
    Uses Cross-Correlation to find the time shift that best aligns 
    target_data to ref_data.
    """
    print("Running systematic synchronization...")
    
    # 1. Define a common time grid (Resample to 100Hz for alignment precision)
    t_start = max(ref_time.min(), target_time.min())
    t_end = min(ref_time.max(), target_time.max())
    
    # Handle non-overlapping case safely
    if t_start >= t_end:
        t_start = min(ref_time.min(), target_time.min())
        t_end = max(ref_time.max(), target_time.max())

    dt = 0.001 # 1ms steps
    t_common = np.arange(t_start, t_end, dt)
    
    # 2. Interpolate both signals onto the common grid
    # Subtract mean (center at 0) so correlation works on SHAPE, not offset
    ref_interp = np.interp(t_common, ref_time, ref_data)
    ref_interp -= np.mean(ref_interp)
    
    target_interp = np.interp(t_common, target_time, target_data)
    target_interp -= np.mean(target_interp)
    
    # 3. Cross-Correlate
    correlation = signal.correlate(ref_interp, target_interp, mode='full')
    lags = signal.correlation_lags(len(ref_interp), len(target_interp), mode='full')
    
    # 4. Find peak lag
    best_lag_idx = np.argmax(correlation)
    time_shift = lags[best_lag_idx] * dt
    
    print(f"  -> Calculated Offset: {time_shift:.4f} seconds")
    return time_shift

def main():
    # --- 1. SETUP ---
    current_script_dir = Path(__file__).parent.resolve()
    data_dir = current_script_dir.parent.parent / "data"
    
    video_path = data_dir / "experiment_data" / "spring_mass_data.npz"
    sensor_path = data_dir / "vibrometer_data" / "symmetric_spring_setup" / "spring-mass-2D-3by3_amp=1_2026-02-04.csv"
    xml_path = data_dir / "camera_data" / "C0989M01.XML"

    # --- 2. PREPARE VIDEO (Target to be synced) ---
    print("Loading Video Data...")
    video_analyzer = TrajectoryAnalyzer(video_path, fps=29.97)
    vid_rel_time, vid_disp = video_analyzer.get_displacement(node_idx=0, axis_idx=0)
    
    # Get rough absolute time from XML
    xml_start_ts = get_video_start_timestamp(xml_path)
    vid_epoch_time = vid_rel_time + xml_start_ts
    vid_signal = -vid_disp  # Invert to match sensor direction if needed

    # --- 3. PREPARE SENSOR (Reference) ---
    print("Loading Sensor Data...")
    sensor_loader = SensorLoader(sensor_path)
    if hasattr(sensor_loader, 'df'):
        df = sensor_loader.df
    else:
        df = pd.read_csv(sensor_path, comment='#')

    # Get Epoch Time
    epoch_col = next((c for c in df.columns if 'Epoch' in c), None)
    sens_epoch_time = pd.to_numeric(df[epoch_col], errors='coerce').values / 1000.0
    
    # Get Data (Zeroed)
    _, sens_raw = sensor_loader.get_data()
    sens_signal = sens_raw - sens_raw[0]

    # --- 4. SYSTEMATIC SYNCHRONIZATION ---
    # Calculate offset
    offset = synchronize_signals(sens_epoch_time, sens_signal, vid_epoch_time, vid_signal)
    
    # Apply offset to Video
    vid_epoch_synced = vid_epoch_time + offset

    # --- 5. PLOT (Separate Subplots) ---
    # sharex=True locks the zoom so you can inspect alignment vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Video (Synced)
    ax1.plot(vid_epoch_synced, vid_signal, color='blue', linewidth=1.5, label='Video (Synced)')
    ax1.set_title(f"Video Data (Time Shifted by {offset:.3f}s)", fontsize=12)
    ax1.set_ylabel("Displacement (pixels)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Plot 2: Sensor (Reference)
    ax2.plot(sens_epoch_time, sens_signal, color='red', linewidth=1.5, label='Sensor (Reference)')
    ax2.set_title("Vibrometer Sensor Data (Original Epoch)", fontsize=12)
    ax2.set_ylabel("Displacement (mm)")
    ax2.set_xlabel("Epoch Time (seconds)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Formatting: Show plain numbers (no scientific notation)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax2.ticklabel_format(useOffset=False, style='plain')
    
    # Optional: Use a formatter to limit decimal places on the large Epoch numbers
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()