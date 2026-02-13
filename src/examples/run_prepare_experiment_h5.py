import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Try to import SensorLoader
try:
    from data_io.loader import SensorLoader
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_io.loader import SensorLoader

def prepare_experiment_slices(input_file: Path):
    """
    Slices a calibrated H5 file into samples.
    Uses video displacement thresholding to find start times,
    then grabs the corresponding sensor data by nearest-timestamp lookup.
    """
    print(f"-> Processing: {input_file.parent.name}/{input_file.name}")
    
    # ==========================================
    # 1. SETUP & LOAD VIDEO DATA
    # ==========================================
    amp_folder_path = input_file.parent
    amp_folder_name = amp_folder_path.name
    topology_folder_path = amp_folder_path.parent
    topology_name = topology_folder_path.name

    try:
        with h5py.File(input_file, 'r') as f_in:
            full_trajectories = f_in['trajectories'][:] 
            aligned_video_time = f_in['time'][:] 
            metadata = dict(f_in.attrs)
            fps = f_in.attrs.get('fps', 29.97)
            source_sensor_name = metadata.get('source_sensor')
            
            # Reconstruct Displacement Signal for Slicing Trigger
            # (Node 0, X-Axis, Inverted)
            raw_x = full_trajectories[:, 0, 0]
            displacement_data = -(raw_x - raw_x[0])

    except Exception as e:
        print(f"  [Error] Failed to read {input_file.name}: {e}")
        return

    # ==========================================
    # 2. LOAD SENSOR DATA (Full)
    # ==========================================
    t_sensor_full, y_sensor_full = None, None
    
    if source_sensor_name:
        data_root = input_file.parents[3] 
        sensor_path = data_root / "vibrometer_data" / topology_name / source_sensor_name

        if sensor_path.exists():
            try:
                loader = SensorLoader(sensor_path)
                t_sensor_full, y_sensor_full = loader.get_data()
                
                # Convert mm to meters immediately
                y_sensor_full = y_sensor_full / 1000.0
            except Exception as e:
                print(f"  [Warning] Sensor error: {e}")
        else:
            print(f"  [Warning] Missing CSV: {sensor_path}")

    # ==========================================
    # 3. TOPOLOGY SETUP
    # ==========================================
    # Ensure 3D
    if full_trajectories.ndim == 3 and full_trajectories.shape[2] == 2:
        temp = np.zeros((full_trajectories.shape[0], full_trajectories.shape[1], 3), dtype=np.float32)
        temp[:, :, :2] = full_trajectories
        full_trajectories = temp

    # bar_indices = np.array([
    #     [0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], # Horizontal
    #     [0, 3], [3, 6], [1, 4], [4, 7], [2, 5], [5, 8]  # Vertical
    # ])

    bar_indices = np.array([
        # Horizontal
        [0, 1], [1, 2], [2, 3],
        [4, 5], [5, 6], [6, 7],
        [8, 9], [9, 10], [10, 11],
        [12, 13], [13, 14], [14, 15],
        # Vertical
        [0, 4], [4, 8], [8, 12],
        [1, 5], [5, 9], [9, 13],
        [2, 6], [6, 10], [10, 14],
        [3, 7], [7, 11], [11, 15]
    ])

    # ==========================================
    # 4. SLICING LOOP (Threshold Logic)
    # ==========================================
    duration = 30 # seconds
    num_samples = 5
    points_per_sample = int(fps * duration)
    
    # Dynamic Threshold
    trigger_threshold = np.max(np.abs(displacement_data)) * 0.1
    print(f"   Slicing Threshold: {trigger_threshold:.2f} pixels")
    
    current_search_idx = 0

    for i in range(num_samples):
        # A. Find Video Trigger
        remaining_data = np.abs(displacement_data[current_search_idx:])
        potential_triggers = np.where(remaining_data > trigger_threshold)[0]
        
        if len(potential_triggers) == 0:
            print(f"   [Stop] No more bursts detected (Sample {i}).")
            break
            
        trigger_rel = potential_triggers[0]
        start_idx = max(0, current_search_idx + trigger_rel - int(0.2 * fps))
        end_idx = start_idx + points_per_sample

        if end_idx > len(full_trajectories):
            print(f"   [Stop] Not enough data for Sample {i}.")
            break

        # B. Slice Video Data
        time_slice = aligned_video_time[start_idx:end_idx] # Absolute Aligned Time
        
        # Make time relative to 0 for simulation
        time_slice_relative = time_slice - time_slice[0] 
        
        pos_slice = full_trajectories[start_idx:end_idx]
        
        # Compute Bars
        p0 = pos_slice[:, bar_indices[:, 0], :]
        p1 = pos_slice[:, bar_indices[:, 1], :]
        lengths_slice = np.sqrt(np.sum((p1 - p0)**2, axis=2))

        # C. Slice Actuation from Video Data
        # Using the displacement of Node 0 as the actuation signal.
        act_slice = displacement_data[start_idx:end_idx]

        # Reshape for H5 (N, 1)
        if act_slice.ndim == 1:
            act_slice = act_slice.reshape(-1, 1)

        # NOTE: This actuation signal is in PIXELS. The original sensor data was in METERS.
        # This may need to be adjusted if physical units are required.

        # --- PLOTTING FOR VERIFICATION ---
        # plt.figure(figsize=(8, 3))
        # plt.plot(time_slice_relative, act_slice, color='darkorange', label='Video Disp. (px)')
        # plt.title(f"Sample {i} Actuation ({amp_folder_name})")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Displacement (px)")
        # plt.grid(True, alpha=0.5)
        # plt.legend()
        # plt.tight_layout()
        # plt.show() # Pop-up window
        # plt.close()

        # D. Save to Experiment H5
        sample_dir = amp_folder_path / f"sample_{i}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_path = sample_dir / "experiment.h5"
        
        try:
            with h5py.File(output_path, 'w') as f_out:
                ts = f_out.create_group('time_series')
                
                ts.create_dataset('time', data=time_slice_relative)
                ts.create_group('nodes').create_dataset('positions', data=pos_slice)
                ts.create_group('elements/bars').create_dataset('lengths', data=lengths_slice)
                
                if act_slice is not None:
                    act = ts.create_group('actuation_signals')
                    # for node in [0, 2, 6, 8]:
                    for node in [0, 3, 12, 15]: 
                        act.create_dataset(str(node), data=act_slice)

                for k, v in metadata.items(): f_out.attrs[k] = v
                f_out.attrs['slice_index'] = i
                f_out.attrs['amplitude_group'] = amp_folder_name

            print(f"   -> Saved: {sample_dir.name}/experiment.h5")
            
        except Exception as e:
            print(f"   [Error] Failed to write {output_path}: {e}")

        # E. Advance Cursor
        current_search_idx = end_idx + int(2.0 * fps)

    print("   Processing complete.\n")


def main():
    print("Starting script: Slice and Organize Experiment Samples.")
    
    current_script_dir = Path(__file__).parent.resolve()
    EXPERIMENT_DATA_DIR = current_script_dir.parent.parent / "data" / "experiment_data" / "topology_5"
    
    if not EXPERIMENT_DATA_DIR.exists():
        print(f"[Error] Directory not found: {EXPERIMENT_DATA_DIR}")
        return
    
    # Recursive Glob
    input_files = sorted(list(EXPERIMENT_DATA_DIR.rglob("tracking_data.h5")))
    
    if not input_files:
        print(f"No 'tracking_data.h5' files found.")
        return

    print(f"Found {len(input_files)} tracking_data.h5 files. Processing...")
    
    for f in input_files:
        prepare_experiment_slices(f)

    print("Script finished.")

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()