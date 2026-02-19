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
    Slices a calibrated H5 file for VIRTUAL NODE MULTIPLEXING.
    Finds one long sequence of 899 steps (x 5 frames).
    Reshapes the extra frames into the 'nodes' dimension.
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
            
            # Reconstruct Displacement for Trigger
            raw_x = full_trajectories[:, 0, 0]
            displacement_data = -(raw_x - raw_x[0])

    except Exception as e:
        print(f"  [Error] Failed to read {input_file.name}: {e}")
        return

    # ==========================================
    # 2. TOPOLOGY INDICES (For Bar Calculation)
    # ==========================================
    # Ensure 3D
    if full_trajectories.ndim == 3 and full_trajectories.shape[2] == 2:
        temp = np.zeros((full_trajectories.shape[0], full_trajectories.shape[1], 3), dtype=np.float32)
        temp[:, :, :2] = full_trajectories
        full_trajectories = temp

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
    # 3. MULTIPLEXING SETUP
    # ==========================================
    virtual_nodes = 3
    num_steps = 899
    
    # We need exactly this many frames to form the matrix
    total_frames_needed = num_steps * virtual_nodes # 4495 frames
    
    # Dynamic Threshold
    trigger_threshold = np.max(np.abs(displacement_data)) * 0.1
    print(f"   Multiplex Threshold: {trigger_threshold:.2f} pixels")
    print(f"   Target: {num_steps} steps x {virtual_nodes} frames = {total_frames_needed} total frames")

    # ==========================================
    # 4. FIND START & SLICE
    # ==========================================
    # Find where the shaker starts moving
    potential_triggers = np.where(np.abs(displacement_data) > trigger_threshold)[0]
    
    if len(potential_triggers) == 0:
        print("   [Stop] No movement detected.")
        return
        
    start_idx = potential_triggers[0]
    end_idx = start_idx + total_frames_needed

    if end_idx > len(full_trajectories):
        print(f"   [Error] Not enough data. Needed {total_frames_needed}, found {len(full_trajectories)-start_idx}")
        return

    # --- A. Slice Raw Data (High Speed) ---
    # We grab the full 4495 frames first
    pos_raw = full_trajectories[start_idx:end_idx]      # Shape: (4495, 16, 3)
    time_raw = aligned_video_time[start_idx:end_idx]    # Shape: (4495,)
    act_raw = displacement_data[start_idx:end_idx]      # Shape: (4495,)

    # --- B. Calculate Bar Lengths (On Raw Data) ---
    # We calculate bars frame-by-frame BEFORE flattening
    p0 = pos_raw[:, bar_indices[:, 0], :]
    p1 = pos_raw[:, bar_indices[:, 1], :]
    lengths_raw = np.sqrt(np.sum((p1 - p0)**2, axis=2)) # Shape: (4495, 24)

    # ==========================================
    # 5. RESHAPE FOR VIRTUAL NODES
    # ==========================================
    # This is the key step: Flatten 5 frames into the Feature Dimension
    
    # 1. Reshape Positions: (899, 80, 3)
    # Logic: Group by 5 -> (899, 5, 16, 3) -> Flatten 5 & 16 -> (899, 80, 3)
    pos_multiplexed = pos_raw.reshape(num_steps, virtual_nodes, 16, 3).reshape(num_steps, -1, 3)
    
    # 2. Reshape Bar Lengths: (899, 120)
    # Logic: Group by 5 -> (899, 5, 24) -> Flatten 5 & 24 -> (899, 120)
    lengths_multiplexed = lengths_raw.reshape(num_steps, virtual_nodes, 24).reshape(num_steps, -1)
    
    # 3. Decimate Time: (899,)
    # We take the timestamp of the FIRST frame of every window
    time_multiplexed = time_raw.reshape(num_steps, virtual_nodes)[:, 0]
    time_relative = time_multiplexed - time_multiplexed[0]
    
    # 4. Decimate Actuation: (899, 1)
    # We take the actuation value of the FIRST frame (or you could mean)
    act_multiplexed = act_raw.reshape(num_steps, virtual_nodes)[:, 0].reshape(-1, 1)

    # ==========================================
    # 6. SAVE TO EXPERIMENT H5
    # ==========================================
    # We save as "sample_0" because it is one continuous experiment
    sample_dir = amp_folder_path / "sample_0"
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / "experiment.h5"
    
    try:
        with h5py.File(output_path, 'w') as f_out:
            ts = f_out.create_group('time_series')
            
            # --- The Main Data ---
            # Note: positions is now (899, 80, 3)
            # Note: lengths is now (899, 120)
            ts.create_dataset('time', data=time_relative)
            ts.create_group('nodes').create_dataset('positions', data=pos_multiplexed)
            ts.create_group('elements/bars').create_dataset('lengths', data=lengths_multiplexed)
            
            # --- Actuation Signals ---
            if act_multiplexed is not None:
                act = ts.create_group('actuation_signals')
                # Save same actuation for the key nodes
                for node in [0, 3, 12, 15]: 
                    act.create_dataset(str(node), data=act_multiplexed)

            # --- Metadata ---
            for k, v in metadata.items(): f_out.attrs[k] = v
            f_out.attrs['slice_index'] = 0
            f_out.attrs['amplitude_group'] = amp_folder_name
            # Crucial metadata for your trainer to understand the shape
            f_out.attrs['virtual_nodes_count'] = virtual_nodes
            f_out.attrs['original_nodes'] = 16
            f_out.attrs['is_multiplexed'] = True

        print(f"   -> Saved: {sample_dir.name}/experiment.h5")
        print(f"      Shape Nodes: {pos_multiplexed.shape} (Expected: 899, 48, 3)")
        print(f"      Shape Bars:  {lengths_multiplexed.shape} (Expected: 899, 72)")
        
    except Exception as e:
        print(f"   [Error] Failed to write {output_path}: {e}")

    print("   Processing complete.\n")


def main():
    print("Starting script: Multiplexed Slice and Organize.")
    
    current_script_dir = Path(__file__).parent.resolve()
    EXPERIMENT_DATA_DIR = current_script_dir.parent.parent / "data" / "experiment_data" / "topology_5_prestress_narma_multiplex_2"
    
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