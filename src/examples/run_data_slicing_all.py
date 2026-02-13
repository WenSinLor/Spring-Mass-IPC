import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os

# Ensure the package root is in the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def process_file(input_h5_path):
    """
    Processes a single calibrated H5 file (inside an amp=X folder).
    - Generates a summary plot of the slicing process.
    - Saves individual displacement slices into the sample_X folders.
    """
    # ==========================================
    # 1. Configuration & Paths
    # ==========================================
    # Input is: .../topology_0/amp=X/calibrated_tracking_data.h5
    amp_folder = input_h5_path.parent
    
    print(f"Processing: {amp_folder.name}/{input_h5_path.name}")

    # Output Plot (Summary goes in the Amplitude folder)
    output_plot_path = amp_folder / 'slicing_summary_plot.svg'

    # Settings
    duration = 30          # seconds per slice
    num_samples = 5        # Number of samples to find

    # ==========================================
    # 2. Load & Preprocess Data
    # ==========================================
    if not input_h5_path.exists():
        print(f"File not found: {input_h5_path}")
        return

    with h5py.File(input_h5_path, 'r') as f:
        # 1. Get FPS
        fps = f.attrs.get('fps', 29.97)
        
        # 2. Load Raw Trajectories
        raw_trajectories = f['trajectories'][:]

        # 3. Reconstruct Displacement Signal
        # Node 0, X-Axis
        raw_x = raw_trajectories[:, 0, 0]
        displacement_data = -(raw_x - raw_x[0])

    # Calculate points per slice
    points_per_sample = int(fps * duration)

    # ==========================================
    # 3. Systematic Trigger & Slice
    # ==========================================
    sliced_data_list = []
    current_search_idx = 0

    # Auto-threshold
    trigger_threshold = np.max(np.abs(displacement_data)) * 0.1
    print(f"   Threshold: {trigger_threshold:.2f} pixels")

    for i in range(num_samples):
        remaining_data = np.abs(displacement_data[current_search_idx:])
        
        if len(remaining_data) == 0:
            print(f"   [Stop] No more data for sample {i}.")
            break

        # Find trigger
        potential_triggers = np.where(remaining_data > trigger_threshold)[0]
        
        if len(potential_triggers) == 0:
            print(f"   [Stop] Signal below threshold for sample {i}.")
            break
            
        trigger_rel = potential_triggers[0]
        start_idx = max(0, current_search_idx + trigger_rel - int(0.2 * fps))
        end_idx = start_idx + points_per_sample

        if end_idx > len(displacement_data):
            print(f"   [Stop] Not enough data for sample {i}.")
            break

        # Slice
        segment = displacement_data[start_idx : end_idx]
        sliced_data_list.append(segment)

        # --- SAVE TO HIERARCHY ---
        # Structure: amp=X / sample_i / displacement_slice.h5
        sample_dir = amp_folder / f"sample_{i}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        slice_h5_path = sample_dir / 'displacement_slice.h5'
        
        try:
            with h5py.File(slice_h5_path, 'w') as hf_out:
                hf_out.create_dataset('displacement', data=segment)
                hf_out.attrs['fps'] = fps
                hf_out.attrs['duration'] = duration
                hf_out.attrs['start_frame'] = start_idx
            print(f"   -> Saved: sample_{i}/displacement_slice.h5")
        except Exception as e:
            print(f"   [Error] Could not save H5: {e}")

        # Advance cursor
        current_search_idx = end_idx + int(2.0 * fps)

    # ==========================================
    # 4. Plotting (Summary)
    # ==========================================
    if sliced_data_list:
        count = len(sliced_data_list)
        t_relative = np.linspace(0, duration, points_per_sample)

        fig, axes = plt.subplots(count, 1, figsize=(10, 2.5 * count), constrained_layout=True)
        if count == 1: axes = [axes]

        for i, data in enumerate(sliced_data_list):
            if len(data) != len(t_relative):
                continue

            ax = axes[i]
            ax.plot(t_relative, data, color='#1f77b4', linewidth=1.5)
            ax.set_title(f'Sample {i}', loc='left', fontsize=11, fontweight='bold')
            ax.set_ylabel('Disp. (px)', fontsize=9)
            ax.set_xlim(0, duration)
            ax.grid(True, linestyle=':', alpha=0.6)

            if i == count - 1:
                ax.set_xlabel('Time (s)', fontsize=11)
            else:
                ax.set_xticklabels([])

        plt.savefig(output_plot_path, format='svg')
        plt.close(fig)
        print(f"   -> Saved Plot: {output_plot_path.name}")
    print("-" * 40)


def main():
    """
    Finds all calibrated H5 files in the hierarchy and processes them.
    """
    print("Starting script: Validate Slices (Plots & Displacement Check).")
    
    current_script_dir = Path(__file__).parent.resolve()
    # Path to Topology Root
    DATA_DIR = current_script_dir.parent.parent / "data" / "experiment_data" / "topology_1"
    
    if not DATA_DIR.exists():
        print(f"[Error] Directory not found: {DATA_DIR}")
        return

    # Use rglob to find files inside amp=X folders
    all_files = sorted(list(DATA_DIR.rglob("tracking_data.h5")))
    files_to_process = [f for f in all_files if "amp=" in str(f)]
    
    if not files_to_process:
        print("No 'tracking_data.h5' files found in any amp=X directory.")
        return
        
    print(f"Found {len(files_to_process)} files to process.\n")
    
    for input_path in files_to_process:
        process_file(input_path)
        
    print("\nAll files processed.")

if __name__ == "__main__":
    main()