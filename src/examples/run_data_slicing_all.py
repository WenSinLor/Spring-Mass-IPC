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
    Processes a single calibrated H5 file to slice it into samples.
    """
    # ==========================================
    # 1. Configuration
    # ==========================================
    DATA_DIR = input_h5_path.parent
    output_base_name = input_h5_path.name.replace('_calibrated_tracking_data.h5', '')
    output_h5_path = DATA_DIR / f'{output_base_name}_calibrated_samples_sliced.h5'
    output_plot_path = DATA_DIR / f'{output_base_name}_calibrated_samples_sliced_plot.svg'

    # Settings
    duration = 30          # seconds per slice
    num_samples = 5        # Number of samples to find

    # ==========================================
    # 2. Load & Preprocess Data
    # ==========================================
    if not input_h5_path.exists():
        print(f"File not found: {input_h5_path}")
        return

    print(f"Loading data from {input_h5_path}...")

    with h5py.File(input_h5_path, 'r') as f:
        # 1. Get FPS (Crucial for correct slicing length)
        fps = f.attrs.get('fps', 29.97) # Default to 29.97 if missing
        print(f"Detected FPS: {fps:.2f}")

        # 2. Load Raw Trajectories [Frames, Markers, Coordinates]
        # Shape is likely (N, 9, 3)
        raw_trajectories = f['trajectories'][:]

        # 3. Reconstruct the Displacement Signal (Same logic as your calibration script)
        # Node 0 (Index 0), X-Axis (Index 0)
        # Logic: vid_signal_x = -(vid_signal_x - vid_signal_x[0])
        raw_x = raw_trajectories[:, 0, 0]
        displacement_data = -(raw_x - raw_x[0])

        print(f"Reconstructed Displacement Signal: {len(displacement_data)} points")

    # Calculate exact points needed per 30s slice based on REAL fps
    points_per_sample = int(fps * duration)
    print(f"Points per {duration}s sample: {points_per_sample}")

    # ==========================================
    # 3. Systematic Trigger & Slice
    # ==========================================
    sliced_data_list = []
    current_search_idx = 0

    # Threshold: Dynamic calculation (10% of max peak) to ensure we catch the start
    trigger_threshold = np.max(np.abs(displacement_data)) * 0.1
    print(f"Auto-calculated trigger threshold: {trigger_threshold:.2f} pixels")

    with h5py.File(output_h5_path, 'w') as hf_out:
        # Save metadata
        hf_out.attrs['fps'] = fps
        hf_out.attrs['duration'] = duration

        for i in range(num_samples):
            remaining_data = np.abs(displacement_data[current_search_idx:])
            
            if len(remaining_data) == 0:
                print(f"Warning: No more data to search for burst #{i+1}.")
                break

            # Find start of burst
            trigger_rel = np.argmax(remaining_data > trigger_threshold)

            # Safety: Check if we actually found something
            if remaining_data[trigger_rel] <= trigger_threshold:
                print(f"Warning: Could not find burst #{i+1} (signal stayed below threshold).")
                break

            start_idx = current_search_idx + trigger_rel
            # Back up slightly (0.5s) to catch the rising edge
            start_idx = max(0, start_idx - int(0.5 * fps))

            end_idx = start_idx + points_per_sample

            # Check bounds
            if end_idx > len(displacement_data):
                print(f"Error: Not enough data left for Burst #{i+1}.")
                break

            # Slice
            segment = displacement_data[start_idx : end_idx]
            sliced_data_list.append(segment)

            # Save
            hf_out.create_dataset(f'sample_{i}', data=segment)
            print(f"  -> Sample {i}: Indices {start_idx}-{end_idx} ({len(segment)} pts)")

            # Move cursor forward (add buffer)
            current_search_idx = end_idx + int(2.0 * fps) # 2 sec buffer to skip any tail noise

    print(f"Saved {len(sliced_data_list)} samples to {output_h5_path}")

    # ==========================================
    # 4. Plotting (Displacement vs Relative Time)
    # ==========================================
    if sliced_data_list:
        count = len(sliced_data_list)

        # Create RELATIVE time vector (0 to 30s)
        t_relative = np.linspace(0, duration, points_per_sample)

        fig, axes = plt.subplots(count, 1, figsize=(10, 2.5 * count), constrained_layout=True)
        if count == 1: axes = [axes]

        for i, data in enumerate(sliced_data_list):
            if len(data) != len(t_relative):
                print(f"Warning: data length ({len(data)}) and time length ({len(t_relative)}) mismatch for sample {i}. Skipping plot.")
                continue

            ax = axes[i]

            # Plot
            ax.plot(t_relative, data, color='#1f77b4', linewidth=1.5)

            # Styling
            ax.set_title(f'Sample {i}', loc='left', fontsize=11, fontweight='bold')
            ax.set_ylabel('Displacement (pixels)', fontsize=9)
            ax.set_xlim(0, duration)
            ax.grid(True, linestyle=':', alpha=0.6)

            # X-Axis Labels (Only on bottom plot)
            if i == count - 1:
                ax.set_xlabel('Time (s)', fontsize=11)
            else:
                ax.set_xticklabels([])

        plt.savefig(output_plot_path, format='svg')
        plt.close(fig)
        print(f"Plot saved to {output_plot_path}")
    print("-" * 40)


def main():
    """
    Finds all calibrated H5 files and processes them.
    """
    current_script_dir = Path(__file__).parent.resolve()
    DATA_DIR = current_script_dir.parent.parent / "data" / "experiment_data" / "topology_0"
    
    files_to_process = [f for f in os.listdir(DATA_DIR) if f.endswith('_calibrated_tracking_data.h5')]
    
    if not files_to_process:
        print("No '*_calibrated_tracking_data.h5' files found to process.")
        return
        
    print(f"Found {len(files_to_process)} files to process.\n")
    
    for file_name in files_to_process:
        process_file(DATA_DIR / file_name)
        
    print("\nAll files processed.")

if __name__ == "__main__":
    main()
