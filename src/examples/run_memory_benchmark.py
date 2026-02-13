import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# --- Core Library Imports ---
from openprc.analysis.benchmarks.memory_benchmark import MemoryBenchmark
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodePositions
from openprc.reservoir.features.bar_features import BarExtensions
from openprc.reservoir.training.trainer import Trainer
from openprc.reservoir.readout.ridge import Ridge
from openprc.analysis.visualization.time_series import TimeSeriesComparison


def main():
    """
    A pipeline to run the memory benchmark on a given experiment.
    This script will first run the benchmark to calculate all memory capacities,
    then prompt the user to select which readout to train and save permanently.
    """
    
    # 1. Define the Experiment Path
    TOPOLOGY = "topology_5"
    AMPLITUDE = "amp=1"
    SAMPLE = "sample_1"
    
    data_root = src_dir.parent / "data" / "experiment_data"
    experiment_dir = data_root / TOPOLOGY / AMPLITUDE / SAMPLE
    h5_path = experiment_dir / "experiment.h5"

    if not h5_path.exists():
        print(f"[Error] Experiment file not found: {h5_path}")
        return

    print(f"-> Loading Experiment: {AMPLITUDE} / {SAMPLE}")
    
    # 2. Shared Setup
    loader = StateLoader(h5_path)
    features = BarExtensions()
    u_input = loader.get_actuation_signal(actuator_idx=0, dof=0)
    
    print(f"Loaded {loader.total_frames} frames from {h5_path.name}")
    
    # 3. Define Benchmark and its arguments
    benchmark = MemoryBenchmark(group_name="memory_benchmark")
    benchmark_args = {
        "tau_s": 10,
        "n_s": 2,
        "k_delay": 3,
        "ridge": 1e-6
    }

    trainer = Trainer(
        loader=loader,
        features=features,
        readout=Ridge(benchmark_args.get("ridge")),
        experiment_dir=experiment_dir,
        washout=5.0,
        train_duration=10.0,
        test_duration=10.0,
    )
    
    # 4. First Run: Calculate all capacities
    print(f"\n--- Running Initial Benchmark to Calculate All Capacities ---")
    score = benchmark.run(trainer, u_input, **benchmark_args)
    score.save()
    print("--- Initial run complete. ---")

    # 5. Print key metrics and prepare for interactive selection
    if not score.metrics:
        print("Benchmark did not produce any metrics. Exiting.")
        return

    print("\n[Benchmark Results]")
    print(f"  >> Total Capacity: {score.metrics.get('total_capacity', 0):.4f}")
    print(f"  >> Linear Memory Capacity: {score.metrics.get('linear_memory_capacity', 0):.4f}")
    print(f"  >> Nonlinear Memory Capacity: {score.metrics.get('nonlinear_memory_capacity', 0):.4f}")

    capacities = score.metrics.get('capacities')
    basis_names_bytes = score.metrics.get('basis_names', [])
    basis_names = [name.decode('utf-8') for name in basis_names_bytes]

    if capacities is None or not basis_names:
        print("No capacities or basis names found in metrics. Exiting.")
        return

    # 6. Interactive Readout Selection
    valid_indices = ~np.isnan(capacities)
    sorted_indices = np.argsort(capacities[valid_indices])[::-1]
    sorted_capacities = capacities[valid_indices][sorted_indices]
    sorted_names = [basis_names[i] for i in np.where(valid_indices)[0][sorted_indices]]

    print("\n--- Interactive Readout Selection ---")
    print("The following memory tasks are available to be saved:")
    for i, (name, cap) in enumerate(zip(sorted_names, sorted_capacities)):
        if cap >= 0.00: # Only show tasks with non-trivial capacity
            print(f"  Index {i}: Capacity = {cap:.4f}, Basis = '{name}'")
    print(f"Mean Capacity: {np.nanmean(sorted_capacities):.4f}")

    try:
        user_input = input("\nEnter the index of the readout to save (or press Enter to skip): ")
        if user_input.strip() == "":
            print("Skipping readout saving.")
        else:
            try:
                selected_index = int(user_input)
                if 0 <= selected_index < len(sorted_names):
                    selected_basis = sorted_names[selected_index]
                    print(f"\nYou selected index {selected_index}: '{selected_basis}'")
                    
                    # Add the selected basis to benchmark_args and re-run
                    benchmark_args['save_readouts_for'] = [selected_basis]
                    
                    print(f"\n--- Re-running benchmark to train and save readout for '{selected_basis}' ---")
                    score = benchmark.run(trainer, u_input, **benchmark_args)
                    print("--- Readout saved successfully. ---")
                    
                    visualizer = TimeSeriesComparison()
    
                    # The benchmark score object contains the path to the trained readout
                    if score.readout_path and visualizer:
                        print("\n[Processing] Visualizing results")
                        plot_path = visualizer.plot(score.readout_path, start_frame=0, end_frame=500).save()
                        print(f" >> Plot saved to: {plot_path}")
                    
                else:
                    print("Invalid index. No readout will be saved.")
            except ValueError:
                print("Invalid input. Please enter a number. No readout will be saved.")
    except EOFError:
        print("\nNon-interactive mode detected (e.g., CI/CD). Skipping readout saving.")

    # 7. Visualize the capacities
    print("\n[Visualizing Benchmark Results]")
    if capacities is not None and basis_names is not None:
        # Filter out near-zero capacities to remove noise
        threshold = 0.00
        filtered_indices = [i for i, c in enumerate(capacities) if c > threshold]
        
        if not filtered_indices:
            print("No capacities above threshold to plot.")
            return

        filtered_scores = capacities[filtered_indices]
        filtered_names = [basis_names[i] for i in filtered_indices]

        # Sort descending
        sort_idx = np.argsort(filtered_scores)[::-1]
        sorted_scores = filtered_scores[sort_idx]
        sorted_names = [filtered_names[i] for i in sort_idx]

        # --- FIX: Limit to Top N and Dynamic Sizing ---
        top_n = 200  # Only plot the top 50 tasks to prevent overlap
        
        if len(sorted_scores) > top_n:
            print(f"  >> Truncating plot to top {top_n} tasks (out of {len(sorted_scores)} valid tasks).")
            sorted_scores = sorted_scores[:top_n]
            sorted_names = sorted_names[:top_n]

        # Calculate width: ensure at least 0.25 inches per bar so text fits
        dynamic_width = max(10, len(sorted_scores) * 0.3)
        
        plt.figure(figsize=(dynamic_width, 8)) # Dynamic width
        plt.bar(range(len(sorted_scores)), sorted_scores, align='center', width=0.8)
        
        # Format Ticks
        plt.xticks(
            range(len(sorted_scores)), 
            sorted_names, 
            rotation=90, 
            fontsize=9,      # Smaller font
            ha='center'      # Center alignment for vertical text
        )
        
        plt.ylabel("Capacity ($R^2$ Score)")
        plt.title(f"Information Processing Capacity (Top {len(sorted_scores)} Tasks)")
        plt.ylim(0, 1.05)
        plt.xlim(-1, len(sorted_scores)) # Tighten x-axis
        plt.tight_layout()

        plot_dir = experiment_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "information_processing_capacity.svg"
        plt.savefig(plot_path)
        plt.close()
        print(f"  >> Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()