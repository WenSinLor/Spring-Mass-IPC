import sys
import numpy as np
from pathlib import Path
import traceback
import matplotlib.pyplot as plt

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# --- Core Library Imports ---
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodePositions
from openprc.reservoir.readout.ridge import Ridge
from openprc.reservoir.training.trainer import Trainer
from openprc.analysis.benchmarks.memory_benchmark import MemoryBenchmark

def process_single_experiment(h5_path):
    """
    Runs the full memory benchmark pipeline for a single experiment file.
    Returns True if successful, False otherwise.
    """
    # Context Logging
    sample_name = h5_path.parent.name      # sample_0
    amp_name = h5_path.parents[1].name     # amp=1
    topo_name = h5_path.parents[2].name    # topology_1
    experiment_dir = h5_path.parent

    print(f"\nProcessing: [{topo_name}] / [{amp_name}] / [{sample_name}]")

    # 1. Setup Loader and Load Actuation
    try:
        loader = StateLoader(h5_path)
        u_input = loader.get_actuation_signal(actuator_idx=0, dof=0)
        if u_input is None:
            print(f"   [Skipping] No actuation signal found in {h5_path.name}")
            return False
    except Exception as e:
        print(f"   [Error] Failed to load actuation data: {e}")
        return False

    # 2. Setup Trainer & Benchmark
    try:
        features = NodePositions()
        benchmark = MemoryBenchmark(group_name="memory_benchmark")

        benchmark_args = {
            "tau_s": 15,
            "n_s": 2,
            "k_delay": 10,
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
    except Exception as e:
        print(f"   [Error] Failed to initialize Trainer: {e}")
        return False

    # 3. Run Standard Benchmark
    try:
        score = benchmark.run(trainer, u_input, **benchmark_args)
        score.save()
    except Exception as e:
        print(f"   [Error] Benchmark execution failed: {e}")
        traceback.print_exc()
        return False

    # 4. Print Summary Metrics
    if score.metrics:
        print(f"  >> Total Capacity: {score.metrics.get('total_capacity', 0):.4f}")
        print(f"  >> Linear Memory Capacity: {score.metrics.get('linear_memory_capacity', 0):.4f}")
        print(f"  >> Nonlinear Memory Capacity: {score.metrics.get('nonlinear_memory_capacity', 0):.4f}")
    else:
        print("   [Done] No metrics were produced.")

    # 5. Visualize and save the capacities plot
    try:
        if score.metrics:
            capacities = score.metrics.get('capacities')
            basis_names_bytes = score.metrics.get('basis_names', [])
            basis_names = [name.decode('utf-8') for name in basis_names_bytes]

            if capacities is not None and basis_names:
                threshold = 1e-9
                filtered_indices = [i for i, c in enumerate(capacities) if c > threshold]

                if not filtered_indices:
                    print("   [Info] No capacities above threshold to plot.")
                    return True

                filtered_scores = capacities[filtered_indices]
                filtered_names = [basis_names[i] for i in filtered_indices]

                # Sort for better visualization
                vis_sorted_indices = np.argsort(filtered_scores)[::-1]
                vis_sorted_scores = filtered_scores[vis_sorted_indices]
                vis_sorted_names = [filtered_names[i] for i in vis_sorted_indices]

                plt.figure(figsize=(12, 8))
                plt.bar(range(len(vis_sorted_scores)), vis_sorted_scores, tick_label=vis_sorted_names)
                plt.xticks(rotation=90)
                plt.ylabel("Capacity (R^2 Score)")
                plt.title(f"Information Processing Capacity - {topo_name}/{amp_name}/{sample_name}")
                plt.ylim(0, 1)
                plt.tight_layout()

                plot_dir = experiment_dir / "plots"
                plot_dir.mkdir(exist_ok=True)
                plot_path = plot_dir / "information_processing_capacity.svg"
                plt.savefig(plot_path)
                plt.close()
                print(f"  >> Capacity plot saved to: {plot_path}")
    except Exception as e:
        print(f"   [Warning] Visualization failed: {e}")

    return True

def main():
    print("Starting Global Memory Benchmark Run...")

    data_root = src_dir.parent / "data" / "experiment_data"

    if not data_root.exists():
        print(f"[Error] Data directory not found: {data_root}")
        return

    all_experiments = sorted(list(data_root.rglob("experiment.h5")))

    if not all_experiments:
        print("No 'experiment.h5' files found.")
        return

    print(f"Found {len(all_experiments)} experiments. Starting processing loop...\n")
    print("="*60)

    success_count = 0
    fail_count = 0

    for h5_path in all_experiments:
        result = process_single_experiment(h5_path)
        if result:
            success_count += 1
        else:
            fail_count += 1

    print("="*60)
    print(f"\nBatch Complete.")
    print(f"Successful: {success_count}")
    print(f"Failed:     {fail_count}")

if __name__ == "__main__":
    main()
