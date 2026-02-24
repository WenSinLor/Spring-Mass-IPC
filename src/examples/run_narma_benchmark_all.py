import sys
import numpy as np
import h5py
from pathlib import Path
import traceback

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# --- Core Library Imports ---
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodePositions, NodeDisplacements
from openprc.reservoir.features.bar_features import BarLengths, BarExtensions
from openprc.reservoir.readout.ridge import Ridge
from openprc.reservoir.training.trainer import Trainer
from openprc.analysis.visualization.time_series import TimeSeriesComparison
from openprc.analysis.benchmarks.narma_benchmark import NARMABenchmark


def process_single_experiment(h5_path):
    """
    Runs the full benchmark pipeline for a single experiment file.
    Returns True if successful, False otherwise.
    """
    # Context Logging
    sample_name = h5_path.parent.name      # sample_0
    amp_name = h5_path.parents[1].name     # amp=1
    topo_name = h5_path.parents[2].name    # topology_1
    experiment_dir = h5_path.parent
    
    print(f"\nProcessing: [{topo_name}] / [{amp_name}] / [{sample_name}]")

    # 1. Load Actuation
    try:
        loader = StateLoader(h5_path)
        u_raw = loader.get_actuation_signal(actuator_idx=1, dof=0)
    except Exception as e:
        print(f"   [Error] Failed to initialize StateLoader or get actuation: {e}")
        return False
        
    if u_raw is None: 
        print(f"   [Skipping] No actuation signal found in {h5_path.name}")
        return False
        
    if u_raw.ndim > 1: u_raw = u_raw.flatten()
    u_scaled = (u_raw - np.nanmin(u_raw)) / (np.nanmax(u_raw) - np.nanmin(u_raw)) * 0.5
    # print(f"   Actuation Loaded (Shape: {u_raw.shape})")

    # 2. Setup Trainer
    try:
        features = NodeDisplacements(reference_node=1, dims=[0]) # Extracts pixels
        
        trainer = Trainer(
            loader=loader,
            features=features,
            readout=Ridge(1e-5),
            experiment_dir=experiment_dir,
            washout=5.0,
            train_duration=10.0,
            test_duration=10.0
        )
    except Exception as e:
        print(f"   [Error] Failed to initialize Trainer: {e}")
        return False

    # 3. Run Standard Benchmark
    try:
        benchmark = NARMABenchmark(group_name="narma_benchmark")
        
        # Run benchmark (Standard 2nd Order NARMA as per your code)
        score = benchmark.run(trainer, u_scaled, order=2)
        score.save()
        
        # print(f"   -> Metrics saved to metrics.h5")

    except Exception as e:
        print(f"   [Error] Benchmark execution failed: {e}")
        traceback.print_exc()
        return False

    # 4. Visualize (Optional but recommended)
    try:
        visualizer = TimeSeriesComparison()
        if score.readout_path and visualizer:
            plot_path = visualizer.plot(score.readout_path, start_frame=0, end_frame=500).save()
            print(f"   -> Plot saved to: {plot_path.name}")
    except Exception as e:
        print(f"   [Warning] Visualization failed: {e}")

    # 5. Print Summary Metric (Optional)
    if score.metrics:
        nrmse = score.metrics.get('narma2_nrmse', 'N/A')
        print(f"   [Done] NRMSE: {nrmse}")
    
    return True

def main():
    print("Starting Global Benchmark Run...")
    
    # 1. Locate Data Root
    data_root = src_dir.parent / "data" / "experiment_data" / "topology_6_narma"
    
    if not data_root.exists():
        print(f"[Error] Data directory not found: {data_root}")
        return

    # 2. Find ALL experiment.h5 files
    # rglob recursively searches through all topologies and amplitudes
    all_experiments = sorted(list(data_root.rglob("experiment.h5")))
    
    if not all_experiments:
        print("No 'experiment.h5' files found. Please run slicing script first.")
        return

    print(f"Found {len(all_experiments)} experiments. Starting processing loop...\n")
    print("="*60)

    # 3. Loop and Process
    success_count = 0
    fail_count = 0
    
    for h5_path in all_experiments:
        result = process_single_experiment(h5_path)
        if result:
            success_count += 1
        else:
            fail_count += 1
            
    print("="*60)
    print(f"Batch Complete.")
    print(f"Successful: {success_count}")
    print(f"Failed:     {fail_count}")

if __name__ == "__main__":
    main()