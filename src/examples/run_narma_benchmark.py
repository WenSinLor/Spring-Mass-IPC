import sys
import numpy as np
import h5py
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# --- Core Library Imports ---
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodePositions
from openprc.reservoir.readout.ridge import Ridge
from openprc.reservoir.training.trainer import Trainer
from openprc.analysis.visualization.time_series import TimeSeriesComparison

# --- Import Standard Benchmark ---
from openprc.analysis.benchmarks.narma_benchmark import NARMABenchmark

def main():
    # ==========================================
    # 1. Configuration
    # ==========================================
    TOPOLOGY = "topology_1"
    AMPLITUDE = "amp=1"
    SAMPLE = "sample_0"
    
    data_root = src_dir.parent / "data" / "experiment_data"
    experiment_dir = data_root / TOPOLOGY / AMPLITUDE / SAMPLE
    h5_path = experiment_dir / "experiment.h5"

    if not h5_path.exists():
        print(f"[Error] Experiment file not found: {h5_path}")
        return

    print(f"-> Loading Experiment: {AMPLITUDE} / {SAMPLE}")

    # ==========================================
    # 2. Load & Scale Actuation
    # ==========================================
    loader = StateLoader(h5_path)
    u_raw = loader.get_actuation_signal(actuator_idx=0, dof=0)
    if u_raw is None: return
    if u_raw.ndim > 1: u_raw = u_raw.flatten()
    
    print(f"   Actuation Loaded (Shape: {u_raw.shape})")

    # ==========================================
    # 3. Setup Trainer
    # ==========================================
    features = NodePositions() # Extracts pixels
    
    # The Trainer automatically standardizes (Z-score) the features (X) internally.
    trainer = Trainer(
        loader=loader,
        features=features,
        readout=Ridge(1e-5),
        experiment_dir=experiment_dir,
        washout=5.0,
        train_duration=10.0,
        test_duration=10.0
    )

    # ==========================================
    # 4. Run Standard Benchmark
    # ==========================================
    print("\n[Workflow: Standard NARMA Benchmark]")

    # Instantiate the standard class
    benchmark = NARMABenchmark(group_name="narma_benchmark")
    
    # Run it with our pre-scaled input
    # The benchmark handles generation, training, scoring, and internal saving.
    score = benchmark.run(trainer, u_raw, order=2)
    
    # Save the metrics.h5 file
    score.save()
    
    print(f"--- Benchmark complete for: {experiment_dir.name} ---")

    # ==========================================
    # 5. Visualize Results
    # ==========================================
    visualizer = TimeSeriesComparison()
    
    # The benchmark score object contains the path to the trained readout
    if score.readout_path and visualizer:
        print("\n[Processing] Visualizing results")
        plot_path = visualizer.plot(score.readout_path, start_frame=0, end_frame=500).save()
        print(f" >> Plot saved to: {plot_path}")

    # Print Metrics
    if score.metrics:
        print("\n[Processing] Benchmark Results:")
        for key, value in score.metrics.items():
            print(f"  >> {key}: {value:.5f}")

if __name__ == "__main__":
    main()