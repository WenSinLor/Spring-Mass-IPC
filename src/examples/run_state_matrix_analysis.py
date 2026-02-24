import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# --- Core Library Imports ---
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodeDisplacements
from openprc.reservoir.readout.ridge import Ridge
from openprc.reservoir.training.trainer import Trainer
from openprc.analysis.benchmarks.custom_benchmark import CustomBenchmark


def state_matrix_analysis_logic(benchmark, trainer, u_input, **kwargs):
    """
    Computes the effective rank and conditioning number of the reservoir states matrix X_full.
    """
    # 1. Get X_full from trainer
    X_full = trainer.features.transform(trainer.loader)
    scaler_X = StandardScaler()
    X_std = scaler_X.fit_transform(X_full)
    
    # 1. Singular Value Decomposition (SVD)
    U, s, Vh = np.linalg.svd(X_std, full_matrices=False)
    
    # Normalize singular values
    s_norm = s / np.sum(s)
    
    # 2. Calculate Effective Rank (Entropy of singular values)
    # Higher value = More independent nodes (Good!)
    effective_rank = np.exp(-np.sum(s_norm * np.log(s_norm + 1e-12)))
    
    # 3. Calculate Condition Number
    # Lower value = More stable readout (Good!)
    cond_num = s[0] / s[-1]

    # 4. PCA Analysis for dimensionality
    pca = PCA()
    pca.fit(X_std)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the number of components to explain 95% of the variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

    # 5. Define the metrics and metadata dictionaries to be returned
    metrics = {
        'effective_rank': effective_rank,
        'condition_number': cond_num,
        'pca_95_variance_components': n_components_95,
    }
    metadata = {
        'feature_type': trainer.features.__class__.__name__,
        'benchmark_class': benchmark.__class__.__name__
    }
    
    return metrics, metadata

def analyze_reservoir_pca(states, n_components=None, plot=True):
    """
    Performs PCA on reservoir states to analyze effective dimensionality.
    
    Args:
        states (np.ndarray): Shape (Time_Steps, Num_Nodes).
        n_components (int): Number of PCs to compute (default: all).
        plot (bool): Whether to plot the Scree plot and PC trajectories.
        
    Returns:
        pca (sklearn.PCA): The fitted PCA object.
        states_pca (np.ndarray): The states projected onto the principal components.
        cumulative_variance (np.ndarray): Cumulative explained variance ratio.
    """
    # 1. STANDARDIZE (Crucial for Physical Reservoirs)
    # This removes the "Loudness" bias so we can see the "Structure"
    scaler = StandardScaler()
    states_std = scaler.fit_transform(states)
    
    # 2. Perform PCA
    if n_components is None:
        n_components = min(states.shape)
        
    pca = PCA(n_components=n_components)
    states_pca = pca.fit_transform(states_std)
    
    # 3. Analyze Variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # 4. Visualization
    if plot:
        plt.figure(figsize=(12, 5))
        
        # Plot A: Scree Plot (The "Energy Distribution")
        plt.subplot(1, 1, 1)
        plt.bar(range(1, len(explained_var) + 1), explained_var, label='Individual')
        plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 's--', alpha=0.6, label='Cumulative')
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Principal Component')
        plt.title('Scree Plot: Where is the Information?')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """
    An example pipeline that runs a custom benchmark to analyze the reservoir state matrix.
    """
    
    # 1. Define the Experiment Path
    NUM_SAMPLES = 5
    for i in range(NUM_SAMPLES):
        TOPOLOGY = "topology_8"
        AMPLITUDE = "amp=1"
        SAMPLE = f"sample_{i}"
        
        data_root = src_dir.parent / "data" / "experiment_data"
        experiment_dir = data_root / TOPOLOGY / AMPLITUDE / SAMPLE
        h5_path = experiment_dir / "experiment.h5"

        if not h5_path.exists():
            print(f"Simulation missing at {h5_path}! Please run a simulation first.")
            continue
    
        # 2. Shared Setup
        loader = StateLoader(h5_path)
        features = NodeDisplacements(reference_node=1, dims=[0])
        u_input = loader.get_actuation_signal(actuator_idx=1, dof=0)
        
        print(f"Loaded {loader.total_frames} frames from {h5_path.name}")

        # Run PCA analysis
        states = features.transform(loader)
        analyze_reservoir_pca(states, plot=True)

        # 3. Define the Trainer
        # Note: The trainer is required by the benchmark, but for this specific analysis,
        # we don't need to run the training process itself. We only need the trainer
        # to get access to the features and other experiment settings.
        trainer = Trainer(
            loader=loader,
            features=features,
            readout=Ridge(1e-5), # Dummy readout, not used in this benchmark
            experiment_dir=experiment_dir
        )

        # 4. Instantiate and run the CustomBenchmark
        benchmark = CustomBenchmark(
            group_name="state_matrix_analysis",
            benchmark_logic=state_matrix_analysis_logic
        )
        
        print(f"Running benchmark: {benchmark.__class__.__name__}")
        score = benchmark.run(trainer, u_input)
        score.save(filename="metrics.h5")
        
        print(f"--- Benchmark complete for: {experiment_dir.name} ---")

        # 5. Print Metrics
        if score.metrics:
            print("\n[Processing] Printing Benchmark Results:")
            def print_metrics(metrics_dict, indent=""):
                for key, value in metrics_dict.items():
                    print_prefix = f"{indent}>> {key}: "
                    if isinstance(value, dict):
                        print(f"{indent}>> {key}:")
                        print_metrics(value, indent + "  ")
                    elif isinstance(value, (int, float, np.number)):
                        print(f"{print_prefix}{value:.5f}")
                    else:
                        print(f"{print_prefix}{value}")
            print_metrics(score.metrics, indent="  ")


if __name__ == "__main__":
    main()
