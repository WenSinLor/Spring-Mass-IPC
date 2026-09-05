"""Generate separate heatmaps for the FPS-matched topology-18 experiment files.

Input files are written by ``prepare_topology18_30fps.py`` under the separate
``topology_18_prestress_30fps`` output root.  Existing topology-10 and native
topology-18 metrics/plots are never overwritten.
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from openprc.analysis.benchmarks.memory_benchmark import MemoryBenchmark
from openprc.reservoir.features.node_features import NodeDisplacements
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.readout.ridge import Ridge
from openprc.reservoir.training.trainer import Trainer

from run_plot_heatmap import plot_heatmap


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOPOLOGY = "topology_18_prestress_30fps"
AMPLITUDE = "amp=1"
NUM_SAMPLES = 5
N_LIST = list(range(1, 5))
TAU_LIST = list(range(30))
K_DELAY = 1
WASHOUT_S = 5.0
TRAIN_S = 10.0
TEST_S = 10.0
RIDGE = 1e-6
P_VALUE = 1e-4


def effective_rank(loader: StateLoader, features: NodeDisplacements) -> float:
    states = features.transform(loader)
    states = StandardScaler().fit_transform(states)
    singular_values = np.linalg.svd(states, compute_uv=False)
    weights = singular_values / np.sum(singular_values)
    return float(np.exp(-np.sum(weights * np.log(weights + 1e-12))))


def dambre_epsilon(rank: float, test_frames: int) -> float:
    return float(2.0 * chi2.isf(P_VALUE, df=rank) / test_frames)


def main() -> None:
    data_root = SRC_DIR.parent / "data" / "experiment_data"
    sample_root = data_root / TOPOLOGY / AMPLITUDE

    for sample_index in range(NUM_SAMPLES):
        sample_dir = sample_root / f"sample_{sample_index}"
        h5_path = sample_dir / "experiment.h5"
        if not h5_path.exists():
            print(f"[SKIP] Missing {h5_path}")
            continue

        loader = StateLoader(h5_path)
        features = NodeDisplacements(reference_node=0, dims=[0, 1])
        u_input = loader.get_actuation_signal(actuator_idx=0, dof=0)
        rank = effective_rank(loader, features)
        test_frames = int(TEST_S / loader.dt)
        epsilon = dambre_epsilon(rank, test_frames)

        print(
            f"sample_{sample_index}: frames={loader.total_frames}, "
            f"dt={loader.dt:.8f} s, test_frames={test_frames}, "
            f"rank={rank:.3f}, eps={epsilon:.6f}"
        )

        heatmap = np.empty((len(N_LIST), len(TAU_LIST)), dtype=float)
        for n_index, tau_index in tqdm(
            list(product(range(len(N_LIST)), range(len(TAU_LIST)))),
            desc=f"sample_{sample_index}",
        ):
            benchmark = MemoryBenchmark(group_name="memory_benchmark_30fps")
            trainer = Trainer(
                loader=loader,
                features=features,
                readout=Ridge(RIDGE),
                experiment_dir=sample_dir,
                washout=WASHOUT_S,
                train_duration=TRAIN_S,
                test_duration=TEST_S,
            )
            score = benchmark.run(
                trainer,
                u_input,
                tau_s=TAU_LIST[tau_index],
                n_s=N_LIST[n_index],
                k_delay=K_DELAY,
                eps=epsilon,
                ridge=RIDGE,
            )
            score.save("metrics_30fps.h5")
            heatmap[n_index, tau_index] = np.nanmean(score.metrics["capacities"])

        lag_step_ms = loader.dt * K_DELAY * 1000.0
        plot_heatmap(
            heatmap,
            N_LIST,
            TAU_LIST,
            k_delay=K_DELAY,
            amp=AMPLITUDE,
            n=16,
            save_dir=sample_dir / "plots",
            save_name="heatmap_R2_topology18_30fps",
            save_svg=True,
            save_png=True,
            show=False,
        )
        print(f"[SAVED] sample_{sample_index} heatmap; lag step = {lag_step_ms:.3f} ms")


if __name__ == "__main__":
    main()
