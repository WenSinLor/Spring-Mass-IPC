import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge


current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from openprc.reservoir.io.state_loader import StateLoader


DEFAULT_TOPOLOGY = "topology_17_prestress"
DEFAULT_AMPLITUDE = "amp=1"
DEFAULT_HIDDEN_NODE = 10
DEFAULT_REFERENCE_NODE = 0
DEFAULT_HORIZON_STEPS = 5
DEFAULT_WASHOUT_S = 5.0
DEFAULT_TRAIN_S = 10.0
DEFAULT_TEST_S = 10.0
DEFAULT_TAU_MAX = 5
DEFAULT_RIDGE = 1e-6
NUM_CV_BLOCKS = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute fixed task-weighted memory score using a common linear input dictionary."
    )
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    parser.add_argument("--topologies", default=None)
    parser.add_argument("--amplitude", default=DEFAULT_AMPLITUDE)
    parser.add_argument("--sample", default="all")
    parser.add_argument("--hidden-node", type=int, default=DEFAULT_HIDDEN_NODE)
    parser.add_argument("--reference-node", type=int, default=DEFAULT_REFERENCE_NODE)
    parser.add_argument("--horizon-steps", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument("--washout", type=float, default=DEFAULT_WASHOUT_S)
    parser.add_argument("--train", type=float, default=DEFAULT_TRAIN_S)
    parser.add_argument("--test", type=float, default=DEFAULT_TEST_S)
    parser.add_argument("--tau-max", type=int, default=DEFAULT_TAU_MAX)
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--memory-per-sample-csv", default=None)
    parser.add_argument("--hidden-metrics-csv", default=None)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_to_unit_interval(u):
    u = np.asarray(u, dtype=float).reshape(-1)
    return 2.0 * (u - np.min(u)) / max(np.max(u) - np.min(u), np.finfo(float).eps) - 1.0


def legendre_p1(x):
    return np.sqrt(3.0) * x


def load_positions(h5_path):
    with h5py.File(h5_path, "r") as f:
        return f["time_series/time"][:], f["time_series/nodes/positions"][:, :, :2]


def find_sample_dirs(base_dir, sample_arg):
    if sample_arg != "all":
        return [base_dir / sample_arg]
    return sorted(p for p in base_dir.glob("sample_*") if (p / "experiment.h5").exists())


def frame_window(loader, washout_s, train_s, test_s):
    start = int(washout_s / loader.dt)
    stop = start + int((train_s + test_s) / loader.dt)
    if stop > loader.total_frames:
        raise ValueError(f"Need {stop} frames but sample has {loader.total_frames}.")
    return start, stop


def nmse_2d(y_true, y_pred):
    err = y_true - y_pred
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    return float(np.sum(err**2) / max(np.sum(centered**2), np.finfo(float).eps))


def cv_nmse(P, y, ridge):
    indices = np.arange(len(P))
    blocks = np.array_split(indices, NUM_CV_BLOCKS)
    scores = []
    for block in blocks:
        train_idx = np.setdiff1d(indices, block, assume_unique=True)
        model = Ridge(alpha=ridge, fit_intercept=True)
        model.fit(P[train_idx], y[train_idx])
        scores.append(nmse_2d(y[block], model.predict(P[block])))
    return float(np.mean(scores)), float(np.std(scores, ddof=0))


def discover_hidden_metrics(data_root, amplitude, hidden_node, horizon_steps):
    return sorted(
        data_root.glob(
            f"*/{amplitude}/plots/hidden_node_{hidden_node}_prediction/"
            f"horizon_{horizon_steps}_steps/"
            f"hidden_node_{hidden_node}_h{horizon_steps}_prediction_metrics.csv"
        )
    )


def hidden_metric_map(paths):
    out = {}
    for path in paths:
        for row in read_csv(path):
            topology = row.get("topology") or Path(path).parents[4].name
            key = (topology, row["sample"], int(float(row["hidden_node"])), int(float(row["reference_node"])), int(float(row["horizon_steps"])))
            out[key] = row
    return out


def memory_map(rows):
    out = {}
    for row in rows:
        key = (
            row["topology"],
            row["sample"],
            int(float(row["hidden_node"])),
            int(float(row["reference_node"])),
            int(float(row["horizon_steps"])),
            int(float(row["tau_frames"])),
        )
        out[key] = float(row["memory_capacity_clipped"])
    return out


def evaluate_sample(sample_dir, args, memory_by_key, hidden_by_key):
    topology = sample_dir.parent.parent.name
    loader = StateLoader(sample_dir / "experiment.h5")
    _, positions = load_positions(loader.sim_path)
    u = normalize_to_unit_interval(loader.get_actuation_signal(actuator_idx=0, dof=0))
    hidden_relative = positions[:, args.hidden_node, :] - positions[:, args.reference_node, :]
    start, stop = frame_window(loader, args.washout, args.train, args.test)
    valid_start = start + args.tau_max
    rows = np.arange(valid_start, stop)
    y = hidden_relative[rows + args.horizon_steps]

    P = np.column_stack([legendre_p1(u[rows - tau]) for tau in range(args.tau_max + 1)])
    basis_nmse_mean, basis_nmse_std = cv_nmse(P, y, args.ridge)
    model = Ridge(alpha=args.ridge, fit_intercept=True)
    model.fit(P, y)

    centered = y - np.mean(y, axis=0, keepdims=True)
    denom_2d = float(np.sum(centered**2))
    score_rows = []
    total = 0.0
    key_base = (topology, sample_dir.name, args.hidden_node, args.reference_node, args.horizon_steps)
    hidden_row = hidden_by_key.get(key_base)
    actual_nmse = float(hidden_row["test_nmse_2d"]) if hidden_row and "test_nmse_2d" in hidden_row else np.nan
    for tau in range(args.tau_max + 1):
        p = P[:, tau]
        coef_x = float(model.coef_[0, tau])
        coef_y = float(model.coef_[1, tau])
        c_tau = ((coef_x**2 + coef_y**2) * float(np.sum(p**2))) / max(denom_2d, np.finfo(float).eps)
        mc_tau = memory_by_key.get((*key_base, tau), np.nan)
        contribution = c_tau * mc_tau if np.isfinite(mc_tau) else np.nan
        if np.isfinite(contribution):
            total += contribution
        score_rows.append(
            {
                "topology": topology,
                "amplitude": sample_dir.parent.name,
                "sample": sample_dir.name,
                "hidden_node": args.hidden_node,
                "reference_node": args.reference_node,
                "horizon_steps": args.horizon_steps,
                "tau": tau,
                "target_weight_c_tau": c_tau,
                "MC_tau": mc_tau,
                "weighted_contribution": contribution,
                "Q_tw_memory": total,
                "basis_nmse_cv_2d": basis_nmse_mean,
                "basis_nmse_cv_2d_std": basis_nmse_std,
                "actual_test_nmse_2d": actual_nmse,
            }
        )
    return score_rows


def main():
    args = parse_args()
    data_root = src_dir.parent / "data" / "experiment_data"
    topologies = [t.strip() for t in args.topologies.split(",")] if args.topologies else [args.topology]
    out_dir = (
        data_root
        / "fixed_memory_analysis"
        / args.amplitude
        / f"hidden_node_{args.hidden_node}"
        / f"horizon_{args.horizon_steps}_steps"
        / "fixed_task_weighted_memory"
    )
    memory_csv = (
        Path(args.memory_per_sample_csv)
        if args.memory_per_sample_csv
        else out_dir.parent / "linear_memory_capacity_per_sample.csv"
    )
    hidden_paths = (
        [Path(p.strip()) for p in args.hidden_metrics_csv.split(",") if p.strip()]
        if args.hidden_metrics_csv
        else discover_hidden_metrics(data_root, args.amplitude, args.hidden_node, args.horizon_steps)
    )
    memory_by_key = memory_map(read_csv(memory_csv))
    hidden_by_key = hidden_metric_map(hidden_paths)

    all_rows = []
    for topology in topologies:
        for sample_dir in find_sample_dirs(data_root / topology / args.amplitude, args.sample):
            print(f"-> Fixed task-weighted memory: {topology}/{sample_dir.name}")
            all_rows.extend(evaluate_sample(sample_dir, args, memory_by_key, hidden_by_key))

    write_csv(all_rows, out_dir / "fixed_task_weighted_memory_score.csv")
    write_csv(all_rows, out_dir / "fixed_task_weights.csv")
    weak = [r for r in all_rows if float(r["basis_nmse_cv_2d"]) >= 0.35]
    if weak:
        print(
            "The fixed input-history dictionary only partially explains the hidden-node target. "
            "Q_tw_memory should be interpreted as a partial input-driven explanation, "
            "not as a complete task-performance predictor."
        )
    print(f"Saved fixed task-weighted memory outputs to: {out_dir}")


if __name__ == "__main__":
    main()
