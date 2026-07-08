import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from openprc.reservoir.features.node_features import NodeDisplacements
from openprc.reservoir.io.state_loader import StateLoader


DEFAULT_TOPOLOGY = "topology_17_prestress"
DEFAULT_AMPLITUDE = "amp=1"
DEFAULT_HIDDEN_NODE = 10
DEFAULT_REFERENCE_NODE = 0
DEFAULT_HORIZON_STEPS = 5
DEFAULT_WASHOUT_S = 5.0
DEFAULT_TRAIN_S = 10.0
DEFAULT_TEST_S = 10.0
DEFAULT_RIDGE = 1e-6
DEFAULT_MAX_DELAY_FRAMES = 30
DEFAULT_RELEVANT_DELAY_MAX = 5

PALETTE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute fixed delayed-input linear memory capacity from visible-node states."
    )
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    parser.add_argument(
        "--topologies",
        default=None,
        help="Optional comma-separated topology names. Overrides --topology.",
    )
    parser.add_argument("--amplitude", default=DEFAULT_AMPLITUDE)
    parser.add_argument("--sample", default="all")
    parser.add_argument("--hidden-node", type=int, default=DEFAULT_HIDDEN_NODE)
    parser.add_argument("--reference-node", type=int, default=DEFAULT_REFERENCE_NODE)
    parser.add_argument("--horizon-steps", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument("--washout", type=float, default=DEFAULT_WASHOUT_S)
    parser.add_argument("--train", type=float, default=DEFAULT_TRAIN_S)
    parser.add_argument("--test", type=float, default=DEFAULT_TEST_S)
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--max-delay-frames", type=int, default=DEFAULT_MAX_DELAY_FRAMES)
    parser.add_argument("--relevant-delay-max", type=int, default=DEFAULT_RELEVANT_DELAY_MAX)
    parser.add_argument("--input-source", default="actuation_signal", choices=["actuation_signal"])
    parser.add_argument(
        "--global-standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use lab convention: z-score full X before adding bias and splitting.",
    )
    return parser.parse_args()


def normalize_to_unit_interval(u):
    u = np.asarray(u, dtype=float).reshape(-1)
    u_min = float(np.min(u))
    u_max = float(np.max(u))
    if abs(u_max - u_min) < 1e-12:
        raise ValueError("Measured input is constant; cannot normalize to [-1, 1].")
    return 2.0 * (u - u_min) / (u_max - u_min) - 1.0


def find_sample_dirs(base_dir: Path, sample_arg: str):
    if sample_arg != "all":
        sample_dir = base_dir / sample_arg
        if not (sample_dir / "experiment.h5").exists():
            raise FileNotFoundError(f"Experiment file not found: {sample_dir / 'experiment.h5'}")
        return [sample_dir]
    sample_dirs = sorted(
        p for p in base_dir.glob("sample_*") if p.is_dir() and (p / "experiment.h5").exists()
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No sample_*/experiment.h5 files found under {base_dir}")
    return sample_dirs


def frame_windows(loader, washout_s, train_s, test_s):
    washout_frames = int(washout_s / loader.dt)
    train_frames = int(train_s / loader.dt)
    test_frames = int(test_s / loader.dt)
    train_start = washout_frames
    train_stop = train_start + train_frames
    test_stop = train_stop + test_frames
    if test_stop > loader.total_frames:
        raise ValueError(
            f"Need {test_stop} frames for washout/train/test, but only "
            f"{loader.total_frames} frames are available."
        )
    return train_start, train_stop, test_stop


def regression_scores(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    nmse = sse / max(sst, np.finfo(float).eps)
    return 1.0 - nmse, nmse


def load_state_and_input(sample_dir, hidden_node, reference_node, global_standardize):
    loader = StateLoader(sample_dir / "experiment.h5")
    with h5py.File(loader.sim_path, "r") as f:
        n_nodes = f["time_series/nodes/positions"].shape[1]

    state_nodes = [node for node in range(n_nodes) if node not in {hidden_node, reference_node}]
    features = NodeDisplacements(reference_node=reference_node, node_ids=state_nodes, dims=[0, 1])
    X_full = features.transform(loader)
    if global_standardize:
        X_full = StandardScaler().fit_transform(X_full)
    design = np.hstack([np.ones((X_full.shape[0], 1)), X_full])
    u = normalize_to_unit_interval(loader.get_actuation_signal(actuator_idx=0, dof=0))
    return loader, design, u, len(state_nodes), X_full.shape[1]


def evaluate_sample(sample_dir, args):
    topology = sample_dir.parent.parent.name
    loader, design, u, num_state_nodes, num_state_features = load_state_and_input(
        sample_dir,
        args.hidden_node,
        args.reference_node,
        args.global_standardize,
    )
    train_start, train_stop, test_stop = frame_windows(loader, args.washout, args.train, args.test)
    records = []
    for tau in range(args.max_delay_frames + 1):
        train_start_valid = max(train_start, tau)
        if train_start_valid >= train_stop or train_stop >= test_stop:
            train_r2 = test_r2 = train_nmse = test_nmse = np.nan
        else:
            X_train = design[train_start_valid:train_stop]
            y_train = u[train_start_valid - tau : train_stop - tau]
            X_test = design[train_stop:test_stop]
            y_test = u[train_stop - tau : test_stop - tau]
            if len(y_test) != len(X_test) or len(y_test) == 0:
                train_r2 = test_r2 = train_nmse = test_nmse = np.nan
            else:
                model = Ridge(alpha=args.ridge, fit_intercept=False)
                model.fit(X_train, y_train)
                train_r2, train_nmse = regression_scores(y_train, model.predict(X_train))
                test_r2, test_nmse = regression_scores(y_test, model.predict(X_test))

        records.append(
            {
                "topology": topology,
                "amplitude": sample_dir.parent.name,
                "sample": sample_dir.name,
                "hidden_node": args.hidden_node,
                "reference_node": args.reference_node,
                "horizon_steps": args.horizon_steps,
                "tau_frames": tau,
                "tau_seconds": float(tau * loader.dt),
                "num_state_nodes": num_state_nodes,
                "num_state_features": num_state_features,
                "train_r2_raw": train_r2,
                "test_r2_raw": test_r2,
                "train_nmse": train_nmse,
                "test_nmse": test_nmse,
                "memory_capacity_raw": test_r2,
                "memory_capacity_clipped": max(test_r2, 0.0) if np.isfinite(test_r2) else np.nan,
            }
        )
    return records


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_tau(records):
    summary = []
    keys = sorted({(r["topology"], r["tau_frames"], r["tau_seconds"]) for r in records})
    for topology, tau, tau_s in keys:
        group = [r for r in records if r["topology"] == topology and r["tau_frames"] == tau]
        clipped = np.asarray([r["memory_capacity_clipped"] for r in group], dtype=float)
        raw = np.asarray([r["memory_capacity_raw"] for r in group], dtype=float)
        summary.append(
            {
                "topology": topology,
                "tau_frames": tau,
                "tau_seconds": tau_s,
                "mc_mean": float(np.nanmean(clipped)),
                "mc_std": float(np.nanstd(clipped, ddof=0)),
                "mc_raw_mean": float(np.nanmean(raw)),
                "mc_raw_std": float(np.nanstd(raw, ddof=0)),
                "num_samples": int(np.sum(np.isfinite(clipped))),
            }
        )
    return summary


def memory_metrics(records, max_delay_frames, relevant_delay_max):
    metrics = []
    keys = sorted({(r["topology"], r["sample"], r["hidden_node"], r["reference_node"], r["horizon_steps"]) for r in records})
    for topology, sample, hidden_node, reference_node, horizon_steps in keys:
        group = [
            r
            for r in records
            if r["topology"] == topology
            and r["sample"] == sample
            and r["hidden_node"] == hidden_node
            and r["reference_node"] == reference_node
            and r["horizon_steps"] == horizon_steps
        ]
        by_tau = {int(r["tau_frames"]): r for r in group}

        def area(stop):
            return float(
                np.nansum(
                    [
                        by_tau[t]["memory_capacity_clipped"]
                        for t in range(min(stop, max_delay_frames) + 1)
                        if t in by_tau
                    ]
                )
            )

        taus = np.asarray(sorted(by_tau), dtype=int)
        vals = np.asarray([by_tau[t]["memory_capacity_clipped"] for t in taus], dtype=float)
        finite = np.isfinite(vals)
        if np.any(finite):
            peak_tau = int(taus[finite][np.nanargmax(vals[finite])])
            above = taus[np.where(vals > 0.05)[0]]
            memory_len = int(np.max(above)) if len(above) else -1
        else:
            peak_tau = -1
            memory_len = -1
        metrics.append(
            {
                "topology": topology,
                "amplitude": group[0]["amplitude"],
                "sample": sample,
                "hidden_node": hidden_node,
                "reference_node": reference_node,
                "horizon_steps": horizon_steps,
                "M_0_5": area(5),
                "M_0_10": area(10),
                "M_0_relevant": area(relevant_delay_max),
                "M_total": area(max_delay_frames),
                "peak_tau": peak_tau,
                "memory_length_threshold_0p05": memory_len,
            }
        )
    return metrics


def plot_memory(summary, metrics, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    topologies = sorted({r["topology"] for r in summary})

    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    for i, topology in enumerate(topologies):
        group = sorted([r for r in summary if r["topology"] == topology], key=lambda r: r["tau_frames"])
        x = np.asarray([r["tau_frames"] for r in group], dtype=float)
        mean = np.asarray([r["mc_mean"] for r in group], dtype=float)
        std = np.asarray([r["mc_std"] for r in group], dtype=float)
        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, mean, marker="o", lw=1.2, color=color, label=topology)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("delay tau (frames)")
    ax.set_ylabel("linear memory capacity")
    ax.set_title("Fixed delayed-input linear memory capacity")
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    ax.legend(fontsize=6)
    fig.savefig(out_dir / "linear_memory_capacity_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    metric_names = ["M_0_5", "M_0_10", "M_total"]
    x = np.arange(len(topologies))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    for j, metric_name in enumerate(metric_names):
        means = []
        stds = []
        for topology in topologies:
            vals = np.asarray([r[metric_name] for r in metrics if r["topology"] == topology], dtype=float)
            means.append(float(np.nanmean(vals)))
            stds.append(float(np.nanstd(vals, ddof=0)))
        ax.bar(x + (j - 1) * width, means, width, yerr=stds, label=metric_name)
    ax.set_xticks(x)
    ax.set_xticklabels(topologies, rotation=25, ha="right")
    ax.set_ylabel("summed clipped memory capacity")
    ax.set_title("Relevant and total memory area")
    ax.legend(fontsize=6)
    fig.savefig(out_dir / "linear_memory_capacity_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def write_readme(out_dir):
    text = """# Fixed Delayed-Input Linear Memory Analysis

This analysis should be used for topology-memory comparison.

Do not compare topology-specific selected H,D from hidden-node basis selection as
the main memory evidence. The hidden-node target changes with topology, so a
topology-specific basis selection does not define a common task demand.

This script measures fixed delayed-input recall:

    s_m^(theta) -> u_{m-tau}

The delayed-input target is common across topologies, so it is a fair memory
benchmark. Actual hidden-node task performance should be measured separately:

    s_m^(theta) -> y_{m+h}^{(theta)}

The comparison script merges these fixed memory metrics with actual hidden-node
prediction NMSE. A positive relationship supports, but does not prove, that
memory-enhanced visible-state encoding contributes to hidden-node prediction.
"""
    (out_dir / "README_fixed_memory_analysis.md").write_text(text)


def main():
    configure_matplotlib()
    args = parse_args()
    data_root = src_dir.parent / "data" / "experiment_data"
    topologies = (
        [t.strip() for t in args.topologies.split(",") if t.strip()]
        if args.topologies
        else [args.topology]
    )
    out_dir = (
        data_root
        / "fixed_memory_analysis"
        / args.amplitude
        / f"hidden_node_{args.hidden_node}"
        / f"horizon_{args.horizon_steps}_steps"
    )

    all_records = []
    for topology in topologies:
        base_dir = data_root / topology / args.amplitude
        for sample_dir in find_sample_dirs(base_dir, args.sample):
            print(f"-> Memory capacity: {topology}/{args.amplitude}/{sample_dir.name}")
            all_records.extend(evaluate_sample(sample_dir, args))

    summary = summarize_by_tau(all_records)
    metrics = memory_metrics(all_records, args.max_delay_frames, args.relevant_delay_max)
    write_csv(all_records, out_dir / "linear_memory_capacity_per_sample.csv")
    write_csv(summary, out_dir / "linear_memory_capacity_summary.csv")
    write_csv(metrics, out_dir / "linear_memory_capacity_metrics.csv")
    plot_memory(summary, metrics, out_dir)
    write_readme(out_dir)

    print(f"Saved memory analysis outputs to: {out_dir}")
    for topology in sorted({r["topology"] for r in metrics}):
        vals = [r["M_0_relevant"] for r in metrics if r["topology"] == topology]
        print(
            f"  {topology}: mean M_0_relevant={np.nanmean(vals):.4f} "
            f"+/- {np.nanstd(vals, ddof=0):.4f}"
        )


if __name__ == "__main__":
    main()
