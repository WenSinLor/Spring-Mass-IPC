import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(src_dir))

from task_specific_ipc_common import (
    PALETTE,
    StateLoader,
    basis_adequacy_label,
    basis_window,
    build_legendre_design,
    data_root,
    frame_counts,
    load_positions,
    normalize_to_unit_interval,
    scalar_r2,
    target_time_delays,
    term_metadata,
    topology_index,
    valid_basis_rows,
    visible_state_design,
    write_csv,
)
from run_hidden_node_prediction import train_hidden_node_readout


DEFAULT_TOPOLOGIES = ["topology_10_prestress", "topology_17_prestress"]
DEFAULT_TOPOLOGY_FORMAT = "topology_{i}_prestress"
DEFAULT_AMPLITUDE = "auto"
DEFAULT_HIDDEN_NODE = 10
DEFAULT_REFERENCE_NODE = 0
DEFAULT_HORIZON_STEPS = 5
DEFAULT_WASHOUT_S = 5.0
DEFAULT_TRAIN_S = 10.0
DEFAULT_TEST_S = 10.0
DEFAULT_H_CUT = 2
DEFAULT_D_CUT = 1
DEFAULT_LAG_STRIDE_FRAMES = 1
DEFAULT_TASK_RIDGE_ALPHA = 0.1
DEFAULT_IPC_RIDGE_ALPHA = 1e-6
DEFAULT_ACTUAL_TASK_RIDGE_ALPHA = 1e-6


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
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
        description=(
            "X-only task-specific IPC comparison for topology_10_prestress and "
            "topology_17_prestress. The hidden-node y target is intentionally "
            "dropped from task-weight fitting and Q_twIPC."
        )
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=DEFAULT_TOPOLOGIES,
        help="Topology names or numeric indices, e.g. topology_10_prestress topology_17_prestress.",
    )
    parser.add_argument("--topology-name-format", default=DEFAULT_TOPOLOGY_FORMAT)
    parser.add_argument(
        "--amplitude",
        default=DEFAULT_AMPLITUDE,
        help="'auto' chooses an available amp folder per topology; otherwise use e.g. amp=2.5.",
    )
    parser.add_argument(
        "--topology-amplitude",
        action="append",
        default=[],
        metavar="TOPOLOGY:AMP",
        help="Override amplitude for one topology, e.g. 17:amp=1. Can repeat.",
    )
    parser.add_argument("--sample", default="all")
    parser.add_argument(
        "--exclude-sample",
        action="append",
        default=[],
        metavar="TOPOLOGY:SAMPLE",
        help="Exclude one sample, e.g. 17:sample_1. Can repeat.",
    )
    parser.add_argument("--hidden-node", type=int, default=DEFAULT_HIDDEN_NODE)
    parser.add_argument("--reference-node", type=int, default=DEFAULT_REFERENCE_NODE)
    parser.add_argument("--horizon-steps", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument("--washout", type=float, default=DEFAULT_WASHOUT_S)
    parser.add_argument("--train", type=float, default=DEFAULT_TRAIN_S)
    parser.add_argument("--test", type=float, default=DEFAULT_TEST_S)
    parser.add_argument("--H-cut", type=int, default=DEFAULT_H_CUT)
    parser.add_argument("--D-cut", type=int, default=DEFAULT_D_CUT)
    parser.add_argument("--lag-stride-frames", type=int, default=DEFAULT_LAG_STRIDE_FRAMES)
    parser.add_argument(
        "--task-ridge-alpha",
        type=float,
        default=DEFAULT_TASK_RIDGE_ALPHA,
        help="Ridge alpha for fitting PDF Eq. S15 task coefficients. Default follows the supplement.",
    )
    parser.add_argument(
        "--ipc-ridge-alpha",
        type=float,
        default=DEFAULT_IPC_RIDGE_ALPHA,
        help="Ridge alpha for visible-state reconstruction of each basis term.",
    )
    parser.add_argument(
        "--actual-task-ridge-alpha",
        type=float,
        default=DEFAULT_ACTUAL_TASK_RIDGE_ALPHA,
        help="Ridge alpha for the actual hidden-node readout used in the NMSE comparison.",
    )
    parser.add_argument(
        "--global-standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Z-score visible-state readout features before IPC readout fitting.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def topology_name_from_arg(value, name_format):
    text = str(value)
    if text.isdigit():
        return name_format.format(i=int(text))
    return text


def parse_topology_value_pairs(values, name_format):
    out = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"Expected TOPOLOGY:VALUE format, got '{value}'.")
        topology, val = [part.strip() for part in value.split(":", 1)]
        if not topology or not val:
            raise ValueError(f"Expected TOPOLOGY:VALUE format, got '{value}'.")
        out[topology_name_from_arg(topology, name_format)] = val
    return out


def parse_exclusions(values, name_format):
    excluded = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"Expected TOPOLOGY:SAMPLE format, got '{value}'.")
        topology, sample = [part.strip() for part in value.split(":", 1)]
        if not topology or not sample:
            raise ValueError(f"Expected TOPOLOGY:SAMPLE format, got '{value}'.")
        topology = topology_name_from_arg(topology, name_format)
        excluded.setdefault(topology, set()).add(sample)
    return excluded


def choose_amplitude_dir(topology_dir, amplitude):
    if amplitude != "auto":
        amp_dir = topology_dir / amplitude
        return amp_dir if amp_dir.exists() else None
    amp_dirs = sorted(
        p
        for p in topology_dir.glob("amp=*")
        if p.is_dir() and any((s / "experiment.h5").exists() for s in p.glob("sample_*"))
    )
    if not amp_dirs:
        return None
    for preferred in ("amp=1", "amp=2.5"):
        for amp_dir in amp_dirs:
            if amp_dir.name == preferred:
                return amp_dir
    return amp_dirs[-1]


def find_sample_dirs(amp_dir, sample_arg, topology, excluded_samples):
    if sample_arg != "all":
        sample_dir = amp_dir / sample_arg
        samples = [sample_dir] if (sample_dir / "experiment.h5").exists() else []
    else:
        samples = sorted(
            p for p in amp_dir.glob("sample_*") if p.is_dir() and (p / "experiment.h5").exists()
        )
    excluded = excluded_samples.get(topology, set())
    return [sample_dir for sample_dir in samples if sample_dir.name not in excluded]


def discover_records(args):
    root = data_root()
    topologies = [topology_name_from_arg(v, args.topology_name_format) for v in args.topologies]
    amplitude_overrides = parse_topology_value_pairs(
        args.topology_amplitude, args.topology_name_format
    )
    excluded_samples = parse_exclusions(args.exclude_sample, args.topology_name_format)
    records = []
    skipped = []
    for topology in topologies:
        topology_dir = root / topology
        if not topology_dir.exists():
            skipped.append(
                {
                    "topology": topology,
                    "amplitude": "",
                    "sample": "",
                    "reason": "topology folder not found",
                }
            )
            continue
        amplitude = amplitude_overrides.get(topology, args.amplitude)
        amp_dir = choose_amplitude_dir(topology_dir, amplitude)
        if amp_dir is None:
            skipped.append(
                {
                    "topology": topology,
                    "amplitude": amplitude,
                    "sample": "",
                    "reason": "amplitude folder not found or has no samples",
                }
            )
            continue
        sample_dirs = find_sample_dirs(amp_dir, args.sample, topology, excluded_samples)
        if not sample_dirs:
            skipped.append(
                {
                    "topology": topology,
                    "amplitude": amp_dir.name,
                    "sample": args.sample,
                    "reason": "no included samples",
                }
            )
            continue
        for sample_dir in sample_dirs:
            records.append(
                {
                    "topology": topology,
                    "amplitude": amp_dir.name,
                    "sample": sample_dir.name,
                    "sample_dir": sample_dir,
                }
            )
    if not records:
        raise FileNotFoundError("No topology/sample records were found.")
    return records, skipped, excluded_samples


def output_dir(args, records, excluded_samples):
    if args.output_dir:
        return Path(args.output_dir)
    topology_labels = sorted({r["topology"] for r in records}, key=topology_index)
    label = "_vs_".join(topology_labels)
    exclude_label = ""
    if excluded_samples:
        parts = []
        for topology in sorted(excluded_samples, key=topology_index):
            for sample in sorted(excluded_samples[topology]):
                parts.append(f"{topology}_{sample}")
        exclude_label = "_exclude_" + "__".join(parts)
    return (
        data_root()
        / "task_specific_ipc"
        / "xonly_topology_task_specific_ipc"
        / f"{label}_H{args.H_cut}_D{args.D_cut}{exclude_label}"
        / f"hidden_node_{args.hidden_node}"
        / f"horizon_{args.horizon_steps}_steps"
    )


def scalar_nmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    centered = y_true - np.mean(y_true)
    return float(
        np.sum((y_true - y_pred) ** 2)
        / np.maximum(np.sum(centered**2), np.finfo(float).eps)
    )


def fit_x_task_coefficients(P, y_x, task_ridge_alpha):
    p_mean = np.mean(P, axis=0, keepdims=True)
    y_mean = float(np.mean(y_x))
    P_centered = P - p_mean
    y_centered = y_x - y_mean
    model = Ridge(alpha=task_ridge_alpha, fit_intercept=False)
    model.fit(P_centered, y_centered)
    pred = model.predict(P_centered) + y_mean
    nmse_x = scalar_nmse(y_x, pred)
    denom_x = float(np.sum(y_centered**2))
    p_norm_sq = np.sum(P_centered**2, axis=0)
    weights = (model.coef_.reshape(-1) ** 2) * p_norm_sq / max(denom_x, np.finfo(float).eps)
    return model, p_mean, y_mean, pred, nmse_x, weights, p_norm_sq, denom_x


def coefficient_rows(record, args, metadata, model, y_mean):
    rows = [
        {
            "topology": record["topology"],
            "amplitude": record["amplitude"],
            "sample": record["sample"],
            "target_component": "x",
            "hidden_node": args.hidden_node,
            "reference_node": args.reference_node,
            "horizon_steps": args.horizon_steps,
            "H_cut": args.H_cut,
            "D_cut": args.D_cut,
            "basis_index": -1,
            "basis_term": "intercept",
            "total_degree": 0,
            "max_readout_time_delay": "",
            "max_target_time_delay": "",
            "max_lag_frames": "",
            "coefficient_x": y_mean,
        }
    ]
    for meta in metadata:
        idx = int(meta["basis_index"])
        rows.append(
            {
                "topology": record["topology"],
                "amplitude": record["amplitude"],
                "sample": record["sample"],
                "target_component": "x",
                "hidden_node": args.hidden_node,
                "reference_node": args.reference_node,
                "horizon_steps": args.horizon_steps,
                "H_cut": args.H_cut,
                "D_cut": args.D_cut,
                **meta,
                "coefficient_x": float(model.coef_[idx]),
            }
        )
    return rows


def task_weight_rows(record, args, metadata, model, weights, p_norm_sq, denom_x):
    rows = []
    for meta in metadata:
        idx = int(meta["basis_index"])
        rows.append(
            {
                "topology": record["topology"],
                "amplitude": record["amplitude"],
                "sample": record["sample"],
                "target_component": "x",
                "hidden_node": args.hidden_node,
                "reference_node": args.reference_node,
                "horizon_steps": args.horizon_steps,
                "H_cut": args.H_cut,
                "D_cut": args.D_cut,
                **meta,
                "coefficient_x": float(model.coef_[idx]),
                "p_alpha_centered_norm_sq": float(p_norm_sq[idx]),
                "target_x_centered_norm_sq": denom_x,
                "c_alpha_x": float(weights[idx]),
            }
        )
    return rows


def compute_ipc_rows(record, args, metadata, P_full):
    loader, X_design, state_nodes, num_state_features = visible_state_design(
        record["sample_dir"],
        args.hidden_node,
        args.reference_node,
        args.global_standardize,
    )
    _, _, _, train_start, train_stop, test_stop = frame_counts(
        loader, args.washout, args.train, args.test
    )
    H = int(args.H_cut)
    lag_stride = int(args.lag_stride_frames)
    train_rows = np.arange(max(train_start, H * lag_stride), train_stop)
    test_rows = np.arange(max(train_stop, H * lag_stride), test_stop)
    if len(train_rows) == 0 or len(test_rows) == 0:
        raise ValueError(f"No valid IPC train/test rows for {record['topology']}/{record['sample']}.")

    rows = []
    for meta in metadata:
        idx = int(meta["basis_index"])
        y_train = P_full[train_rows, idx]
        y_test = P_full[test_rows, idx]
        model = Ridge(alpha=args.ipc_ridge_alpha, fit_intercept=False)
        model.fit(X_design[train_rows], y_train)
        train_r2, train_nmse = scalar_r2(y_train, model.predict(X_design[train_rows]))
        test_r2, test_nmse = scalar_r2(y_test, model.predict(X_design[test_rows]))
        rows.append(
            {
                "topology": record["topology"],
                "amplitude": record["amplitude"],
                "sample": record["sample"],
                "hidden_node": args.hidden_node,
                "reference_node": args.reference_node,
                "horizon_steps": args.horizon_steps,
                "H_cut": args.H_cut,
                "D_cut": args.D_cut,
                **meta,
                "num_state_nodes": len(state_nodes),
                "num_state_features": int(num_state_features),
                "train_r2_raw": train_r2,
                "test_r2_raw": test_r2,
                "train_nmse": train_nmse,
                "test_nmse": test_nmse,
                "IPC_alpha_raw": test_r2,
                "IPC_alpha_clipped": max(test_r2, 0.0),
            }
        )
    return rows


def grouped_contributions(weights, ipc, basis_nmse_x):
    ipc_by_index = {int(r["basis_index"]): r for r in ipc}
    groups = {}
    for w in weights:
        idx = int(w["basis_index"])
        ipc_row = ipc_by_index[idx]
        for group_type, group_value in (
            ("degree", int(w["total_degree"])),
            ("readout_time_delay", int(w["max_readout_time_delay"])),
            ("target_time_delay", int(w["max_target_time_delay"])),
        ):
            key = (
                w["topology"],
                w["amplitude"],
                w["sample"],
                group_type,
                group_value,
            )
            if key not in groups:
                groups[key] = {
                    "topology": w["topology"],
                    "amplitude": w["amplitude"],
                    "sample": w["sample"],
                    "target_component": "x",
                    "group_type": group_type,
                    "group_value": group_value,
                    "c_alpha_x_sum": 0.0,
                    "IPC_alpha_clipped_sum": 0.0,
                    "weighted_contribution_x_sum": 0.0,
                    "num_terms": 0,
                    "basis_nmse_x": basis_nmse_x,
                    "basis_adequacy_label_x": basis_adequacy_label(basis_nmse_x),
                }
            groups[key]["c_alpha_x_sum"] += float(w["c_alpha_x"])
            groups[key]["IPC_alpha_clipped_sum"] += float(ipc_row["IPC_alpha_clipped"])
            groups[key]["weighted_contribution_x_sum"] += float(w["c_alpha_x"]) * float(
                ipc_row["IPC_alpha_clipped"]
            )
            groups[key]["num_terms"] += 1
    return list(groups.values())


def evaluate_sample(record, args):
    sample_dir = record["sample_dir"]
    loader = StateLoader(sample_dir / "experiment.h5")
    _, positions = load_positions(sample_dir / "experiment.h5")
    n_nodes = positions.shape[1]
    if args.hidden_node >= n_nodes or args.reference_node >= n_nodes:
        raise ValueError(
            f"{record['topology']}/{record['sample']}: node index outside range 0..{n_nodes - 1}."
        )
    if args.hidden_node == args.reference_node:
        raise ValueError("Hidden node cannot also be reference node.")

    u_norm = normalize_to_unit_interval(loader.get_actuation_signal(actuator_idx=0, dof=0))
    hidden_relative = positions[:, args.hidden_node, :] - positions[:, args.reference_node, :]
    P_full, exps = build_legendre_design(
        u_norm, args.H_cut, args.D_cut, args.lag_stride_frames
    )
    metadata = term_metadata(exps, args.horizon_steps, args.lag_stride_frames)
    window_start, window_stop = basis_window(
        loader, args.washout, args.train, args.test, args.horizon_steps
    )
    basis_rows = valid_basis_rows(
        window_start, window_stop, args.H_cut, args.lag_stride_frames
    )
    P = P_full[basis_rows]
    y_x = hidden_relative[basis_rows + args.horizon_steps, 0]

    model, p_mean, y_mean, pred_x, basis_nmse_x, c_weights, p_norm_sq, denom_x = (
        fit_x_task_coefficients(P, y_x, args.task_ridge_alpha)
    )
    del p_mean, pred_x

    coeffs = coefficient_rows(record, args, metadata, model, y_mean)
    weights = task_weight_rows(record, args, metadata, model, c_weights, p_norm_sq, denom_x)
    ipc = compute_ipc_rows(record, args, metadata, P_full)
    ipc_by_index = {int(r["basis_index"]): r for r in ipc}

    term_rows = []
    q_raw = 0.0
    q_clipped = 0.0
    total_ipc_raw = 0.0
    total_ipc_clipped = 0.0
    for w in weights:
        ipc_row = ipc_by_index[int(w["basis_index"])]
        contribution_raw = float(w["c_alpha_x"]) * float(ipc_row["IPC_alpha_raw"])
        contribution_clipped = float(w["c_alpha_x"]) * float(ipc_row["IPC_alpha_clipped"])
        q_raw += contribution_raw
        q_clipped += contribution_clipped
        total_ipc_raw += float(ipc_row["IPC_alpha_raw"])
        total_ipc_clipped += float(ipc_row["IPC_alpha_clipped"])
        term_rows.append(
            {
                **w,
                "IPC_alpha_raw": ipc_row["IPC_alpha_raw"],
                "IPC_alpha_clipped": ipc_row["IPC_alpha_clipped"],
                "weighted_contribution_x_raw": contribution_raw,
                "weighted_contribution_x_clipped": contribution_clipped,
                "basis_nmse_x": basis_nmse_x,
                "basis_R2_x": 1.0 - basis_nmse_x,
                "task_ridge_alpha": args.task_ridge_alpha,
                "ipc_ridge_alpha": args.ipc_ridge_alpha,
                "basis_adequacy_label_x": basis_adequacy_label(basis_nmse_x),
            }
        )

    score = {
        "topology": record["topology"],
        "amplitude": record["amplitude"],
        "sample": record["sample"],
        "target_component": "x",
        "hidden_node": args.hidden_node,
        "reference_node": args.reference_node,
        "horizon_steps": args.horizon_steps,
        "H_cut": args.H_cut,
        "D_cut": args.D_cut,
        "lag_stride_frames": args.lag_stride_frames,
        "target_time_delays": str(
            target_time_delays(args.H_cut, args.horizon_steps, args.lag_stride_frames)
        ),
        "number_of_terms": len(metadata),
        "task_ridge_alpha": args.task_ridge_alpha,
        "ipc_ridge_alpha": args.ipc_ridge_alpha,
        "basis_nmse_x": basis_nmse_x,
        "basis_R2_x": 1.0 - basis_nmse_x,
        "basis_adequacy_label_x": basis_adequacy_label(basis_nmse_x),
        "c_alpha_x_sum": float(sum(float(w["c_alpha_x"]) for w in weights)),
        "Q_x_twIPC_raw": q_raw,
        "Q_x_twIPC_clipped": q_clipped,
        "generic_total_ipc_raw": total_ipc_raw,
        "generic_total_ipc_clipped": total_ipc_clipped,
    }
    actual = train_hidden_node_readout(
        sample_dir,
        args.hidden_node,
        args.reference_node,
        args.washout,
        args.train,
        args.test,
        args.actual_task_ridge_alpha,
        args.horizon_steps,
    )["metrics"]
    score.update(
        {
            "actual_task_train_nmse_x": actual["train_nmse_x"],
            "actual_task_test_nmse_x": actual["test_nmse_x"],
            "actual_task_test_nmse_y": actual["test_nmse_y"],
            "actual_task_test_nmse_2d": actual["test_nmse_2d"],
        }
    )
    actual_row = {
        "topology": record["topology"],
        "amplitude": record["amplitude"],
        "sample": record["sample"],
        "hidden_node": args.hidden_node,
        "reference_node": args.reference_node,
        "horizon_steps": args.horizon_steps,
        "target_component_used_for_ipc": "x",
        "ridge_alpha": args.actual_task_ridge_alpha,
        **actual,
    }
    grouped = grouped_contributions(weights, ipc, basis_nmse_x)
    return coeffs, weights, ipc, term_rows, grouped, score, actual_row


def summarize_scores(score_rows):
    summary = []
    for topology in sorted({r["topology"] for r in score_rows}, key=topology_index):
        group = [r for r in score_rows if r["topology"] == topology]
        row = {
            "topology_index": topology_index(topology),
            "topology": topology,
            "amplitude": group[0]["amplitude"],
            "target_component": "x",
            "num_samples": len(group),
        }
        for key in (
            "basis_nmse_x",
            "basis_R2_x",
            "c_alpha_x_sum",
            "Q_x_twIPC_raw",
            "Q_x_twIPC_clipped",
            "generic_total_ipc_clipped",
            "actual_task_test_nmse_x",
            "actual_task_test_nmse_y",
            "actual_task_test_nmse_2d",
        ):
            values = np.asarray([float(r[key]) for r in group], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(values))
            row[f"{key}_std"] = float(np.nanstd(values, ddof=0))
        summary.append(row)
    return summary


def summarize_grouped(grouped_rows):
    summary = []
    keys = sorted(
        {
            (r["topology"], r["amplitude"], r["group_type"], int(r["group_value"]))
            for r in grouped_rows
        },
        key=lambda item: (topology_index(item[0]), item[2], item[3]),
    )
    for topology, amplitude, group_type, group_value in keys:
        group = [
            r
            for r in grouped_rows
            if r["topology"] == topology
            and r["amplitude"] == amplitude
            and r["group_type"] == group_type
            and int(r["group_value"]) == group_value
        ]
        row = {
            "topology": topology,
            "amplitude": amplitude,
            "target_component": "x",
            "group_type": group_type,
            "group_value": group_value,
            "num_samples": len(group),
        }
        for key in ("c_alpha_x_sum", "IPC_alpha_clipped_sum", "weighted_contribution_x_sum"):
            values = np.asarray([float(r[key]) for r in group], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(values))
            row[f"{key}_std"] = float(np.nanstd(values, ddof=0))
        summary.append(row)
    return summary


def save_summary_plot(summary, out_path):
    labels = [str(r["topology_index"]) for r in summary]
    x = np.arange(len(summary))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(summary))]
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.8), constrained_layout=True)

    axes[0].bar(
        x,
        [r["Q_x_twIPC_clipped_mean"] for r in summary],
        yerr=[r["Q_x_twIPC_clipped_std"] for r in summary],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Q_x_twIPC")
    axes[0].set_title("X-only task-specific IPC; higher is better")
    axes[0].grid(axis="y", color="#E5E7EB", lw=0.6)

    axes[1].bar(
        x,
        [r["basis_nmse_x_mean"] for r in summary],
        yerr=[r["basis_nmse_x_std"] for r in summary],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("basis NMSE_x")
    axes[1].set_title("X-target Legendre expansion error; lower is better")
    axes[1].grid(axis="y", color="#E5E7EB", lw=0.6)

    axes[2].bar(
        x,
        [r["actual_task_test_nmse_x_mean"] for r in summary],
        yerr=[r["actual_task_test_nmse_x_std"] for r in summary],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel("topology index")
    axes[2].set_ylabel("actual test NMSE_x")
    axes[2].set_title("Actual hidden-node x prediction error; lower is better")
    axes[2].grid(axis="y", color="#E5E7EB", lw=0.6)

    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_metadata(args, records, skipped, excluded_samples, out_dir):
    payload = {
        "script": "run_xonly_task_specific_ipc_topology10_vs17.py",
        "purpose": "x-only task-specific IPC comparison; y target intentionally dropped",
        "task_weight_formula": (
            "c_alpha_x = a_alpha_x^2 * ||P_alpha - mean(P_alpha)||^2 / "
            "||x_target - mean(x_target)||^2"
        ),
        "Q_x_twIPC_formula": "sum_alpha c_alpha_x * IPC_alpha",
        "topologies": sorted({r["topology"] for r in records}, key=topology_index),
        "amplitudes_by_topology": {
            topology: sorted({r["amplitude"] for r in records if r["topology"] == topology})
            for topology in sorted({r["topology"] for r in records}, key=topology_index)
        },
        "hidden_node": args.hidden_node,
        "reference_node": args.reference_node,
        "horizon_steps": args.horizon_steps,
        "washout_s": args.washout,
        "train_s": args.train,
        "test_s": args.test,
        "H_cut": args.H_cut,
        "D_cut": args.D_cut,
        "lag_stride_frames": args.lag_stride_frames,
        "target_time_delays": target_time_delays(
            args.H_cut, args.horizon_steps, args.lag_stride_frames
        ),
        "task_ridge_alpha": args.task_ridge_alpha,
        "ipc_ridge_alpha": args.ipc_ridge_alpha,
        "actual_task_ridge_alpha": args.actual_task_ridge_alpha,
        "global_standardize": args.global_standardize,
        "skipped": skipped,
        "excluded_samples": {
            topology: sorted(samples) for topology, samples in excluded_samples.items()
        },
    }
    out_path = out_dir / "xonly_task_specific_ipc_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def write_readme(out_dir):
    text = """# X-Only Task-Specific IPC

This output compares task-specific IPC for the hidden-node x target only.
The hidden-node y target is intentionally excluded because its input-history
Legendre expansion was not accurate enough for the current data.

For each sample, the script:

1. Builds one common Legendre input-history dictionary.
2. Fits the x target as a centered expansion of that dictionary.
3. Computes task weights using c_alpha_x = a_alpha_x^2 ||P_alpha||^2 / ||x||^2.
4. Computes IPC_alpha by asking how well visible reservoir states reconstruct
   each basis term.
5. Computes Q_x_twIPC = sum_alpha c_alpha_x * IPC_alpha.
6. Computes the actual hidden-node readout NMSE_x for comparison.

Topology-level values are means and standard deviations across samples, not
one pooled fit across all samples.

The task weights are not renormalized. With strongly correlated measured input,
the finite Legendre basis terms are not orthogonal in the sampled data, so
sum_alpha c_alpha_x can exceed 1. This is a diagnostic warning from the data,
not a post-processing error.
"""
    (out_dir / "README_xonly_task_specific_ipc.txt").write_text(text)


def main():
    configure_matplotlib()
    args = parse_args()
    records, skipped, excluded_samples = discover_records(args)
    out_dir = output_dir(args, records, excluded_samples)

    print("X-only task-specific IPC comparison")
    print(
        f"  common dictionary: H={args.H_cut}, D={args.D_cut}, "
        f"target-time delays={target_time_delays(args.H_cut, args.horizon_steps, args.lag_stride_frames)}"
    )
    print("  y target is not used for task weights or Q_twIPC")
    print("  included samples:")
    for record in records:
        print(f"    {record['topology']}/{record['amplitude']}/{record['sample']}")

    coefficient_rows_all = []
    task_weight_rows_all = []
    ipc_rows_all = []
    term_rows_all = []
    grouped_rows_all = []
    score_rows = []
    actual_task_rows = []

    for record in records:
        print(f"-> X-only IPC: {record['topology']}/{record['amplitude']}/{record['sample']}")
        coeffs, weights, ipc, terms, grouped, score, actual = evaluate_sample(record, args)
        coefficient_rows_all.extend(coeffs)
        task_weight_rows_all.extend(weights)
        ipc_rows_all.extend(ipc)
        term_rows_all.extend(terms)
        grouped_rows_all.extend(grouped)
        score_rows.append(score)
        actual_task_rows.append(actual)

    summary = summarize_scores(score_rows)
    grouped_summary = summarize_grouped(grouped_rows_all)

    write_csv(coefficient_rows_all, out_dir / "xonly_task_coefficients_all_samples.csv")
    write_csv(task_weight_rows_all, out_dir / "xonly_task_weights_per_sample.csv")
    write_csv(ipc_rows_all, out_dir / "ipc_per_term_per_sample.csv")
    write_csv(term_rows_all, out_dir / "xonly_task_specific_ipc_per_term_per_sample.csv")
    write_csv(grouped_rows_all, out_dir / "xonly_task_specific_ipc_grouped_contributions.csv")
    write_csv(grouped_summary, out_dir / "xonly_task_specific_ipc_grouped_summary.csv")
    write_csv(score_rows, out_dir / "xonly_task_specific_ipc_per_sample.csv")
    write_csv(summary, out_dir / "xonly_task_specific_ipc_summary.csv")
    write_csv(actual_task_rows, out_dir / "actual_hidden_node_prediction_x_task_metrics.csv")
    write_csv(skipped, out_dir / "skipped_records.csv") if skipped else (out_dir / "skipped_records.csv").write_text(
        "topology,amplitude,sample,reason\n"
    )
    write_metadata(args, records, skipped, excluded_samples, out_dir)
    write_readme(out_dir)
    save_summary_plot(summary, out_dir / "xonly_task_specific_ipc_summary.pdf")

    print(f"Saved x-only task-specific IPC outputs to: {out_dir}")
    for row in summary:
        print(
            f"  topology {row['topology_index']} ({row['amplitude']}): "
            f"Q_x_twIPC={row['Q_x_twIPC_clipped_mean']:.4f} +/- "
            f"{row['Q_x_twIPC_clipped_std']:.4f}; "
            f"basis NMSE_x={row['basis_nmse_x_mean']:.4f}; "
            f"actual task NMSE_x={row['actual_task_test_nmse_x_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
