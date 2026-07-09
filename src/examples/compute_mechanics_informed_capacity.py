import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


current_dir = Path(__file__).parent
src_dir = current_dir.parent
repo_root = src_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(src_dir))

from openprc.reservoir.io.state_loader import StateLoader

from run_hidden_node_basis_selection import (
    DEFAULT_AMPLITUDE,
    DEFAULT_HIDDEN_NODE,
    DEFAULT_HORIZON_STEPS,
    DEFAULT_REFERENCE_NODE,
    DEFAULT_TOPOLOGY_FORMAT,
    DEFAULT_TRAIN_S,
    DEFAULT_VALIDATION_S,
    DEFAULT_WASHOUT_S,
    PALETTE,
    basis_label,
    configure_matplotlib,
    data_root,
    discover_records,
    frame_windows,
    load_positions,
    nmse_components,
    parse_exclusions,
    topology_name_from_arg,
    write_csv,
    write_json,
)
from run_hidden_node_mechanical_response_basis import (
    build_mechanical_features,
    expand_mechanical_products,
    mechanical_output_dir,
    standardize_drive_channels_from_train,
    standardize_features,
    svd_project,
)


DEFAULT_TOPOLOGIES = ["10", "17"]
DEFAULT_CAPACITY_ALPHA = 1e-6
DEFAULT_ACTUAL_READOUT_ALPHA = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute mechanics-informed task-specific capacity: task weights in the "
            "selected mechanical response basis multiplied by visible-state capacity "
            "to reconstruct each mechanical mode. This is not standard IPC."
        )
    )
    parser.add_argument("--topology", default=None)
    parser.add_argument("--topologies", nargs="+", default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--topology-start", type=int, default=None)
    parser.add_argument("--topology-stop", type=int, default=None)
    parser.add_argument("--topology-name-format", default=DEFAULT_TOPOLOGY_FORMAT)
    parser.add_argument("--amplitude", default=DEFAULT_AMPLITUDE)
    parser.add_argument("--topology-amplitude", action="append", default=[], metavar="TOPOLOGY:AMP")
    parser.add_argument("--sample", default="all")
    parser.add_argument("--exclude-sample", action="append", default=[], metavar="TOPOLOGY:SAMPLE")
    parser.add_argument("--hidden-node", type=int, default=DEFAULT_HIDDEN_NODE)
    parser.add_argument("--reference-node", type=int, default=DEFAULT_REFERENCE_NODE)
    parser.add_argument("--horizon-steps", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument("--washout", type=float, default=DEFAULT_WASHOUT_S)
    parser.add_argument("--train", type=float, default=DEFAULT_TRAIN_S)
    parser.add_argument("--validation", type=float, default=DEFAULT_VALIDATION_S)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--selected-dictionary", default=None)
    parser.add_argument("--mechanical-basis-output-dir", default=None)
    parser.add_argument("--capacity-alpha", type=float, default=DEFAULT_CAPACITY_ALPHA)
    parser.add_argument("--actual-readout-alpha", type=float, default=DEFAULT_ACTUAL_READOUT_ALPHA)
    parser.add_argument("--task-weight-alpha", type=float, default=None)
    parser.add_argument("--state-scaler", choices=["train", "global"], default="train")
    parser.add_argument("--state-components", nargs="+", choices=["x", "y"], default=["x", "y"])
    parser.add_argument(
        "--clip-negative-capacity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use max(0, validation R2) for weighted capacity while saving raw R2.",
    )
    parser.add_argument(
        "--capacity-target-window",
        choices=["validation", "train", "trainval"],
        default="validation",
    )
    parser.add_argument("--task-weight-window", choices=["train", "trainval"], default="train")
    parser.add_argument("--max-modes", type=int, default=None)
    parser.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def topology_index(topology):
    digits = "".join(ch for ch in str(topology) if ch.isdigit())
    return int(digits) if digits else 10**9


def output_dir(args, records, excluded_samples):
    if args.output_dir:
        return Path(args.output_dir)
    topologies = sorted({r["topology"] for r in records}, key=topology_index)
    label = "_vs_".join(topologies)
    exclude_label = ""
    if excluded_samples:
        parts = []
        for topology in sorted(excluded_samples, key=topology_index):
            for sample in sorted(excluded_samples[topology]):
                parts.append(f"{topology}_{sample}")
        exclude_label = "_exclude_" + "__".join(parts)
    components = "".join(args.state_components)
    return (
        data_root()
        / "task_specific_ipc"
        / "mechanics_informed_capacity"
        / f"{label}{exclude_label}"
        / f"hidden_node_{args.hidden_node}"
        / f"horizon_{args.horizon_steps}_steps"
        / f"state_{components}_{args.state_scaler}"
    )


def selected_dictionary_path_for_record(record, args, records, excluded_samples):
    if args.selected_dictionary:
        return Path(args.selected_dictionary)
    if args.mechanical_basis_output_dir:
        return Path(args.mechanical_basis_output_dir) / "selected_mechanical_response_dictionary.json"
    topology_records = [r for r in records if r["topology"] == record["topology"]]
    topology_exclusions = {
        record["topology"]: excluded_samples.get(record["topology"], set())
    } if excluded_samples.get(record["topology"]) else {}
    base = mechanical_output_dir(args, topology_records, topology_exclusions)
    path = base / "selected_mechanical_response_dictionary.json"
    if path.exists():
        return path
    multi_path = mechanical_output_dir(args, records, excluded_samples) / "selected_mechanical_response_dictionary.json"
    if multi_path.exists():
        return multi_path
    raise FileNotFoundError(
        f"No selected mechanical dictionary found for {record['topology']}. "
        f"Expected {path}. Run run_hidden_node_mechanical_response_basis.py for that topology "
        "or pass --selected-dictionary/--mechanical-basis-output-dir."
    )


def read_selected_dictionaries(args, records, excluded_samples):
    out = {}
    paths = {}
    for record in records:
        path = selected_dictionary_path_for_record(record, args, records, excluded_samples)
        if path not in out:
            out[path] = json.loads(path.read_text())
        paths[(record["topology"], record["sample"])] = path
    return out, paths


def validate_dictionary(record, args, selected):
    for arg_name, field in (
        ("hidden_node", "hidden_node"),
        ("reference_node", "reference_node"),
        ("horizon_steps", "horizon_steps"),
    ):
        if int(getattr(args, arg_name)) != int(selected[field]):
            raise ValueError(
                f"{record['topology']}/{record['sample']}: argument --{arg_name.replace('_', '-')}="
                f"{getattr(args, arg_name)} does not match selected dictionary {field}={selected[field]}."
            )


def frequencies_for_record(record, selected):
    key = f"{record['topology']}/{record['sample']}"
    selected_freqs = selected.get("selected_frequencies_hz", {})
    if isinstance(selected_freqs, dict) and key in selected_freqs:
        return [float(v) for v in selected_freqs[key]]
    if selected.get("selected_frequency_set_type") == "refined_single":
        freq = selected.get("selected_refined_frequency_hz")
        if freq is not None:
            return [float(freq)]
    if isinstance(selected_freqs, list):
        return [float(v) for v in selected_freqs]
    raise KeyError(
        f"Selected dictionary has no frequencies for {key}. For FFT-prefix dictionaries, "
        "run mechanical basis calibration on the same topology/sample set."
    )


def make_capacity_args(args, selected):
    class Obj:
        pass

    cap_args = Obj()
    cap_args.svd_rcond = 1e-8
    cap_args.svd_energy = 0.999
    cap_args.max_svd_modes = 50 if args.max_modes is None else int(args.max_modes)
    cap_args.mechanical_degree = int(selected.get("mechanical_degree", 1))
    cap_args.include_nonlinear_products = False
    cap_args.max_nonlinear_features = int(selected.get("number_candidate_features", 5000) * 2 + 1000)
    return cap_args


def visible_state_matrix(positions, rows, hidden_node, reference_node, components):
    n_nodes = positions.shape[1]
    visible_nodes = [n for n in range(n_nodes) if n not in {hidden_node, reference_node}]
    rel = positions[rows][:, visible_nodes, :] - positions[rows, reference_node, :][:, None, :]
    cols = []
    if "x" in components:
        cols.append(rel[:, :, 0])
    if "y" in components:
        cols.append(rel[:, :, 1])
    if not cols:
        raise ValueError("At least one state component is required.")
    X = np.concatenate(cols, axis=1)
    return X, visible_nodes


def standardize_state(X_train, X_val, scaler):
    if scaler == "global":
        print("Warning: global scaler is descriptive and leaks validation statistics.")
        ref = np.vstack([X_train, X_val])
    else:
        ref = X_train
    mean = np.mean(ref, axis=0, keepdims=True)
    std = np.std(ref, axis=0, keepdims=True)
    keep = std.reshape(-1) > 1e-10
    if not np.any(keep):
        raise ValueError("All visible-state dimensions have near-zero variance.")
    return (X_train[:, keep] - mean[:, keep]) / std[:, keep], (X_val[:, keep] - mean[:, keep]) / std[:, keep], keep


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= np.finfo(float).eps:
        return np.nan
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def fit_actual_readout(X_train, X_val, Y_train, Y_val, alpha):
    model = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
    model.fit(X_train, Y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_x, train_y, train_2d = nmse_components(Y_train, train_pred)
    val_x, val_y, val_2d = nmse_components(Y_val, val_pred)
    return model, train_pred, val_pred, train_x, train_y, train_2d, val_x, val_y, val_2d


def task_weights_from_q(Q_train, Q_val, Y_train, Y_val, alpha):
    y_mean = np.mean(Y_train, axis=0, keepdims=True)
    model = Ridge(alpha=alpha, fit_intercept=False, solver="svd")
    model.fit(Q_train, Y_train - y_mean)
    train_pred = model.predict(Q_train) + y_mean
    val_pred = model.predict(Q_val) + y_mean
    basis_train_x, basis_train_y, basis_train_2d = nmse_components(Y_train, train_pred)
    basis_x, basis_y, basis_2d = nmse_components(Y_val, val_pred)
    y_centered = Y_train - y_mean
    denom_x = float(np.sum(y_centered[:, 0] ** 2))
    denom_y = float(np.sum(y_centered[:, 1] ** 2))
    denom_2d = float(np.sum(y_centered**2))
    q_norm_sq = np.sum(Q_train**2, axis=0)
    coef_x = model.coef_[0]
    coef_y = model.coef_[1]
    weights = {
        "coef_x": coef_x,
        "coef_y": coef_y,
        "q_norm_sq": q_norm_sq,
        "c_x": coef_x**2 * q_norm_sq / max(denom_x, np.finfo(float).eps),
        "c_y": coef_y**2 * q_norm_sq / max(denom_y, np.finfo(float).eps),
        "c_2d": (coef_x**2 + coef_y**2) * q_norm_sq / max(denom_2d, np.finfo(float).eps),
    }
    return model, train_pred, val_pred, weights, basis_train_x, basis_train_y, basis_train_2d, basis_x, basis_y, basis_2d


def reconstruct_mechanical_basis(record, args, selected):
    sample_dir = record["sample_dir"]
    loader = StateLoader(sample_dir / "experiment.h5")
    time, positions = load_positions(sample_dir / "experiment.h5")
    u = loader.get_actuation_signal(actuator_idx=0, dof=0)
    hidden_relative = positions[:, args.hidden_node, :] - positions[:, args.reference_node, :]
    train_start, train_stop, validation_stop = frame_windows(loader, args)
    base_train_rows = np.arange(train_start, train_stop)
    selected_channels = selected.get("selected_drive_channels", ["u"])
    if selected_channels != ["u"]:
        raise NotImplementedError(
            f"Only displacement-like selected_drive_channels ['u'] are supported, got {selected_channels}."
        )
    drives = {"u": np.asarray(u, dtype=float)}
    drives_z, drive_stats = standardize_drive_channels_from_train(drives, base_train_rows, selected_channels)
    frequencies = frequencies_for_record(record, selected)
    filter_length_s = float(selected["selected_filter_length_s"])
    Psi_raw, raw_meta, L_frames = build_mechanical_features(
        drives_z, loader.dt, frequencies, selected["decay_times_s"], filter_length_s
    )
    train_rows = np.arange(max(train_start, L_frames), train_stop)
    val_rows = np.arange(max(train_stop, L_frames), validation_stop)
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise ValueError(f"{record['topology']}/{record['sample']}: no valid rows after filter length.")
    Psi_linear, keep = standardize_features(Psi_raw, train_rows)
    kept_linear_meta = [meta for meta, use in zip(raw_meta, keep) if use]
    cap_args = make_capacity_args(args, selected)
    Psi, kept_meta, number_linear_features, number_candidate_features = expand_mechanical_products(
        Psi_linear, kept_linear_meta, train_rows, cap_args
    )
    Psi_train = Psi[train_rows]
    Psi_val = Psi[val_rows]
    Q_train, Q_val, singular_values, Vt, rank, condition_number = svd_project(Psi_train, Psi_val, cap_args)
    if args.max_modes is not None:
        rank = min(rank, int(args.max_modes), Q_train.shape[1])
        Q_train = Q_train[:, :rank]
        Q_val = Q_val[:, :rank]
        Vt = Vt[:rank, :]
        singular_values = singular_values[:rank]
    Y_train = hidden_relative[train_rows + args.horizon_steps]
    Y_val = hidden_relative[val_rows + args.horizon_steps]
    return {
        "loader": loader,
        "time": time,
        "positions": positions,
        "hidden_relative": hidden_relative,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "Y_train": Y_train,
        "Y_val": Y_val,
        "Q_train": Q_train,
        "Q_val": Q_val,
        "Vt": Vt,
        "singular_values": singular_values,
        "rank": rank,
        "condition_number": condition_number,
        "kept_meta": kept_meta,
        "number_linear_features": number_linear_features,
        "number_candidate_features": number_candidate_features,
        "number_features": len(kept_meta),
        "L_frames": L_frames,
        "frequencies": frequencies,
        "filter_length_s": filter_length_s,
        "drive_stats": drive_stats,
    }


def evaluate_sample(record, args, selected, dictionary_path):
    validate_dictionary(record, args, selected)
    basis = reconstruct_mechanical_basis(record, args, selected)
    X_train_raw, visible_nodes = visible_state_matrix(
        basis["positions"], basis["train_rows"], args.hidden_node, args.reference_node, args.state_components
    )
    X_val_raw, _ = visible_state_matrix(
        basis["positions"], basis["val_rows"], args.hidden_node, args.reference_node, args.state_components
    )
    X_train, X_val, state_keep = standardize_state(X_train_raw, X_val_raw, args.state_scaler)
    if args.task_weight_window == "trainval":
        Q_weight = np.vstack([basis["Q_train"], basis["Q_val"]])
        Y_weight = np.vstack([basis["Y_train"], basis["Y_val"]])
    else:
        Q_weight = basis["Q_train"]
        Y_weight = basis["Y_train"]
    task_alpha = float(args.task_weight_alpha if args.task_weight_alpha is not None else selected.get("ridge_alpha", 1e-6))
    task_model, _, basis_val_pred, weights, basis_train_x, basis_train_y, basis_train_2d, basis_x, basis_y, basis_2d = (
        task_weights_from_q(Q_weight, basis["Q_val"], Y_weight, basis["Y_val"], task_alpha)
    )
    actual_model, _, actual_val_pred, actual_train_x, actual_train_y, actual_train_2d, actual_x, actual_y, actual_2d = (
        fit_actual_readout(X_train, X_val, basis["Y_train"], basis["Y_val"], args.actual_readout_alpha)
    )

    per_mode = []
    capacities = []
    validation_capacities = []
    for j in range(basis["rank"]):
        q_train = basis["Q_train"][:, j]
        q_val = basis["Q_val"][:, j]
        model = Ridge(alpha=args.capacity_alpha, fit_intercept=True, solver="svd")
        model.fit(X_train, q_train)
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        raw_train = r2_score(q_train, pred_train)
        raw_val = r2_score(q_val, pred_val)
        raw_trainval = r2_score(np.concatenate([q_train, q_val]), np.concatenate([pred_train, pred_val]))
        if args.capacity_target_window == "train":
            raw_for_capacity = raw_train
        elif args.capacity_target_window == "trainval":
            raw_for_capacity = raw_trainval
        else:
            raw_for_capacity = raw_val
        cap_used = 0.0 if not np.isfinite(raw_for_capacity) else float(
            max(0.0, raw_for_capacity) if args.clip_negative_capacity else raw_for_capacity
        )
        cap_val = 0.0 if not np.isfinite(raw_val) else float(
            max(0.0, raw_val) if args.clip_negative_capacity else raw_val
        )
        capacities.append(cap_used)
        validation_capacities.append(cap_val)
        dominant_idx = int(np.argmax(np.abs(basis["Vt"][j, :])))
        dominant = basis["kept_meta"][dominant_idx]
        row = {
            "topology": record["topology"],
            "amplitude": record["amplitude"],
            "sample": record["sample"],
            "mode_index": j,
            "raw_R2_train": raw_train,
            "raw_R2_validation": raw_val,
            "raw_R2_trainval": raw_trainval,
            "capacity_validation": cap_val,
            "capacity_used": cap_used,
            "capacity_target_window": args.capacity_target_window,
            "c_mech_x": float(weights["c_x"][j]),
            "c_mech_y": float(weights["c_y"][j]),
            "c_mech_2d": float(weights["c_2d"][j]),
            "contribution_x": float(weights["c_x"][j] * cap_used),
            "contribution_y": float(weights["c_y"][j] * cap_used),
            "contribution_2d": float(weights["c_2d"][j] * cap_used),
            "dominant_raw_feature_name": dominant["raw_feature_name"],
            "dominant_product_degree": dominant.get("product_degree", 1),
            "dominant_frequency_hz": dominant["frequency_hz"],
            "dominant_decay_time_s": dominant["decay_time_s"],
            "dominant_phase": dominant["phase"],
            "dominant_raw_feature_loading": float(basis["Vt"][j, dominant_idx]),
        }
        per_mode.append(row)
    capacities = np.asarray(capacities, dtype=float)
    validation_capacities = np.asarray(validation_capacities, dtype=float)
    sum_c_x = float(np.sum(weights["c_x"]))
    sum_c_y = float(np.sum(weights["c_y"]))
    sum_c_2d = float(np.sum(weights["c_2d"]))
    q_x = float(np.sum(weights["c_x"] * capacities))
    q_y = float(np.sum(weights["c_y"] * capacities))
    q_2d = float(np.sum(weights["c_2d"] * capacities))
    summary = {
        "topology": record["topology"],
        "amplitude": record["amplitude"],
        "sample": record["sample"],
        "basis_nmse_x": basis_x,
        "basis_nmse_y": basis_y,
        "basis_nmse_2d": basis_2d,
        "basis_R2_x": 1.0 - basis_x,
        "basis_R2_y": 1.0 - basis_y,
        "basis_R2_2d": 1.0 - basis_2d,
        "actual_nmse_x": actual_x,
        "actual_nmse_y": actual_y,
        "actual_nmse_2d": actual_2d,
        "actual_train_nmse_x": actual_train_x,
        "actual_train_nmse_y": actual_train_y,
        "actual_train_nmse_2d": actual_train_2d,
        "sum_c_x": sum_c_x,
        "sum_c_y": sum_c_y,
        "sum_c_2d": sum_c_2d,
        "Q_mech_x_abs": q_x,
        "Q_mech_y_abs": q_y,
        "Q_mech_2d_abs": q_2d,
        "Q_mech_x_norm": q_x / max(sum_c_x, np.finfo(float).eps),
        "Q_mech_y_norm": q_y / max(sum_c_y, np.finfo(float).eps),
        "Q_mech_2d_norm": q_2d / max(sum_c_2d, np.finfo(float).eps),
        "mean_capacity_validation": float(np.nanmean(validation_capacities)) if len(validation_capacities) else np.nan,
        "median_capacity_validation": float(np.nanmedian(validation_capacities)) if len(validation_capacities) else np.nan,
        "max_capacity_validation": float(np.nanmax(validation_capacities)) if len(validation_capacities) else np.nan,
        "mean_capacity_used": float(np.nanmean(capacities)) if len(capacities) else np.nan,
        "median_capacity_used": float(np.nanmedian(capacities)) if len(capacities) else np.nan,
        "max_capacity_used": float(np.nanmax(capacities)) if len(capacities) else np.nan,
        "capacity_target_window": args.capacity_target_window,
        "num_modes": int(basis["rank"]),
        "num_features": int(basis["number_features"]),
        "number_linear_features": int(basis["number_linear_features"]),
        "number_candidate_features": int(basis["number_candidate_features"]),
        "svd_rank": int(basis["rank"]),
        "condition_number": float(basis["condition_number"]),
        "selected_filter_length_s": float(basis["filter_length_s"]),
        "selected_filter_length_frames": int(basis["L_frames"]),
        "selected_frequencies_hz": " ".join(f"{f:.8g}" for f in basis["frequencies"]),
        "mechanical_degree": int(selected.get("mechanical_degree", 1)),
        "selected_dictionary_path": str(dictionary_path),
        "num_state_nodes": int(len(visible_nodes)),
        "num_state_features": int(np.sum(state_keep)),
        "basis_adequacy_label_x": basis_label(basis_x),
        "basis_adequacy_label_y": basis_label(basis_y),
        "basis_adequacy_label_2d": basis_label(basis_2d),
    }
    trace = {
        "topology": record["topology"],
        "amplitude": record["amplitude"],
        "sample": record["sample"],
        "time_s": basis["time"][basis["val_rows"] + args.horizon_steps]
        - basis["time"][basis["val_rows"][0] + args.horizon_steps],
        "target": basis["Y_val"],
        "mechanical_basis_fit": basis_val_pred,
        "actual_readout": actual_val_pred,
    }
    return summary, per_mode, trace


def summarize_by_topology(rows):
    out = []
    for topology in sorted({r["topology"] for r in rows}, key=topology_index):
        group = [r for r in rows if r["topology"] == topology]
        row = {
            "topology_index": topology_index(topology),
            "topology": topology,
            "amplitude": group[0]["amplitude"],
            "num_samples": len(group),
        }
        for key in (
            "basis_nmse_x",
            "basis_nmse_y",
            "basis_nmse_2d",
            "actual_nmse_x",
            "actual_nmse_y",
            "actual_nmse_2d",
            "Q_mech_x_abs",
            "Q_mech_y_abs",
            "Q_mech_2d_abs",
            "Q_mech_x_norm",
            "Q_mech_y_norm",
            "Q_mech_2d_norm",
            "mean_capacity_validation",
            "mean_capacity_used",
            "sum_c_x",
            "sum_c_y",
            "sum_c_2d",
        ):
            vals = np.asarray([float(r[key]) for r in group], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals))
            row[f"{key}_std"] = float(np.nanstd(vals, ddof=0))
        out.append(row)
    return out


def save_with_companions(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label):
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=9)


def paired_topology_bars(ax, topology_rows, x_key, y_key, x_err_key, y_err_key, ylabel, title):
    x = np.arange(len(topology_rows))
    labels = [str(r["topology_index"]) for r in topology_rows]
    ax.bar(
        x - 0.18,
        [r[x_key] for r in topology_rows],
        yerr=[r[x_err_key] for r in topology_rows],
        width=0.36,
        color="#8FBAD9",
        edgecolor="white",
        linewidth=0.6,
        label="x",
    )
    ax.bar(
        x + 0.18,
        [r[y_key] for r in topology_rows],
        yerr=[r[y_err_key] for r in topology_rows],
        width=0.36,
        color="#8DD3C7",
        edgecolor="white",
        linewidth=0.6,
        label="y",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    ax.legend(fontsize=7)


def save_summary_plot(topology_rows, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.8), constrained_layout=True)
    paired_topology_bars(
        axes[0],
        topology_rows,
        "Q_mech_x_abs_mean",
        "Q_mech_y_abs_mean",
        "Q_mech_x_abs_std",
        "Q_mech_y_abs_std",
        "Q_mech",
        "Mechanics-informed task-specific capacity; higher is better",
    )
    paired_topology_bars(
        axes[1],
        topology_rows,
        "basis_nmse_x_mean",
        "basis_nmse_y_mean",
        "basis_nmse_x_std",
        "basis_nmse_y_std",
        "basis NMSE",
        "Mechanical response basis error; lower is better",
    )
    paired_topology_bars(
        axes[2],
        topology_rows,
        "actual_nmse_x_mean",
        "actual_nmse_y_mean",
        "actual_nmse_x_std",
        "actual_nmse_y_std",
        "actual validation NMSE",
        "Actual hidden-node prediction error; lower is better",
    )
    axes[2].set_xlabel("topology index")
    for label, ax in zip(("a", "b", "c"), axes):
        panel_label(ax, label)
    save_with_companions(fig, out_path)


def representative_trace_rows(trace):
    rows = []
    for i, t in enumerate(trace["time_s"]):
        rows.append(
            {
                "topology": trace["topology"],
                "amplitude": trace["amplitude"],
                "sample": trace["sample"],
                "time_s": float(t),
                "target_y": float(trace["target"][i, 1]),
                "mechanical_basis_fit_y": float(trace["mechanical_basis_fit"][i, 1]),
                "actual_readout_y": float(trace["actual_readout"][i, 1]),
                "target_x": float(trace["target"][i, 0]),
                "mechanical_basis_fit_x": float(trace["mechanical_basis_fit"][i, 0]),
                "actual_readout_x": float(trace["actual_readout"][i, 0]),
            }
        )
    return rows


def plot_points_rows(sample_rows):
    return [
        {
            "topology": r["topology"],
            "amplitude": r["amplitude"],
            "sample": r["sample"],
            "component": comp,
            "Q_mech_abs": r[f"Q_mech_{comp}_abs"],
            "Q_mech_norm": r[f"Q_mech_{comp}_norm"],
            "actual_nmse": r[f"actual_nmse_{comp}"],
            "basis_R2": r[f"basis_R2_{comp}"],
            "basis_nmse": r[f"basis_nmse_{comp}"],
        }
        for r in sample_rows
        for comp in ("x", "y")
    ]


def interpretation(sample_rows):
    mean_basis_y = float(np.nanmean([r["basis_nmse_y"] for r in sample_rows]))
    mean_actual_y = float(np.nanmean([r["actual_nmse_y"] for r in sample_rows]))
    mean_q_y = float(np.nanmean([r["Q_mech_y_abs"] for r in sample_rows]))
    if mean_basis_y > 0.5 and mean_actual_y < 0.2:
        return (
            "Actual y prediction is much better than the input-driven mechanical basis "
            "representation. This suggests a strong state-observability component beyond "
            "mechanics-informed input capacity."
        )
    if mean_basis_y <= 0.5 and mean_q_y > 0.2 and mean_actual_y < 0.2:
        return "Mechanics-informed task-specific capacity explains a substantial portion of y-task performance."
    if mean_basis_y > 0.5 and mean_actual_y > 0.5:
        return "The selected morphology does not provide sufficient task-relevant capacity or observability for this target."
    return "Mechanics-informed capacity, basis adequacy, and actual readout performance show mixed evidence; inspect x/y panels and per-mode contributions."


def write_results_json(args, records, selected_paths, sample_rows, out_dir):
    interp = interpretation(sample_rows)
    payload = {
        "script": "compute_mechanics_informed_capacity.py",
        "name": "mechanics-informed task-specific capacity",
        "note": "This is not standard IPC.",
        "selected_dictionary_paths": sorted({str(p) for p in selected_paths.values()}),
        "topologies": sorted({r["topology"] for r in records}, key=topology_index),
        "hidden_node": args.hidden_node,
        "reference_node": args.reference_node,
        "horizon_steps": args.horizon_steps,
        "washout_s": args.washout,
        "train_s": args.train,
        "validation_s": args.validation,
        "state_components": args.state_components,
        "state_scaler": args.state_scaler,
        "capacity_alpha": args.capacity_alpha,
        "capacity_target_window": args.capacity_target_window,
        "task_weight_window": args.task_weight_window,
        "actual_readout_alpha": args.actual_readout_alpha,
        "clip_negative_capacity": args.clip_negative_capacity,
        "mean_Q_mech_x_abs": float(np.nanmean([r["Q_mech_x_abs"] for r in sample_rows])),
        "mean_Q_mech_y_abs": float(np.nanmean([r["Q_mech_y_abs"] for r in sample_rows])),
        "mean_Q_mech_y_norm": float(np.nanmean([r["Q_mech_y_norm"] for r in sample_rows])),
        "mean_actual_nmse_x": float(np.nanmean([r["actual_nmse_x"] for r in sample_rows])),
        "mean_actual_nmse_y": float(np.nanmean([r["actual_nmse_y"] for r in sample_rows])),
        "mean_basis_nmse_x": float(np.nanmean([r["basis_nmse_x"] for r in sample_rows])),
        "mean_basis_nmse_y": float(np.nanmean([r["basis_nmse_y"] for r in sample_rows])),
        "interpretation": interp,
    }
    write_json(payload, out_dir / "mechanics_informed_capacity_results.json")


def main():
    configure_matplotlib()
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    args = parse_args()
    records, skipped, excluded_samples = discover_records(args)
    out_dir = output_dir(args, records, excluded_samples)
    selected_by_path, selected_paths = read_selected_dictionaries(args, records, excluded_samples)
    print("Mechanics-informed task-specific capacity")
    print("  This is not standard IPC.")
    print(f"  state components: {args.state_components}; scaler={args.state_scaler}")
    print("  included samples:")
    for record in records:
        print(f"    {record['topology']}/{record['amplitude']}/{record['sample']}")

    sample_rows = []
    mode_rows = []
    traces = []
    for record in records:
        selected_path = selected_paths[(record["topology"], record["sample"])]
        selected = selected_by_path[selected_path]
        print(f"-> Capacity: {record['topology']}/{record['amplitude']}/{record['sample']}")
        summary, per_mode, trace = evaluate_sample(record, args, selected, selected_path)
        sample_rows.append(summary)
        mode_rows.extend(per_mode)
        traces.append(trace)

    topology_rows = summarize_by_topology(sample_rows)
    representative = min(sample_rows, key=lambda r: r["actual_nmse_y"])
    trace = next(
        t for t in traces if t["topology"] == representative["topology"] and t["sample"] == representative["sample"]
    )
    write_csv(mode_rows, out_dir / "mechanics_informed_capacity_per_mode.csv")
    write_csv(sample_rows, out_dir / "mechanics_informed_capacity_summary.csv")
    write_csv(topology_rows, out_dir / "mechanics_informed_capacity_by_topology.csv")
    write_csv(plot_points_rows(sample_rows), out_dir / "mechanics_informed_capacity_plot_points.csv")
    write_csv(representative_trace_rows(trace), out_dir / "representative_capacity_trace.csv")
    write_csv(skipped, out_dir / "skipped_records.csv")
    write_results_json(args, records, selected_paths, sample_rows, out_dir)
    if args.make_plots:
        save_summary_plot(
            topology_rows,
            out_dir / "mechanics_informed_capacity_summary.pdf",
        )
    print(f"Saved mechanics-informed capacity outputs to: {out_dir}")
    for row in topology_rows:
        print(
            f"  topology {row['topology_index']}: "
            f"Q_x={row['Q_mech_x_abs_mean']:.4f}, Q_y={row['Q_mech_y_abs_mean']:.4f}; "
            f"actual NMSE x/y={row['actual_nmse_x_mean']:.4f}/{row['actual_nmse_y_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
