import argparse
import csv
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
sys.path.insert(0, str(src_dir))

from openprc.reservoir.io.state_loader import StateLoader


DEFAULT_TOPOLOGY = "topology_17_prestress"
DEFAULT_TOPOLOGY_FORMAT = "topology_{i}_prestress"
DEFAULT_AMPLITUDE = "auto"
DEFAULT_HIDDEN_NODE = 10
DEFAULT_REFERENCE_NODE = 0
DEFAULT_HORIZON_STEPS = 5
DEFAULT_WASHOUT_S = 5.0
DEFAULT_TRAIN_S = 10.0
DEFAULT_VALIDATION_S = 10.0
DEFAULT_H_MIN = 0
DEFAULT_H_MAX = 30
DEFAULT_D_VALUES = [1, 2]
DEFAULT_LAG_STRIDE_FRAMES = 1
DEFAULT_RIDGE_ALPHA = 1e-6
DEFAULT_TOLERANCE_FRAC = 0.05
DEFAULT_CONDITION_THRESHOLD = 1e12

PALETTE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
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
            "PDF-style hidden-node input-history basis calibration. The script fits "
            "Legendre input-history models on the train window and selects H,D by "
            "validation-window NMSE. It can calibrate one topology or a chosen "
            "topology family on one common dictionary."
        )
    )
    parser.add_argument(
        "--topology",
        default=None,
        help="Single topology name. Kept for convenience; equivalent to --topologies.",
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=None,
        help="Topology names or numeric indices, e.g. topology_17_prestress or 17.",
    )
    parser.add_argument("--topology-start", type=int, default=None)
    parser.add_argument("--topology-stop", type=int, default=None)
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
        help="Override amplitude for one topology, e.g. topology_17_prestress:amp=1.",
    )
    parser.add_argument("--sample", default="all")
    parser.add_argument(
        "--exclude-sample",
        action="append",
        default=[],
        metavar="TOPOLOGY:SAMPLE",
        help="Exclude one sample, e.g. topology_17_prestress:sample_1. Can repeat.",
    )
    parser.add_argument("--hidden-node", type=int, default=DEFAULT_HIDDEN_NODE)
    parser.add_argument("--reference-node", type=int, default=DEFAULT_REFERENCE_NODE)
    parser.add_argument("--horizon-steps", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument("--washout", type=float, default=DEFAULT_WASHOUT_S)
    parser.add_argument("--train", type=float, default=DEFAULT_TRAIN_S)
    parser.add_argument("--validation", type=float, default=DEFAULT_VALIDATION_S)
    parser.add_argument("--H-min", type=int, default=DEFAULT_H_MIN)
    parser.add_argument("--H-max", type=int, default=DEFAULT_H_MAX)
    parser.add_argument("--Q-min", type=int, dest="H_min", help=argparse.SUPPRESS)
    parser.add_argument("--Q-max", type=int, dest="H_max", help=argparse.SUPPRESS)
    parser.add_argument("--D-values", type=int, nargs="+", default=DEFAULT_D_VALUES)
    parser.add_argument("--lag-stride-frames", type=int, default=DEFAULT_LAG_STRIDE_FRAMES)
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument(
        "--tolerance-frac",
        type=float,
        default=DEFAULT_TOLERANCE_FRAC,
        help="Choose the simplest candidate within this fraction of the best validation NMSE.",
    )
    parser.add_argument(
        "--condition-threshold",
        type=float,
        default=DEFAULT_CONDITION_THRESHOLD,
        help="Prefer candidates below this condition number when possible.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def data_root():
    return src_dir.parent / "data" / "experiment_data"


def parse_key_value_pairs(values):
    out = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"Expected TOPOLOGY:VALUE format, got '{value}'.")
        key, val = [part.strip() for part in value.split(":", 1)]
        if not key or not val:
            raise ValueError(f"Expected TOPOLOGY:VALUE format, got '{value}'.")
        out[key] = val
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


def topology_name_from_arg(value, name_format):
    text = str(value)
    if text.isdigit():
        return name_format.format(i=int(text))
    return text


def selected_topologies(args):
    if args.topologies:
        return [topology_name_from_arg(v, args.topology_name_format) for v in args.topologies]
    if args.topology is not None:
        return [topology_name_from_arg(args.topology, args.topology_name_format)]
    if args.topology_start is not None or args.topology_stop is not None:
        if args.topology_start is None or args.topology_stop is None:
            raise ValueError("Use both --topology-start and --topology-stop.")
        return [
            args.topology_name_format.format(i=i)
            for i in range(args.topology_start, args.topology_stop + 1)
        ]
    return [DEFAULT_TOPOLOGY]


def choose_amplitude_dir(topology_dir, amplitude):
    if amplitude != "auto":
        amp_dir = topology_dir / amplitude
        if not amp_dir.exists():
            return None
        return amp_dir
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
    overrides = parse_key_value_pairs(args.topology_amplitude)
    excluded_samples = parse_exclusions(args.exclude_sample, args.topology_name_format)
    records = []
    skipped = []
    for topology in selected_topologies(args):
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
        amplitude = overrides.get(topology, args.amplitude)
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
        raise FileNotFoundError("No topology/sample records were found for the requested selection.")
    return records, skipped, excluded_samples


def load_positions(h5_path):
    with h5py.File(h5_path, "r") as f:
        time = f["time_series/time"][:]
        positions = f["time_series/nodes/positions"][:, :, :2]
    return time, positions


def normalize_to_unit_interval(u):
    u = np.asarray(u, dtype=float).reshape(-1)
    u_min = float(np.min(u))
    u_max = float(np.max(u))
    if abs(u_max - u_min) < 1e-12:
        raise ValueError("Measured input is constant; cannot normalize to [-1, 1].")
    return 2.0 * (u - u_min) / (u_max - u_min) - 1.0


def legendre_normalized(n, x):
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return np.sqrt(3.0) * x
    p_nm2 = np.ones_like(x)
    p_nm1 = x
    for k in range(1, n):
        p_n = ((2 * k + 1) * x * p_nm1 - k * p_nm2) / (k + 1)
        p_nm2, p_nm1 = p_nm1, p_n
    return np.sqrt(2 * n + 1) * p_nm1


def exponent_vectors(num_lags, max_degree):
    exps = []
    vec = np.zeros(num_lags, dtype=np.int16)

    def rec(remaining, idx):
        if idx == num_lags - 1:
            vec[idx] = remaining
            exps.append(vec.copy())
            return
        for value in range(remaining + 1):
            vec[idx] = value
            rec(remaining - value, idx + 1)

    for degree in range(1, max_degree + 1):
        rec(degree, 0)
    return np.asarray(exps, dtype=np.int16)


def build_legendre_design(u_norm, H, D, lag_stride_frames):
    num_lags = H + 1
    exps = exponent_vectors(num_lags, D)
    n_rows = len(u_norm)
    design = np.empty((n_rows, len(exps)), dtype=np.float32)

    values_by_degree = {
        degree: legendre_normalized(degree, u_norm).astype(np.float32)
        for degree in range(D + 1)
    }
    lagged_values = {}
    for degree in range(D + 1):
        lagged = np.empty((n_rows, num_lags), dtype=np.float32)
        lagged[:, :] = np.nan
        source = values_by_degree[degree]
        for q in range(num_lags):
            lag = q * lag_stride_frames
            if lag == 0:
                lagged[:, q] = source
            else:
                lagged[lag:, q] = source[:-lag]
        lagged_values[degree] = lagged

    for col, exp_vec in enumerate(exps):
        term = np.ones(n_rows, dtype=np.float32)
        for q, degree in enumerate(exp_vec):
            if degree > 0:
                term *= lagged_values[int(degree)][:, q]
        design[:, col] = term
    return design, exps


def term_name(exp_vec, horizon_steps, lag_stride_frames):
    parts = []
    for q, degree in enumerate(exp_vec):
        if degree > 0:
            target_delay = horizon_steps + q * lag_stride_frames
            parts.append(f"P{int(degree)}(u[target-{target_delay}])")
    return " * ".join(parts)


def target_time_delays(H, horizon_steps, lag_stride_frames):
    return [horizon_steps + q * lag_stride_frames for q in range(H + 1)]


def nmse_components(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    err = y_true - y_pred
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    sse = np.sum(err**2, axis=0)
    sst = np.sum(centered**2, axis=0)
    nmse_xy = sse / np.maximum(sst, np.finfo(float).eps)
    nmse_2d = float(np.sum(err**2) / np.maximum(np.sum(centered**2), np.finfo(float).eps))
    return float(nmse_xy[0]), float(nmse_xy[1]), nmse_2d


def fit_centered_ridge(P_train, Y_train, alpha):
    p_mean = np.mean(P_train, axis=0, keepdims=True)
    y_mean = np.mean(Y_train, axis=0, keepdims=True)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(P_train - p_mean, Y_train - y_mean)
    return model, p_mean, y_mean


def predict_centered(model, p_mean, y_mean, P):
    return model.predict(P - p_mean) + y_mean


def frame_windows(loader, args):
    washout_frames = int(args.washout / loader.dt)
    train_frames = int(args.train / loader.dt)
    validation_frames = int(args.validation / loader.dt)
    train_start = washout_frames
    train_stop = train_start + train_frames
    validation_stop = train_stop + validation_frames
    if validation_stop + args.horizon_steps > loader.total_frames:
        raise ValueError(
            f"Need target frame {validation_stop + args.horizon_steps}, "
            f"but sample has only {loader.total_frames} frames."
        )
    return train_start, train_stop, validation_stop


def valid_rows(start, stop, H, lag_stride_frames):
    valid_start = max(start, H * lag_stride_frames)
    if valid_start >= stop:
        return np.asarray([], dtype=int)
    return np.arange(valid_start, stop)


def design_condition_number(P_train_centered, ridge_alpha):
    gram = P_train_centered.T @ P_train_centered
    if ridge_alpha > 0:
        gram = gram + ridge_alpha * np.eye(gram.shape[0])
    try:
        return float(np.linalg.cond(gram))
    except np.linalg.LinAlgError:
        return float("inf")


def evaluate_candidate(record, args, H, D):
    sample_dir = record["sample_dir"]
    loader = StateLoader(sample_dir / "experiment.h5")
    _, positions = load_positions(sample_dir / "experiment.h5")
    n_nodes = positions.shape[1]
    if args.hidden_node >= n_nodes:
        raise ValueError(
            f"{record['topology']}/{record['sample']}: hidden node {args.hidden_node} "
            f"is outside node range 0..{n_nodes - 1}."
        )
    if args.hidden_node == args.reference_node:
        raise ValueError("Hidden node cannot also be the reference node.")

    u_norm = normalize_to_unit_interval(loader.get_actuation_signal(actuator_idx=0, dof=0))
    hidden_relative = positions[:, args.hidden_node, :] - positions[:, args.reference_node, :]
    train_start, train_stop, validation_stop = frame_windows(loader, args)
    design, exps = build_legendre_design(u_norm, H, D, args.lag_stride_frames)

    train_rows = valid_rows(train_start, train_stop, H, args.lag_stride_frames)
    val_rows = valid_rows(train_stop, validation_stop, H, args.lag_stride_frames)
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise ValueError(f"No valid train/validation rows for H={H}, D={D}.")
    P_train = design[train_rows]
    P_val = design[val_rows]
    Y_train = hidden_relative[train_rows + args.horizon_steps]
    Y_val = hidden_relative[val_rows + args.horizon_steps]
    if np.isnan(P_train).any() or np.isnan(P_val).any():
        raise ValueError(f"NaNs remained in design matrix for H={H}, D={D}.")

    model, p_mean, y_mean = fit_centered_ridge(P_train, Y_train, args.ridge_alpha)
    train_pred = predict_centered(model, p_mean, y_mean, P_train)
    val_pred = predict_centered(model, p_mean, y_mean, P_val)
    train_x, train_y, train_2d = nmse_components(Y_train, train_pred)
    val_x, val_y, val_2d = nmse_components(Y_val, val_pred)
    condition_number = design_condition_number(P_train - p_mean, args.ridge_alpha)
    number_of_terms = int(design.shape[1])
    overparameterized = number_of_terms > 0.8 * len(train_rows)
    ill_conditioned = condition_number > args.condition_threshold
    return {
        "topology": record["topology"],
        "amplitude": record["amplitude"],
        "sample": record["sample"],
        "H": H,
        "D": D,
        "horizon_steps": args.horizon_steps,
        "lag_stride_frames": args.lag_stride_frames,
        "readout_time_delays": str(list(range(H + 1))),
        "target_time_delays": str(target_time_delays(H, args.horizon_steps, args.lag_stride_frames)),
        "number_of_terms": number_of_terms,
        "num_train_rows": int(len(train_rows)),
        "num_validation_rows": int(len(val_rows)),
        "terms_per_train_row": float(number_of_terms / max(len(train_rows), 1)),
        "condition_number": condition_number,
        "overparameterized": overparameterized,
        "ill_conditioned": ill_conditioned,
        "condition_warning": "ill_conditioned" if ill_conditioned else "",
        "train_nmse_x": train_x,
        "train_nmse_y": train_y,
        "train_nmse_2d": train_2d,
        "validation_nmse_x": val_x,
        "validation_nmse_y": val_y,
        "validation_nmse_2d": val_2d,
        "train_validation_gap_2d": val_2d - train_2d,
    }


def candidate_pairs(args):
    if args.H_min < 0 or args.H_max < args.H_min:
        raise ValueError("Require 0 <= H-min <= H-max.")
    if args.lag_stride_frames < 1:
        raise ValueError("--lag-stride-frames must be at least 1.")
    return [(H, D) for D in sorted(set(args.D_values)) for H in range(args.H_min, args.H_max + 1)]


def summarize_candidates(rows):
    summary = []
    for D in sorted({int(r["D"]) for r in rows}):
        for H in sorted({int(r["H"]) for r in rows if int(r["D"]) == D}):
            group = [r for r in rows if int(r["H"]) == H and int(r["D"]) == D]
            row = {
                "H": H,
                "D": D,
                "horizon_steps": int(group[0]["horizon_steps"]),
                "lag_stride_frames": int(group[0]["lag_stride_frames"]),
                "readout_time_delays": str(list(range(H + 1))),
                "target_time_delays": str(
                    target_time_delays(
                        H,
                        int(group[0]["horizon_steps"]),
                        int(group[0]["lag_stride_frames"]),
                    )
                ),
                "number_of_terms": int(group[0]["number_of_terms"]),
                "num_topology_sample_pairs": len(group),
                "overparameterized_fraction": float(
                    np.mean([bool(r["overparameterized"]) for r in group])
                ),
                "ill_conditioned_fraction": float(
                    np.mean([bool(r["ill_conditioned"]) for r in group])
                ),
            }
            for key in (
                "num_train_rows",
                "num_validation_rows",
                "terms_per_train_row",
                "condition_number",
                "train_nmse_x",
                "train_nmse_y",
                "train_nmse_2d",
                "validation_nmse_x",
                "validation_nmse_y",
                "validation_nmse_2d",
                "train_validation_gap_2d",
            ):
                values = np.asarray([float(r[key]) for r in group], dtype=float)
                row[f"{key}_mean"] = float(np.nanmean(values))
                row[f"{key}_std"] = float(np.nanstd(values, ddof=0))
            summary.append(row)
    return summary


def summarize_by_topology(rows):
    summary = []
    keys = sorted({(r["topology"], int(r["H"]), int(r["D"])) for r in rows})
    for topology, H, D in keys:
        group = [r for r in rows if r["topology"] == topology and int(r["H"]) == H and int(r["D"]) == D]
        row = {
            "topology": topology,
            "amplitude": group[0]["amplitude"],
            "H": H,
            "D": D,
            "num_samples": len(group),
            "number_of_terms": int(group[0]["number_of_terms"]),
            "target_time_delays": group[0]["target_time_delays"],
        }
        for key in ("train_nmse_2d", "validation_nmse_2d", "condition_number"):
            values = np.asarray([float(r[key]) for r in group], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(values))
            row[f"{key}_std"] = float(np.nanstd(values, ddof=0))
        summary.append(row)
    return summary


def choose_common_cutoff(summary, tolerance_frac):
    valid = [r for r in summary if np.isfinite(r["validation_nmse_2d_mean"])]
    if not valid:
        raise ValueError("No valid H,D candidates.")
    best = min(valid, key=lambda r: r["validation_nmse_2d_mean"])
    tolerance = (1.0 + tolerance_frac) * best["validation_nmse_2d_mean"]
    eligible = [r for r in valid if r["validation_nmse_2d_mean"] <= tolerance]
    non_ill = [r for r in eligible if r["ill_conditioned_fraction"] == 0.0]
    if non_ill:
        eligible = non_ill
    non_over = [r for r in eligible if r["overparameterized_fraction"] == 0.0]
    if non_over:
        eligible = non_over
    selected = min(eligible, key=lambda r: (int(r["D"]), int(r["H"]), int(r["number_of_terms"])))
    return selected, best, tolerance


def basis_label(validation_nmse_2d):
    if validation_nmse_2d < 0.10:
        return "strong"
    if validation_nmse_2d < 0.20:
        return "acceptable"
    if validation_nmse_2d < 0.35:
        return "partial"
    return "weak"


def selected_design(record, args, H, D):
    sample_dir = record["sample_dir"]
    loader = StateLoader(sample_dir / "experiment.h5")
    _, positions = load_positions(sample_dir / "experiment.h5")
    u_norm = normalize_to_unit_interval(loader.get_actuation_signal(actuator_idx=0, dof=0))
    hidden_relative = positions[:, args.hidden_node, :] - positions[:, args.reference_node, :]
    train_start, train_stop, validation_stop = frame_windows(loader, args)
    rows = valid_rows(train_start, validation_stop, H, args.lag_stride_frames)
    design, exps = build_legendre_design(u_norm, H, D, args.lag_stride_frames)
    P = design[rows]
    Y = hidden_relative[rows + args.horizon_steps]
    return P, Y, exps


def selected_validation_trace(record, args, selected):
    H = int(selected["H"])
    D = int(selected["D"])
    sample_dir = record["sample_dir"]
    loader = StateLoader(sample_dir / "experiment.h5")
    time, positions = load_positions(sample_dir / "experiment.h5")
    u_norm = normalize_to_unit_interval(loader.get_actuation_signal(actuator_idx=0, dof=0))
    hidden_relative = positions[:, args.hidden_node, :] - positions[:, args.reference_node, :]
    train_start, train_stop, validation_stop = frame_windows(loader, args)
    design, _ = build_legendre_design(u_norm, H, D, args.lag_stride_frames)

    train_rows = valid_rows(train_start, train_stop, H, args.lag_stride_frames)
    val_rows = valid_rows(train_stop, validation_stop, H, args.lag_stride_frames)
    P_train = design[train_rows]
    P_val = design[val_rows]
    Y_train = hidden_relative[train_rows + args.horizon_steps]
    Y_val = hidden_relative[val_rows + args.horizon_steps]
    model, p_mean, y_mean = fit_centered_ridge(P_train, Y_train, args.ridge_alpha)
    Y_pred = predict_centered(model, p_mean, y_mean, P_val)
    nmse_x, nmse_y, nmse_2d = nmse_components(Y_val, Y_pred)
    target_time = time[val_rows + args.horizon_steps]
    target_time = target_time - target_time[0]
    return {
        "topology": record["topology"],
        "amplitude": record["amplitude"],
        "sample": record["sample"],
        "time": target_time,
        "target": Y_val,
        "predicted": Y_pred,
        "validation_nmse_x": nmse_x,
        "validation_nmse_y": nmse_y,
        "validation_nmse_2d": nmse_2d,
    }


def final_task_weight_rows(records, args, selected):
    H = int(selected["H"])
    D = int(selected["D"])
    rows = []
    reliability = []
    for record in records:
        P, Y, exps = selected_design(record, args, H, D)
        model, p_mean, y_mean = fit_centered_ridge(P, Y, args.ridge_alpha)
        pred = predict_centered(model, p_mean, y_mean, P)
        nmse_x, nmse_y, nmse_2d = nmse_components(Y, pred)
        p_centered = P - p_mean
        y_centered = Y - np.mean(Y, axis=0, keepdims=True)
        denom_x = float(np.sum(y_centered[:, 0] ** 2))
        denom_y = float(np.sum(y_centered[:, 1] ** 2))
        denom_2d = float(np.sum(y_centered**2))
        reliability.append(
            {
                "topology": record["topology"],
                "amplitude": record["amplitude"],
                "sample": record["sample"],
                "H": H,
                "D": D,
                "full_window_nmse_x": nmse_x,
                "full_window_nmse_y": nmse_y,
                "full_window_nmse_2d": nmse_2d,
                "basis_R2_2d": 1.0 - nmse_2d,
            }
        )
        for basis_index, exp_vec in enumerate(exps):
            active = np.where(exp_vec > 0)[0]
            max_delay = int(np.max(active)) if len(active) else 0
            target_delay = args.horizon_steps + max_delay * args.lag_stride_frames
            p_alpha = p_centered[:, basis_index]
            p_norm_sq = float(np.sum(p_alpha**2))
            coef_x = float(model.coef_[0, basis_index])
            coef_y = float(model.coef_[1, basis_index])
            rows.append(
                {
                    "topology": record["topology"],
                    "amplitude": record["amplitude"],
                    "sample": record["sample"],
                    "basis_index": basis_index,
                    "basis_term": term_name(exp_vec, args.horizon_steps, args.lag_stride_frames),
                    "total_degree": int(np.sum(exp_vec)),
                    "max_readout_time_delay": max_delay,
                    "max_target_time_delay": target_delay,
                    "exponents": " ".join(str(int(v)) for v in exp_vec),
                    "coefficient_x": coef_x,
                    "coefficient_y": coef_y,
                    "p_alpha_norm_sq": p_norm_sq,
                    "c_alpha_x": coef_x**2 * p_norm_sq / max(denom_x, np.finfo(float).eps),
                    "c_alpha_y": coef_y**2 * p_norm_sq / max(denom_y, np.finfo(float).eps),
                    "c_alpha": (coef_x**2 + coef_y**2)
                    * p_norm_sq
                    / max(denom_2d, np.finfo(float).eps),
                }
            )
    return rows, reliability


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def safe_path_part(text):
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(text))


def exclusion_path_part(excluded_samples):
    if not excluded_samples:
        return None
    parts = []
    for topology in sorted(excluded_samples):
        for sample in sorted(excluded_samples[topology]):
            parts.append(f"{safe_path_part(topology)}_{safe_path_part(sample)}")
    return "exclude_" + "__".join(parts)


def output_dir(args, records, excluded_samples):
    if args.output_dir:
        return Path(args.output_dir)
    topologies = sorted({r["topology"] for r in records})
    if len(topologies) == 1:
        label = topologies[0]
    else:
        label = f"{topologies[0]}_to_{topologies[-1]}"
    base = (
        data_root()
        / "task_specific_ipc"
        / "pdf_style_basis_calibration"
        / label
        / f"hidden_node_{args.hidden_node}"
        / f"horizon_{args.horizon_steps}_steps"
    )
    exclusion_label = exclusion_path_part(excluded_samples)
    if exclusion_label:
        base = base / exclusion_label
    return base


def save_plot(summary, selected, trace, out_path):
    rows = sorted(summary, key=lambda r: (int(r["D"]), int(r["H"])))
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.8, 7.4),
        sharex=False,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.25]},
    )
    selected_H = int(selected["H"])
    selected_D = int(selected["D"])
    selected_nmse = float(selected["validation_nmse_2d_mean"])
    selected_terms = int(selected["number_of_terms"])
    selected_condition = float(selected["condition_number_mean"])
    selected_text = (
        "Selected cutoff\n"
        f"H_cut = {selected_H}\n"
        f"D_cut = {selected_D}\n"
        f"val NMSE_2D = {selected_nmse:.4f}\n"
        f"terms = {selected_terms}"
    )
    for i, D in enumerate(sorted({int(r["D"]) for r in rows})):
        group = [r for r in rows if int(r["D"]) == D]
        axes[0].plot(
            [int(r["H"]) for r in group],
            [float(r["validation_nmse_2d_mean"]) for r in group],
            marker="o",
            lw=1.2,
            color=PALETTE[i % len(PALETTE)],
            label=f"D={D}",
        )
    axes[0].scatter(
        selected_H,
        selected_nmse,
        marker="*",
        s=140,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    axes[0].annotate(
        f"H={selected_H}, D={selected_D}\nNMSE={selected_nmse:.4f}",
        xy=(selected_H, selected_nmse),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=7,
        color="#111827",
        arrowprops={"arrowstyle": "->", "color": "#6B7280", "lw": 0.7},
    )
    axes[0].text(
        0.98,
        0.96,
        selected_text,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )
    axes[0].set_xlabel("input-history cutoff H")
    axes[0].set_ylabel("validation NMSE, 2D")
    axes[0].set_title("PDF-style holdout calibration")
    axes[0].grid(axis="y", color="#E5E7EB", lw=0.6)
    axes[0].legend(fontsize=7)

    colors = [PALETTE[(int(r["D"]) - 1) % len(PALETTE)] for r in rows]
    axes[1].scatter(
        [float(r["condition_number_mean"]) for r in rows],
        [float(r["validation_nmse_2d_mean"]) for r in rows],
        c=colors,
        s=32,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].scatter(
        selected_condition,
        selected_nmse,
        marker="*",
        s=140,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    axes[1].annotate(
        f"H={selected_H}, D={selected_D}\nNMSE={selected_nmse:.4f}",
        xy=(selected_condition, selected_nmse),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=7,
        color="#111827",
        arrowprops={"arrowstyle": "->", "color": "#6B7280", "lw": 0.7},
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("mean condition number")
    axes[1].set_ylabel("validation NMSE, 2D")
    axes[1].set_title("Conditioning check")
    axes[1].grid(axis="y", color="#E5E7EB", lw=0.6)

    t = trace["time"]
    target = trace["target"]
    predicted = trace["predicted"]
    axes[2].plot(
        t,
        target[:, 0],
        color="#0072B2",
        lw=1.15,
        label="x measured",
    )
    axes[2].plot(
        t,
        predicted[:, 0],
        color="#0072B2",
        lw=1.15,
        ls="--",
        label="x basis fit",
    )
    axes[2].plot(
        t,
        target[:, 1],
        color="#009E73",
        lw=1.15,
        label="y measured",
    )
    axes[2].plot(
        t,
        predicted[:, 1],
        color="#009E73",
        lw=1.15,
        ls="--",
        label="y basis fit",
    )
    trace_label = f"{trace['topology']}/{trace['amplitude']}/{trace['sample']}"
    axes[2].set_title("Selected Legendre basis reconstruction on validation window")
    axes[2].set_xlabel("validation time (s)")
    axes[2].set_ylabel("relative hidden-node position")
    axes[2].grid(axis="y", color="#E5E7EB", lw=0.6)
    axes[2].legend(ncol=4, fontsize=7, loc="upper left")
    axes[2].text(
        0.98,
        0.96,
        (
            f"{trace_label}\n"
            f"NMSE_x = {trace['validation_nmse_x']:.3f}\n"
            f"NMSE_y = {trace['validation_nmse_y']:.3f}\n"
            f"NMSE_2D = {trace['validation_nmse_2d']:.3f}"
        ),
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )

    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(
            -0.08,
            1.06,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
            va="bottom",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if out_path.suffix.lower() == ".pdf":
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    configure_matplotlib()
    args = parse_args()
    records, skipped, excluded_samples = discover_records(args)
    pairs = candidate_pairs(args)
    out_dir = output_dir(args, records, excluded_samples)

    print("PDF-style basis calibration:")
    print("  fit basis model on train window")
    print("  evaluate H,D on validation window")
    print("  no blocked CV, no optional degree trigger")
    print("  selected topology/sample records:")
    for record in records:
        print(f"    {record['topology']}/{record['amplitude']}/{record['sample']}")

    per_sample_rows = []
    for record in records:
        print(f"-> Evaluating {record['topology']}/{record['amplitude']}/{record['sample']}")
        for H, D in pairs:
            per_sample_rows.append(evaluate_candidate(record, args, H, D))

    summary = summarize_candidates(per_sample_rows)
    topology_summary = summarize_by_topology(per_sample_rows)
    selected, best, tolerance = choose_common_cutoff(summary, args.tolerance_frac)
    task_weights, reliability = final_task_weight_rows(records, args, selected)
    trace = selected_validation_trace(records[0], args, selected)
    label = basis_label(float(selected["validation_nmse_2d_mean"]))

    payload = {
        "script": "run_hidden_node_basis_selection.py",
        "selection_protocol": "PDF-style holdout: fit train window, validate validation window",
        "topologies": sorted({r["topology"] for r in records}),
        "amplitudes_by_topology": {
            topology: sorted({r["amplitude"] for r in records if r["topology"] == topology})
            for topology in sorted({r["topology"] for r in records})
        },
        "hidden_node": int(args.hidden_node),
        "reference_node": int(args.reference_node),
        "horizon_steps": int(args.horizon_steps),
        "washout_s": float(args.washout),
        "train_s": float(args.train),
        "validation_s": float(args.validation),
        "ridge_alpha": float(args.ridge_alpha),
        "H_cut": int(selected["H"]),
        "D_cut": int(selected["D"]),
        "readout_time_delays": list(range(int(selected["H"]) + 1)),
        "target_time_delays": target_time_delays(
            int(selected["H"]), args.horizon_steps, args.lag_stride_frames
        ),
        "selected_validation_nmse_2d": float(selected["validation_nmse_2d_mean"]),
        "global_best_H": int(best["H"]),
        "global_best_D": int(best["D"]),
        "global_best_validation_nmse_2d": float(best["validation_nmse_2d_mean"]),
        "selection_tolerance_nmse_2d": float(tolerance),
        "basis_adequacy_label": label,
        "num_terms_selected": int(selected["number_of_terms"]),
        "representative_trace": {
            "topology": trace["topology"],
            "amplitude": trace["amplitude"],
            "sample": trace["sample"],
            "validation_nmse_x": float(trace["validation_nmse_x"]),
            "validation_nmse_y": float(trace["validation_nmse_y"]),
            "validation_nmse_2d": float(trace["validation_nmse_2d"]),
        },
        "skipped": skipped,
        "excluded_samples": {
            topology: sorted(samples) for topology, samples in excluded_samples.items()
        },
    }
    trace_rows = [
        {
            "topology": trace["topology"],
            "amplitude": trace["amplitude"],
            "sample": trace["sample"],
            "time_s": float(time_value),
            "target_x": float(target_xy[0]),
            "target_y": float(target_xy[1]),
            "basis_fit_x": float(pred_xy[0]),
            "basis_fit_y": float(pred_xy[1]),
        }
        for time_value, target_xy, pred_xy in zip(
            trace["time"], trace["target"], trace["predicted"]
        )
    ]

    write_csv(per_sample_rows, out_dir / "basis_calibration_per_sample.csv")
    write_csv(summary, out_dir / "basis_calibration_summary.csv")
    write_csv(topology_summary, out_dir / "basis_calibration_by_topology.csv")
    write_csv(task_weights, out_dir / "task_weights_selected_dictionary.csv")
    write_csv(reliability, out_dir / "selected_dictionary_reliability.csv")
    write_csv(trace_rows, out_dir / "selected_basis_validation_trace.csv")
    write_csv(skipped, out_dir / "skipped_records.csv")
    write_json(payload, out_dir / "selected_dictionary.json")
    save_plot(summary, selected, trace, out_dir / "basis_calibration_summary.pdf")

    print(f"Saved PDF-style basis calibration outputs to: {out_dir}")
    print(
        f"Selected common cutoff: H={payload['H_cut']}, D={payload['D_cut']}, "
        f"validation NMSE_2D={payload['selected_validation_nmse_2d']:.4f}, "
        f"adequacy={label}."
    )
    print(
        f"Global best by validation NMSE: H={payload['global_best_H']}, "
        f"D={payload['global_best_D']}, "
        f"NMSE_2D={payload['global_best_validation_nmse_2d']:.4f}."
    )


if __name__ == "__main__":
    main()
