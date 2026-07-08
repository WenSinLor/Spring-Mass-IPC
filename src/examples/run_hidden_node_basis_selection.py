import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import LinAlgWarning

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
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
DEFAULT_Q_MIN = 0
DEFAULT_Q_MAX = 10
DEFAULT_D_VALUES = [1, 2]
DEFAULT_D3_TRIGGER_NMSE = 0.25
DEFAULT_LAG_STRIDE_FRAMES = 1
DEFAULT_CV_BLOCKS = 4
ALPHA_GRID = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0]

PALETTE = {
    "degree1": "#88CCEE",
    "degree2": "#44AA99",
    "degree3": "#CC6677",
    "train": "#6B7280",
    "validation": "#D55E00",
    "grid": "#E5E7EB",
    "text": "#1F2937",
}


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
        description=(
            "Select extra input-history depth Q and polynomial degree D for a hidden-node "
            "target using blocked CV on Legendre bases of the measured actuator input."
        )
    )
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    parser.add_argument("--amplitude", default=DEFAULT_AMPLITUDE)
    parser.add_argument(
        "--sample",
        default="all",
        help="Sample name such as sample_0, or 'all' for every sample_* directory.",
    )
    parser.add_argument("--hidden-node", type=int, default=DEFAULT_HIDDEN_NODE)
    parser.add_argument("--reference-node", type=int, default=DEFAULT_REFERENCE_NODE)
    parser.add_argument("--horizon-steps", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument("--washout", type=float, default=DEFAULT_WASHOUT_S)
    parser.add_argument(
        "--train",
        type=float,
        default=DEFAULT_TRAIN_S,
        help="Original train duration; combined with --test to define the basis-selection window.",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=DEFAULT_TEST_S,
        help="Original test duration; combined with --train to define the basis-selection window.",
    )
    parser.add_argument("--Q-min", type=int, default=DEFAULT_Q_MIN)
    parser.add_argument("--Q-max", type=int, default=DEFAULT_Q_MAX)
    parser.add_argument("--H-min", type=int, dest="Q_min", help=argparse.SUPPRESS)
    parser.add_argument("--H-max", type=int, dest="Q_max", help=argparse.SUPPRESS)
    parser.add_argument(
        "--D-values",
        type=int,
        nargs="+",
        default=DEFAULT_D_VALUES,
        help="Main polynomial degrees to scan over the full Q range.",
    )
    parser.add_argument(
        "--D3-trigger-nmse",
        type=float,
        default=DEFAULT_D3_TRIGGER_NMSE,
        help="Run optional D=3 search if the best D=2 CV NMSE_2D is above this value.",
    )
    parser.add_argument(
        "--lag-stride-frames",
        type=int,
        default=DEFAULT_LAG_STRIDE_FRAMES,
        help="Lag stride K: history is [u_m, u_{m-K}, u_{m-2K}, ...].",
    )
    parser.add_argument(
        "--cv-blocks",
        type=int,
        default=DEFAULT_CV_BLOCKS,
        help="Number of contiguous leave-one-block-out CV folds.",
    )
    return parser.parse_args()


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


def load_positions(h5_path: Path):
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
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return np.sqrt(3.0) * x
    p_nm2 = np.ones_like(x, dtype=float)
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


def target_delay_set_frames(extra_history_depth, horizon_steps, lag_stride_frames):
    return [horizon_steps + q * lag_stride_frames for q in range(extra_history_depth + 1)]


def basis_term_name(exp_vec, horizon_steps, lag_stride_frames):
    parts = []
    for q, degree in enumerate(exp_vec):
        if degree > 0:
            target_delay = horizon_steps + q * lag_stride_frames
            parts.append(f"P{int(degree)}(u[target-{target_delay}])")
    return " * ".join(parts)


def build_legendre_design(u_norm, Q_extra, D, lag_stride_frames):
    num_lags = Q_extra + 1
    exps = exponent_vectors(num_lags, D)
    n_rows = len(u_norm)
    design = np.empty((n_rows, len(exps)), dtype=np.float32)

    values_by_degree = {
        degree: legendre_normalized(degree, u_norm) for degree in range(D + 1)
    }

    lagged_values = {}
    for degree in range(D + 1):
        lagged = np.empty((n_rows, num_lags), dtype=np.float32)
        lagged[:, :] = np.nan
        source = values_by_degree[degree]
        for lag_index in range(num_lags):
            frame_lag = lag_index * lag_stride_frames
            if frame_lag == 0:
                lagged[:, lag_index] = source
            else:
                lagged[frame_lag:, lag_index] = source[:-frame_lag]
        lagged_values[degree] = lagged

    for col, exp_vec in enumerate(exps):
        term = np.ones(n_rows, dtype=np.float32)
        for lag_index, degree in enumerate(exp_vec):
            if degree > 0:
                term *= lagged_values[int(degree)][:, lag_index]
        design[:, col] = term

    return design, exps


def nmse_components(y_true, y_pred):
    err = y_true - y_pred
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    sse = np.sum(err**2, axis=0)
    sst = np.sum(centered**2, axis=0)
    nmse_xy = sse / np.maximum(sst, np.finfo(float).eps)
    nmse_2d = float(np.sum(err**2) / np.maximum(np.sum(centered**2), np.finfo(float).eps))
    return float(nmse_xy[0]), float(nmse_xy[1]), nmse_2d


def fit_ridge(P_train, y_train, alpha):
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(P_train, y_train)
    return model


def make_cv_folds(n_rows, num_blocks):
    indices = np.arange(n_rows)
    blocks = np.array_split(indices, num_blocks)
    folds = []
    for block in blocks:
        if len(block) == 0:
            continue
        train_idx = np.setdiff1d(indices, block, assume_unique=True)
        folds.append((train_idx, block))
    if len(folds) < 2:
        raise ValueError("Not enough rows for blocked cross-validation.")
    return folds


def cv_evaluate(P, y, number_of_terms, num_blocks):
    folds = make_cv_folds(len(P), num_blocks)
    min_train_rows = min(len(train_idx) for train_idx, _ in folds)
    if number_of_terms > 0.5 * min_train_rows:
        return {
            "status": "too_many_terms",
            "min_cv_train_rows": int(min_train_rows),
            "number_of_terms": int(number_of_terms),
        }

    alpha_scores = []
    for alpha in ALPHA_GRID:
        fold_scores = []
        for train_idx, val_idx in folds:
            model = fit_ridge(P[train_idx], y[train_idx], alpha)
            pred_val = model.predict(P[val_idx])
            _, _, val_nmse_2d = nmse_components(y[val_idx], pred_val)
            fold_scores.append(val_nmse_2d)
        alpha_scores.append((float(np.mean(fold_scores)), alpha))
    selected_alpha = min(alpha_scores, key=lambda item: item[0])[1]

    fold_records = []
    for fold_id, (train_idx, val_idx) in enumerate(folds):
        model = fit_ridge(P[train_idx], y[train_idx], selected_alpha)
        pred_train = model.predict(P[train_idx])
        pred_val = model.predict(P[val_idx])
        train_nmse_x, train_nmse_y, train_nmse_2d = nmse_components(y[train_idx], pred_train)
        val_nmse_x, val_nmse_y, val_nmse_2d = nmse_components(y[val_idx], pred_val)
        fold_records.append(
            {
                "fold": fold_id,
                "train_nmse_x": train_nmse_x,
                "val_nmse_x": val_nmse_x,
                "train_nmse_y": train_nmse_y,
                "val_nmse_y": val_nmse_y,
                "train_nmse_2d": train_nmse_2d,
                "val_nmse_2d": val_nmse_2d,
            }
        )

    out = {
        "status": "ok",
        "selected_alpha": selected_alpha,
        "min_cv_train_rows": int(min_train_rows),
    }
    for key in (
        "train_nmse_x",
        "val_nmse_x",
        "train_nmse_y",
        "val_nmse_y",
        "train_nmse_2d",
        "val_nmse_2d",
    ):
        values = np.asarray([r[key] for r in fold_records], dtype=float)
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values, ddof=0))
    out["train_val_gap_2d"] = out["val_nmse_2d_mean"] - out["train_nmse_2d_mean"]
    return out


def frame_windows(loader, washout_s, train_s, test_s):
    washout_frames = int(washout_s / loader.dt)
    selection_frames = int((train_s + test_s) / loader.dt)
    window_start = washout_frames
    window_stop = window_start + selection_frames
    if window_stop > loader.total_frames:
        raise ValueError(
            f"Need {window_stop} frames for washout + basis-selection window, "
            f"but only {loader.total_frames} frames are available."
        )
    return window_start, window_stop


def candidate_pairs(args, include_degree3=False):
    if args.Q_min < 0 or args.Q_max < args.Q_min:
        raise ValueError("Require 0 <= Q_min <= Q_max.")
    d_values = sorted(set(int(d) for d in args.D_values))
    if include_degree3 and 3 not in d_values:
        d_values.append(3)
    return [(Q_extra, D) for D in d_values for Q_extra in range(args.Q_min, args.Q_max + 1)]


def evaluate_sample(sample_dir, args, pairs):
    h5_path = sample_dir / "experiment.h5"
    loader = StateLoader(h5_path)
    time, positions = load_positions(h5_path)
    n_nodes = positions.shape[1]

    if args.hidden_node >= n_nodes:
        raise ValueError(f"Hidden node {args.hidden_node} is outside node range 0..{n_nodes - 1}.")
    if args.hidden_node == args.reference_node:
        raise ValueError("The hidden node cannot also be the reference node.")
    if args.lag_stride_frames < 1:
        raise ValueError("--lag-stride-frames must be at least 1.")

    u_norm = normalize_to_unit_interval(loader.get_actuation_signal(actuator_idx=0, dof=0))
    hidden_relative = positions[:, args.hidden_node, :] - positions[:, args.reference_node, :]

    window_start, window_stop = frame_windows(loader, args.washout, args.train, args.test)
    if window_stop + args.horizon_steps > loader.total_frames:
        raise ValueError(
            f"Horizon requires target frame {window_stop + args.horizon_steps}, "
            f"but sample has {loader.total_frames} frames."
        )

    records = []
    design_cache = {}
    for Q_extra, D in pairs:
        max_lag_frames = Q_extra * args.lag_stride_frames
        valid_start = window_start + max_lag_frames
        valid_stop = window_stop
        if valid_start >= valid_stop:
            raise ValueError(
                f"No valid rows for Q={Q_extra}, lag stride={args.lag_stride_frames}; "
                "reduce Q or lag stride."
            )

        design, exps = build_legendre_design(u_norm, Q_extra, D, args.lag_stride_frames)
        row_idx = np.arange(valid_start, valid_stop)
        P = design[row_idx]
        y = hidden_relative[row_idx + args.horizon_steps]
        if np.isnan(P).any():
            raise ValueError(f"NaNs remained in design matrix for Q={Q_extra}, D={D}.")

        cv = cv_evaluate(P, y, design.shape[1], args.cv_blocks)
        delay_set = target_delay_set_frames(Q_extra, args.horizon_steps, args.lag_stride_frames)
        record = {
            "sample": sample_dir.name,
            "Q_extra": Q_extra,
            "D": D,
            "horizon_steps": args.horizon_steps,
            "lag_stride_frames": args.lag_stride_frames,
            "target_delay_frames_min": min(delay_set),
            "target_delay_frames_max": max(delay_set),
            "target_delay_set_frames": str(delay_set),
            "max_lag_frames": max_lag_frames,
            "number_of_terms": int(design.shape[1]),
            "status": cv["status"],
            "selected_alpha": cv.get("selected_alpha", np.nan),
            "min_cv_train_rows": cv["min_cv_train_rows"],
            "window_rows": int(len(P)),
            "terms_per_min_cv_train_row": float(design.shape[1] / max(cv["min_cv_train_rows"], 1)),
            "train_nmse_x_mean": cv.get("train_nmse_x_mean", np.nan),
            "train_nmse_x_std": cv.get("train_nmse_x_std", np.nan),
            "val_nmse_x_mean": cv.get("val_nmse_x_mean", np.nan),
            "val_nmse_x_std": cv.get("val_nmse_x_std", np.nan),
            "train_nmse_y_mean": cv.get("train_nmse_y_mean", np.nan),
            "train_nmse_y_std": cv.get("train_nmse_y_std", np.nan),
            "val_nmse_y_mean": cv.get("val_nmse_y_mean", np.nan),
            "val_nmse_y_std": cv.get("val_nmse_y_std", np.nan),
            "train_nmse_2d_mean": cv.get("train_nmse_2d_mean", np.nan),
            "train_nmse_2d_std": cv.get("train_nmse_2d_std", np.nan),
            "val_nmse_2d_mean": cv.get("val_nmse_2d_mean", np.nan),
            "val_nmse_2d_std": cv.get("val_nmse_2d_std", np.nan),
            "train_val_gap_2d": cv.get("train_val_gap_2d", np.nan),
        }
        records.append(record)
        design_cache[(Q_extra, D)] = {
            "P": P,
            "y": y,
            "exps": exps,
            "time": time[row_idx],
            "row_idx": row_idx,
        }

    return records, design_cache


def aggregate_records(records, pairs):
    keys = [
        "number_of_terms",
        "selected_alpha",
        "min_cv_train_rows",
        "window_rows",
        "terms_per_min_cv_train_row",
        "train_nmse_x_mean",
        "val_nmse_x_mean",
        "train_nmse_y_mean",
        "val_nmse_y_mean",
        "train_nmse_2d_mean",
        "val_nmse_2d_mean",
        "train_val_gap_2d",
    ]
    summary = []
    for Q_extra, D in pairs:
        group = [r for r in records if r["Q_extra"] == Q_extra and r["D"] == D]
        if not group:
            continue
        ok_group = [r for r in group if r["status"] == "ok"]
        delay_set = target_delay_set_frames(
            Q_extra,
            int(group[0]["horizon_steps"]),
            int(group[0]["lag_stride_frames"]),
        )
        row = {
            "Q_extra": Q_extra,
            "D": D,
            "horizon_steps": int(group[0]["horizon_steps"]),
            "lag_stride_frames": int(group[0]["lag_stride_frames"]),
            "target_delay_frames_min": min(delay_set),
            "target_delay_frames_max": max(delay_set),
            "target_delay_set_frames": str(delay_set),
        }
        row["status"] = "ok" if len(ok_group) == len(group) else "too_many_terms"
        for key in keys:
            values = np.asarray([r[key] for r in ok_group], dtype=float)
            if len(values) == 0:
                row[key] = np.nan
                row[f"{key}_std_across_samples"] = np.nan
            else:
                row[key] = float(np.nanmean(values))
                row[f"{key}_std_across_samples"] = float(np.nanstd(values, ddof=0))
        val_std_values = np.asarray([r["val_nmse_2d_std"] for r in ok_group], dtype=float)
        row["val_nmse_2d_cv_std_mean"] = (
            float(np.nanmean(val_std_values)) if len(val_std_values) else np.nan
        )
        row["num_samples"] = len(group)
        row["num_ok_samples"] = len(ok_group)
        summary.append(row)
    return summary


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def basis_label(nmse):
    if nmse < 0.10:
        return "strong"
    if nmse < 0.20:
        return "acceptable"
    if nmse < 0.35:
        return "partial"
    return "weak"


def choose_recommendation(summary):
    valid = [r for r in summary if r["status"] == "ok" and np.isfinite(r["val_nmse_2d_mean"])]
    if not valid:
        raise ValueError("No valid candidate satisfies the term-count constraint.")
    global_best = min(valid, key=lambda r: r["val_nmse_2d_mean"])
    tolerance = 1.05 * global_best["val_nmse_2d_mean"]
    eligible = [
        r
        for r in valid
        if r["val_nmse_2d_mean"] <= tolerance
        and r["train_val_gap_2d"] < max(0.10, 0.5 * r["val_nmse_2d_mean"])
    ]
    if not eligible:
        eligible = [r for r in valid if r["val_nmse_2d_mean"] <= tolerance]
    recommended = min(eligible, key=lambda r: (r["D"], r["Q_extra"], r["number_of_terms"]))

    larger = [
        r
        for r in valid
        if (r["D"] > recommended["D"] or r["Q_extra"] > recommended["Q_extra"])
    ]
    if larger:
        best_larger = min(larger, key=lambda r: r["val_nmse_2d_mean"])
        relative_gain = (
            recommended["val_nmse_2d_mean"] - best_larger["val_nmse_2d_mean"]
        ) / max(recommended["val_nmse_2d_mean"], np.finfo(float).eps)
    else:
        best_larger = None
        relative_gain = 0.0
    return recommended, global_best, best_larger, float(relative_gain)


def selected_sample_designs(sample_dirs, args, selected_pair):
    designs = []
    for sample_dir in sample_dirs:
        records, cache = evaluate_sample(sample_dir, args, [selected_pair])
        record = records[0]
        if record["status"] != "ok":
            raise ValueError(f"Selected basis is invalid for {sample_dir.name}.")
        designs.append((sample_dir.name, record, cache[selected_pair]))
    return designs


def refit_selected_basis(sample_designs, selected, out_dir, hidden_node, horizon_steps, lag_stride_frames):
    coefficient_rows = []
    task_weight_rows = []
    reliability_rows = []
    Q_extra = int(selected["Q_extra"])
    D = int(selected["D"])
    delay_set = target_delay_set_frames(Q_extra, horizon_steps, lag_stride_frames)

    for sample_name, record, design_info in sample_designs:
        alpha = record["selected_alpha"]
        model = fit_ridge(design_info["P"], design_info["y"], alpha)
        pred = model.predict(design_info["P"])
        nmse_x, nmse_y, nmse_2d = nmse_components(design_info["y"], pred)
        y_centered = design_info["y"] - np.mean(design_info["y"], axis=0, keepdims=True)
        denom_x = float(np.sum(y_centered[:, 0] ** 2))
        denom_y = float(np.sum(y_centered[:, 1] ** 2))
        denom_2d = float(np.sum(y_centered**2))
        reliability_rows.append(
            {
                "sample": sample_name,
                "Q_extra": Q_extra,
                "D": D,
                "horizon_steps": horizon_steps,
                "lag_stride_frames": lag_stride_frames,
                "target_delay_frames_min": min(delay_set),
                "target_delay_frames_max": max(delay_set),
                "target_delay_set_frames": str(delay_set),
                "selected_alpha": alpha,
                "full_window_nmse_x": nmse_x,
                "full_window_nmse_y": nmse_y,
                "full_window_nmse_2d": nmse_2d,
                "cv_val_nmse_2d_mean": record["val_nmse_2d_mean"],
                "cv_val_nmse_2d_std": record["val_nmse_2d_std"],
            }
        )

        coefficient_rows.append(
            {
                "sample": sample_name,
                "basis_index": -1,
                "basis_term": "intercept",
                "total_degree": 0,
                "Q_extra": Q_extra,
                "max_effective_delay_Q": 0,
                "target_delay_frames": "",
                "max_lag_frames": 0,
                "c_alpha_x": float(model.intercept_[0]),
                "c_alpha_y": float(model.intercept_[1]),
            }
        )
        for basis_index, exp_vec in enumerate(design_info["exps"]):
            active = np.where(exp_vec > 0)[0]
            max_q = int(np.max(active)) if len(active) else 0
            target_delay = horizon_steps + max_q * lag_stride_frames
            p_alpha = design_info["P"][:, basis_index]
            p_norm_sq = float(np.sum(p_alpha**2))
            coef_x = float(model.coef_[0, basis_index])
            coef_y = float(model.coef_[1, basis_index])
            weight_x = (coef_x**2 * p_norm_sq) / max(denom_x, np.finfo(float).eps)
            weight_y = (coef_y**2 * p_norm_sq) / max(denom_y, np.finfo(float).eps)
            weight_2d = ((coef_x**2 + coef_y**2) * p_norm_sq) / max(
                denom_2d, np.finfo(float).eps
            )
            coefficient_rows.append(
                {
                    "sample": sample_name,
                    "basis_index": basis_index,
                    "basis_term": basis_term_name(exp_vec, horizon_steps, lag_stride_frames),
                    "total_degree": int(np.sum(exp_vec)),
                    "Q_extra": Q_extra,
                    "max_effective_delay_Q": max_q,
                    "target_delay_frames": target_delay,
                    "max_lag_frames": max_q * lag_stride_frames,
                    "c_alpha_x": coef_x,
                    "c_alpha_y": coef_y,
                }
            )
            task_weight_rows.append(
                {
                    "sample": sample_name,
                    "basis_index": basis_index,
                    "basis_term": basis_term_name(exp_vec, horizon_steps, lag_stride_frames),
                    "total_degree": int(np.sum(exp_vec)),
                    "Q_extra": Q_extra,
                    "max_effective_delay_Q": max_q,
                    "target_delay_frames": target_delay,
                    "p_alpha_norm_sq": p_norm_sq,
                    "basis_contribution_weight_x": weight_x,
                    "basis_contribution_weight_y": weight_y,
                    "basis_contribution_weight_2d": weight_2d,
                }
            )

    coeff_path = out_dir / f"hidden_node_{hidden_node}_final_coefficients.csv"
    task_weight_path = out_dir / f"hidden_node_{hidden_node}_final_task_weights.csv"
    reliability_path = out_dir / f"hidden_node_{hidden_node}_selected_basis_reliability.csv"
    write_csv(coefficient_rows, coeff_path)
    write_csv(task_weight_rows, task_weight_path)
    write_csv(reliability_rows, reliability_path)
    return coeff_path, task_weight_path, reliability_path


def coefficient_heatmap_from_rows(coefficient_rows):
    data_rows = [r for r in coefficient_rows if r["basis_index"] >= 0]
    if not data_rows:
        return np.zeros((1, 1))
    max_degree = max(int(r["total_degree"]) for r in data_rows)
    delays = sorted({int(r["target_delay_frames"]) for r in data_rows})
    delay_to_col = {delay: col for col, delay in enumerate(delays)}
    heatmap = np.zeros((max_degree, len(delays)), dtype=float)
    for r in data_rows:
        degree = int(r["total_degree"])
        h = delay_to_col[int(r["target_delay_frames"])]
        mag = np.sqrt(float(r["c_alpha_x"]) ** 2 + float(r["c_alpha_y"]) ** 2)
        heatmap[degree - 1, h] += mag
    return heatmap, delays


def save_metadata(out_dir, args, selected, global_best, adequacy_label):
    metadata = {
        "script": "run_hidden_node_basis_selection.py",
        "purpose": (
            "Scalar measured-input-history basis selection for hidden-node target. "
            "This is not the reservoir hidden-node readout task."
        ),
        "distinction_from_reservoir_readout": {
            "this_script_answers": (
                "How well can scalar measured input history explain the hidden-node target?"
            ),
            "hidden_node_prediction_script_answers": (
                "How well can visible reservoir states predict the hidden node?"
            ),
        },
        "terminology": {
            "Q_extra": "extra history depth before the prediction time",
            "horizon_steps": args.horizon_steps,
            "lag_stride_frames": args.lag_stride_frames,
            "target_time_delay_formula": (
                "target_delay_frames(q) = horizon_steps + q * lag_stride_frames"
            ),
            "important_note": (
                "Q=0 with horizon_steps>0 corresponds to a nonzero input-to-target "
                "delay equal to horizon_steps, not zero memory."
            ),
        },
        "selection": {
            "Q_extra": int(selected["Q_extra"]),
            "D": int(selected["D"]),
            "target_delay_set_frames": selected["target_delay_set_frames"],
            "cv_nmse_2d_mean": float(selected["val_nmse_2d_mean"]),
            "cv_nmse_2d_std_mean": float(selected["val_nmse_2d_cv_std_mean"]),
            "adequacy": adequacy_label,
        },
        "global_best": {
            "Q_extra": int(global_best["Q_extra"]),
            "D": int(global_best["D"]),
            "target_delay_set_frames": global_best["target_delay_set_frames"],
            "cv_nmse_2d_mean": float(global_best["val_nmse_2d_mean"]),
        },
        "scientific_wording": (
            "When the best CV basis NMSE is weak, the selected compact basis should "
            "be interpreted only as the dominant input-driven component. It should "
            "not be interpreted as the full delay or degree required by the physical "
            "hidden-node task."
        ),
    }
    metadata_path = out_dir / "basis_selection_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_summary(summary, selected, coefficient_rows, out_path):
    rows = [r for r in summary if r["status"] == "ok"]
    fig = plt.figure(figsize=(7.4, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_terms = fig.add_subplot(gs[0, 0])
    ax_delay = fig.add_subplot(gs[0, 1])
    ax_gap = fig.add_subplot(gs[1, 0])
    ax_coef = fig.add_subplot(gs[1, 1])

    colors = [PALETTE[f"degree{int(r['D'])}"] for r in rows]
    ax_terms.scatter(
        [r["number_of_terms"] for r in rows],
        [r["val_nmse_2d_mean"] for r in rows],
        s=34,
        c=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_terms.scatter(
        selected["number_of_terms"],
        selected["val_nmse_2d_mean"],
        s=110,
        marker="*",
        color=PALETTE["validation"],
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    ax_terms.annotate(
        f"selected\nQ={int(selected['Q_extra'])}, D={int(selected['D'])}",
        xy=(selected["number_of_terms"], selected["val_nmse_2d_mean"]),
        xytext=(10, 14),
        textcoords="offset points",
        fontsize=6,
        color=PALETTE["validation"],
        arrowprops={"arrowstyle": "-", "color": PALETTE["validation"], "lw": 0.7},
    )
    ax_terms.set_xscale("log")
    ax_terms.set_xlabel("number of Legendre terms")
    ax_terms.set_ylabel("CV NMSE, 2D")
    ax_terms.set_title("Basis size versus CV error")
    ax_terms.grid(axis="y", color=PALETTE["grid"], lw=0.6)
    panel_label(ax_terms, "a")

    for D in sorted({int(r["D"]) for r in rows}):
        group = sorted(
            [r for r in rows if int(r["D"]) == D],
            key=lambda r: r["target_delay_frames_max"],
        )
        ax_delay.plot(
            [r["target_delay_frames_max"] for r in group],
            [r["val_nmse_2d_mean"] for r in group],
            marker="o",
            lw=1.2,
            color=PALETTE[f"degree{D}"],
            label=f"D={D}",
        )
    ax_delay.scatter(
        selected["target_delay_frames_max"],
        selected["val_nmse_2d_mean"],
        s=110,
        marker="*",
        color=PALETTE["validation"],
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    max_delay = int(max(r["target_delay_frames_max"] for r in rows))
    min_delay = int(min(r["target_delay_frames_max"] for r in rows))
    ax_delay.set_xlim(min_delay - 0.5, max_delay + 0.5)
    ax_delay.set_xticks(
        np.arange(min_delay, max_delay + 1, max(1, (max_delay - min_delay) // 5))
    )
    ax_delay.set_xlabel("maximum target-time input delay (frames)")
    ax_delay.set_ylabel("CV NMSE, 2D")
    ax_delay.set_title("Delay sweep grouped by degree")
    ax_delay.grid(axis="y", color=PALETTE["grid"], lw=0.6)
    ax_delay.legend(fontsize=6)
    panel_label(ax_delay, "b")

    ax_gap.scatter(
        [r["train_nmse_2d_mean"] for r in rows],
        [r["val_nmse_2d_mean"] for r in rows],
        s=34,
        c=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_gap.scatter(
        selected["train_nmse_2d_mean"],
        selected["val_nmse_2d_mean"],
        s=110,
        marker="*",
        color=PALETTE["validation"],
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    lim_max = max(
        max(r["train_nmse_2d_mean"] for r in rows),
        max(r["val_nmse_2d_mean"] for r in rows),
    ) * 1.08
    ax_gap.plot([0, lim_max], [0, lim_max], color="#4B5563", lw=0.8, ls=":")
    ax_gap.set_xlim(left=0)
    ax_gap.set_ylim(bottom=0)
    ax_gap.set_xlabel("train NMSE, 2D")
    ax_gap.set_ylabel("validation NMSE, 2D")
    ax_gap.set_title("Blocked CV train-validation gap")
    ax_gap.grid(axis="y", color=PALETTE["grid"], lw=0.6)
    panel_label(ax_gap, "c")

    heatmap, heatmap_delays = coefficient_heatmap_from_rows(coefficient_rows)
    im = ax_coef.imshow(heatmap, aspect="auto", origin="lower", cmap="magma")
    cbar = fig.colorbar(im, ax=ax_coef, fraction=0.046, pad=0.02)
    cbar.set_label("|c_alpha| sum")
    ax_coef.set_yticks(np.arange(heatmap.shape[0]))
    ax_coef.set_yticklabels([str(i + 1) for i in range(heatmap.shape[0])])
    ax_coef.set_xticks(np.arange(len(heatmap_delays)))
    ax_coef.set_xticklabels([str(delay) for delay in heatmap_delays])
    ax_coef.set_xlabel("target-time delay (frames)")
    ax_coef.set_ylabel("total degree")
    ax_coef.set_title(
        f"Full-window coefficients: Q={int(selected['Q_extra'])}, D={int(selected['D'])}"
    )
    panel_label(ax_coef, "d")

    fig.suptitle(
        f"Selected compact basis: Q={int(selected['Q_extra'])}, D={int(selected['D'])}; "
        f"target-time delays = {selected['target_delay_set_frames']} frames; "
        f"explanation = {basis_label(selected['val_nmse_2d_mean'])}",
        fontsize=8,
        y=1.02,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", message="Singular matrix in solving dual problem.*")
    configure_matplotlib()
    args = parse_args()

    data_root = src_dir.parent / "data" / "experiment_data"
    base_dir = data_root / args.topology / args.amplitude
    sample_dirs = find_sample_dirs(base_dir, args.sample)
    out_dir = (
        base_dir
        / "plots"
        / f"hidden_node_{args.hidden_node}_basis_selection"
        / f"horizon_{args.horizon_steps}_steps"
        / f"lag_stride_{args.lag_stride_frames}_frames"
    )

    print(
        "Basis-selection setup: "
        f"target=node {args.hidden_node} relative to node {args.reference_node}, "
        f"input=measured actuator node, horizon h={args.horizon_steps} frame(s), "
        f"lag stride K={args.lag_stride_frames} frame(s)."
    )
    print(
        "Using one 20 s post-washout basis-selection window with "
        "4 contiguous leave-one-block-out folds."
    )
    print("This does not use reservoir states or reservoir readout NMSE.")
    print(
        "Q is extra history depth before the prediction time. The actual "
        "input-to-target delay is horizon_steps + Q * lag_stride_frames."
    )

    main_pairs = candidate_pairs(args, include_degree3=False)
    all_records = []
    per_sample_cache = {}
    for sample_dir in sample_dirs:
        print(f"-> Main search for {sample_dir.name}")
        records, cache = evaluate_sample(sample_dir, args, main_pairs)
        all_records.extend(records)
        per_sample_cache[sample_dir.name] = cache

    all_pairs = list(main_pairs)
    summary = aggregate_records(all_records, all_pairs)
    d2_rows = [r for r in summary if int(r["D"]) == 2 and r["status"] == "ok"]
    best_d2 = min(d2_rows, key=lambda r: r["val_nmse_2d_mean"]) if d2_rows else None
    if best_d2 is not None and best_d2["val_nmse_2d_mean"] > args.D3_trigger_nmse:
        d3_pairs = [(Q_extra, 3) for Q_extra in range(args.Q_min, args.Q_max + 1)]
        print(
            f"Best D=2 CV NMSE_2D={best_d2['val_nmse_2d_mean']:.4f} "
            f"> {args.D3_trigger_nmse:.4f}; running optional D=3 search."
        )
        for sample_dir in sample_dirs:
            print(f"-> Optional D=3 search for {sample_dir.name}")
            records, cache = evaluate_sample(sample_dir, args, d3_pairs)
            all_records.extend(records)
            per_sample_cache[sample_dir.name].update(cache)
        all_pairs.extend(d3_pairs)
        summary = aggregate_records(all_records, all_pairs)
    elif best_d2 is not None:
        print(
            f"Skipping D=3 search: best D=2 CV NMSE_2D={best_d2['val_nmse_2d_mean']:.4f} "
            f"<= {args.D3_trigger_nmse:.4f}."
        )

    selected, global_best, best_larger, relative_gain = choose_recommendation(summary)
    selected_pair = (int(selected["Q_extra"]), int(selected["D"]))
    sample_designs = selected_sample_designs(sample_dirs, args, selected_pair)
    coeff_path, task_weight_path, reliability_path = refit_selected_basis(
        sample_designs,
        selected,
        out_dir,
        args.hidden_node,
        args.horizon_steps,
        args.lag_stride_frames,
    )

    with coeff_path.open("r", newline="") as f:
        coefficient_rows = [
            {
                **row,
                "basis_index": int(row["basis_index"]),
                "total_degree": int(row["total_degree"]),
                "max_effective_delay_Q": int(row["max_effective_delay_Q"]),
                "target_delay_frames": int(row["target_delay_frames"])
                if row["target_delay_frames"]
                else "",
                "c_alpha_x": float(row["c_alpha_x"]),
                "c_alpha_y": float(row["c_alpha_y"]),
            }
            for row in csv.DictReader(f)
        ]

    per_sample_csv = out_dir / f"hidden_node_{args.hidden_node}_basis_selection_cv_per_sample.csv"
    summary_csv = out_dir / f"hidden_node_{args.hidden_node}_basis_selection_cv_summary.csv"
    figure_pdf = out_dir / f"hidden_node_{args.hidden_node}_basis_selection_cv_summary.pdf"
    write_csv(all_records, per_sample_csv)
    write_csv(summary, summary_csv)
    plot_summary(summary, selected, coefficient_rows, figure_pdf)

    label = basis_label(selected["val_nmse_2d_mean"])
    metadata_path = save_metadata(out_dir, args, selected, global_best, label)
    print(f"Saved per-sample CV CSV: {per_sample_csv}")
    print(f"Saved summary CV CSV: {summary_csv}")
    print(f"Saved final coefficients: {coeff_path}")
    print(f"Saved final task contribution weights: {task_weight_path}")
    print(f"Saved selected-basis reliability: {reliability_path}")
    print(f"Saved basis-selection metadata: {metadata_path}")
    print(f"Saved PDF figure: {figure_pdf}")
    print(
        "Selected compact basis:\n"
        f"    Q = {int(selected['Q_extra'])}\n"
        f"    D = {int(selected['D'])}\n"
        f"    horizon h = {args.horizon_steps} frames\n"
        f"    lag stride K = {args.lag_stride_frames} frame(s)\n"
        f"    target-time delay set = {selected['target_delay_set_frames']} frames\n"
        f"Mean CV NMSE_2D={selected['val_nmse_2d_mean']:.4f} "
        f"+/- {selected['val_nmse_2d_cv_std_mean']:.4f}; adequacy={label}."
    )
    print(
        "Interpretation: Q=0 does not mean zero memory. It means the dominant "
        f"input-driven component is associated with input {args.horizon_steps} "
        "frames before the target."
    )
    if label == "weak":
        print(
            "No adequate scalar input-history basis was found. The selected compact "
            "basis only captures the dominant input-driven component."
        )
    elif label == "partial":
        print(
            "Interpretation: the tested input-history polynomial basis captures only "
            "the dominant input-driven component; do not claim the full task only "
            "requires this Q,D."
        )
    if best_larger is not None:
        print(
            f"Best larger candidate: Q={int(best_larger['Q_extra'])}, D={int(best_larger['D'])}, "
            f"target delay max={int(best_larger['target_delay_frames_max'])} frames, "
            f"CV NMSE_2D={best_larger['val_nmse_2d_mean']:.4f}; "
            f"relative gain={100.0 * relative_gain:.1f}%."
        )
    print(
        f"Global best by CV NMSE alone: Q={int(global_best['Q_extra'])}, "
        f"D={int(global_best['D'])}, "
        f"target delay max={int(global_best['target_delay_frames_max'])} frames, "
        f"CV NMSE_2D={global_best['val_nmse_2d_mean']:.4f}."
    )


if __name__ == "__main__":
    main()
