import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from openprc.reservoir.features.node_features import NodeDisplacements
from openprc.reservoir.io.state_loader import StateLoader


ALPHA_GRID = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0]
PALETTE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]


def data_root():
    return src_dir.parent / "data" / "experiment_data"


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


def topology_names(start, stop, name_format):
    return [name_format.format(i=i) for i in range(start, stop + 1)]


def find_sample_dirs(base_dir, sample_arg="all"):
    base_dir = Path(base_dir)
    if sample_arg != "all":
        sample_dir = base_dir / sample_arg
        if not (sample_dir / "experiment.h5").exists():
            raise FileNotFoundError(f"Experiment file not found: {sample_dir / 'experiment.h5'}")
        return [sample_dir]
    return sorted(
        p for p in base_dir.glob("sample_*") if p.is_dir() and (p / "experiment.h5").exists()
    )


def discover_sample_dirs(root, topologies, amplitude, sample_arg="all", skip_missing=True):
    records = []
    missing = []
    for topology in topologies:
        base_dir = Path(root) / topology / amplitude
        samples = find_sample_dirs(base_dir, sample_arg) if base_dir.exists() else []
        if not samples:
            missing.append(str(base_dir))
            if not skip_missing:
                raise FileNotFoundError(f"No sample_*/experiment.h5 files found under {base_dir}")
            continue
        records.extend((topology, sample_dir) for sample_dir in samples)
    if not records:
        raise FileNotFoundError(
            "No experiment.h5 files found for the requested topology/amplitude selection."
        )
    return records, missing


def read_csv(path):
    with Path(path).open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {out_path}")
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(obj, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


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


def frame_counts(loader, washout_s, train_s, test_s):
    washout_frames = int(washout_s / loader.dt)
    train_frames = int(train_s / loader.dt)
    test_frames = int(test_s / loader.dt)
    train_start = washout_frames
    train_stop = train_start + train_frames
    test_stop = train_stop + test_frames
    if test_stop > loader.total_frames:
        raise ValueError(
            f"Need {test_stop} frames for washout/train/test, "
            f"but only {loader.total_frames} frames are available."
        )
    return washout_frames, train_frames, test_frames, train_start, train_stop, test_stop


def basis_window(loader, washout_s, train_s, test_s, horizon_steps):
    washout_frames, train_frames, test_frames, _, _, _ = frame_counts(
        loader, washout_s, train_s, test_s
    )
    window_start = washout_frames
    window_stop = washout_frames + train_frames + test_frames
    if window_stop + horizon_steps > loader.total_frames:
        raise ValueError(
            f"Horizon requires target frame {window_stop + horizon_steps}, "
            f"but sample has only {loader.total_frames} frames."
        )
    return window_start, window_stop


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


def build_legendre_design(u_norm, H, D, lag_stride_frames=1):
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


def target_time_delays(H, horizon_steps, lag_stride_frames=1):
    return [horizon_steps + q * lag_stride_frames for q in range(H + 1)]


def term_metadata(exps, horizon_steps, lag_stride_frames=1):
    rows = []
    for basis_index, exp_vec in enumerate(exps):
        active = np.where(exp_vec > 0)[0]
        max_readout_delay = int(np.max(active)) if len(active) else 0
        total_degree = int(np.sum(exp_vec))
        parts = []
        for q, degree in enumerate(exp_vec):
            if degree > 0:
                target_delay = horizon_steps + q * lag_stride_frames
                parts.append(f"P{int(degree)}(u[target-{target_delay}])")
        rows.append(
            {
                "basis_index": basis_index,
                "basis_term": " * ".join(parts),
                "total_degree": total_degree,
                "max_readout_time_delay": max_readout_delay,
                "max_target_time_delay": horizon_steps + max_readout_delay * lag_stride_frames,
                "max_lag_frames": max_readout_delay * lag_stride_frames,
                "exponents": " ".join(str(int(v)) for v in exp_vec),
            }
        )
    return rows


def valid_basis_rows(window_start, window_stop, H, lag_stride_frames):
    max_lag = H * lag_stride_frames
    valid_start = window_start + max_lag
    if valid_start >= window_stop:
        raise ValueError(
            f"No valid rows for H={H}, lag stride={lag_stride_frames}, "
            f"window=({window_start}, {window_stop})."
        )
    return np.arange(valid_start, window_stop)


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


def scalar_r2(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    nmse = float(
        np.sum((y_true - y_pred) ** 2)
        / np.maximum(np.sum((y_true - np.mean(y_true)) ** 2), np.finfo(float).eps)
    )
    return 1.0 - nmse, nmse


def make_blocked_folds(n_rows, cv_blocks):
    indices = np.arange(n_rows)
    blocks = np.array_split(indices, cv_blocks)
    folds = []
    for block in blocks:
        if len(block) == 0:
            continue
        train_idx = np.setdiff1d(indices, block, assume_unique=True)
        folds.append((train_idx, block))
    if len(folds) < 2:
        raise ValueError("Not enough rows for blocked cross-validation.")
    return folds


def select_alpha_by_blocked_cv(P, Y, alpha_grid=ALPHA_GRID, cv_blocks=4):
    folds = make_blocked_folds(len(P), cv_blocks)
    scores = []
    for alpha in alpha_grid:
        vals = []
        for train_idx, val_idx in folds:
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(P[train_idx], Y[train_idx])
            vals.append(nmse_components(Y[val_idx], model.predict(P[val_idx]))[2])
        scores.append((float(np.mean(vals)), float(alpha)))
    return min(scores, key=lambda item: item[0])[1], folds


def basis_adequacy_label(nmse_2d):
    if nmse_2d < 0.10:
        return "strong"
    if nmse_2d < 0.20:
        return "acceptable"
    if nmse_2d < 0.35:
        return "partial"
    return "weak"


def visible_state_design(sample_dir, hidden_node, reference_node, global_standardize=True):
    loader = StateLoader(Path(sample_dir) / "experiment.h5")
    with h5py.File(loader.sim_path, "r") as f:
        n_nodes = f["time_series/nodes/positions"].shape[1]
    if hidden_node == reference_node:
        raise ValueError("Hidden node cannot also be the reference node.")
    if hidden_node >= n_nodes or reference_node >= n_nodes:
        raise ValueError(f"Node index outside available range 0..{n_nodes - 1}.")

    state_nodes = [node for node in range(n_nodes) if node not in {hidden_node, reference_node}]
    features = NodeDisplacements(reference_node=reference_node, node_ids=state_nodes, dims=[0, 1])
    X_full = features.transform(loader)
    if global_standardize:
        X_full = StandardScaler().fit_transform(X_full)
    X_design = np.hstack([np.ones((X_full.shape[0], 1)), X_full])
    return loader, X_design, state_nodes, X_full.shape[1]


def parse_csv_paths(value):
    if not value:
        return []
    return [Path(p.strip()) for p in value.split(",") if p.strip()]


def topology_index(topology):
    try:
        return int(str(topology).split("_")[1])
    except Exception:
        return str(topology)
