import argparse
import itertools
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
    output_dir as legendre_output_dir,
    parse_exclusions,
    write_csv,
    write_json,
)


try:
    from scipy.signal import coherence as scipy_coherence
    from scipy.signal import find_peaks as scipy_find_peaks

    SCIPY_SIGNAL_AVAILABLE = True
except Exception:
    scipy_coherence = None
    scipy_find_peaks = None
    SCIPY_SIGNAL_AVAILABLE = False


DEFAULT_RIDGE_ALPHA = 1e-6
DEFAULT_TOLERANCE_FRAC = 0.05
DEFAULT_CONDITION_THRESHOLD = 1e12
DEFAULT_TARGET_COMPONENT = "y"
DEFAULT_FREQUENCY_SOURCE = "hidden_target"
DEFAULT_MAX_FREQUENCIES = 5
DEFAULT_MIN_FREQUENCY_HZ = 0.2
DEFAULT_MAX_FREQUENCY_HZ = "auto"
DEFAULT_MIN_PEAK_DISTANCE_HZ = 0.2
DEFAULT_PEAK_PROMINENCE_FRAC = 0.05
DEFAULT_COHERENCE_THRESHOLD = 0.1
DEFAULT_COHERENCE_NPERSEG = 256
DEFAULT_FILTER_LENGTHS_S = [0.5, 1.0, 2.0, 3.0]
DEFAULT_DECAY_TIMES_S = [0.2, 0.5, 1.0, 2.0, 5.0]
DEFAULT_SVD_RCOND = 1e-8
DEFAULT_SVD_ENERGY = 0.999
DEFAULT_MAX_SVD_MODES = 50
DEFAULT_SELECTION_METRIC = "validation_nmse_2d"
DEFAULT_FREQUENCY_REFINEMENT = True
DEFAULT_REFINE_TARGET_HZ = 5.9
DEFAULT_REFINE_HALF_WIDTH_HZ = 0.5
DEFAULT_REFINE_STEP_HZ = 0.01
DEFAULT_DRIVE_CHANNELS = ["u"]
DEFAULT_DRIVE_SMOOTHING_FRAMES = 1
DEFAULT_MECHANICAL_DEGREE = 1
DEFAULT_MAX_NONLINEAR_FEATURES = 5000


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Mechanics-informed hidden-node basis calibration using damped sinusoidal "
            "response features instead of raw delayed Legendre products."
        )
    )
    parser.add_argument("--topology", default=None)
    parser.add_argument("--topologies", nargs="+", default=None)
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
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--tolerance-frac", type=float, default=DEFAULT_TOLERANCE_FRAC)
    parser.add_argument("--condition-threshold", type=float, default=DEFAULT_CONDITION_THRESHOLD)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-component", choices=["x", "y", "2d"], default=DEFAULT_TARGET_COMPONENT)
    parser.add_argument(
        "--frequency-source",
        choices=["hidden_target", "visible_mean_y", "visible_pca1_y"],
        default=DEFAULT_FREQUENCY_SOURCE,
    )
    parser.add_argument("--max-frequencies", type=int, default=DEFAULT_MAX_FREQUENCIES)
    parser.add_argument("--min-frequency-hz", type=float, default=DEFAULT_MIN_FREQUENCY_HZ)
    parser.add_argument("--max-frequency-hz", default=DEFAULT_MAX_FREQUENCY_HZ)
    parser.add_argument("--min-peak-distance-hz", type=float, default=DEFAULT_MIN_PEAK_DISTANCE_HZ)
    parser.add_argument("--peak-prominence-frac", type=float, default=DEFAULT_PEAK_PROMINENCE_FRAC)
    parser.add_argument("--coherence-threshold", type=float, default=DEFAULT_COHERENCE_THRESHOLD)
    parser.add_argument("--coherence-nperseg", type=int, default=DEFAULT_COHERENCE_NPERSEG)
    parser.add_argument("--filter-lengths-s", type=float, nargs="+", default=DEFAULT_FILTER_LENGTHS_S)
    parser.add_argument("--decay-times-s", type=float, nargs="+", default=DEFAULT_DECAY_TIMES_S)
    parser.add_argument("--svd-rcond", type=float, default=DEFAULT_SVD_RCOND)
    parser.add_argument("--svd-energy", type=float, default=DEFAULT_SVD_ENERGY)
    parser.add_argument("--max-svd-modes", type=int, default=DEFAULT_MAX_SVD_MODES)
    parser.add_argument(
        "--selection-metric",
        choices=["validation_nmse_x", "validation_nmse_y", "validation_nmse_2d"],
        default=DEFAULT_SELECTION_METRIC,
    )
    parser.add_argument(
        "--mechanical-degree",
        type=int,
        default=DEFAULT_MECHANICAL_DEGREE,
        help=(
            "Maximum total degree for products of standardized mechanical response "
            "coordinates. Degree 1 is the linear response basis."
        ),
    )
    parser.add_argument(
        "--max-nonlinear-features",
        type=int,
        default=DEFAULT_MAX_NONLINEAR_FEATURES,
        help="Skip a candidate if the nonlinear product expansion would exceed this many terms.",
    )
    parser.add_argument(
        "--include-nonlinear-products",
        action="store_true",
        help=(
            "Compatibility shortcut. If --mechanical-degree is left at 1, this uses "
            "degree 2 products."
        ),
    )
    parser.add_argument(
        "--frequency-refinement",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FREQUENCY_REFINEMENT,
        help=(
            "Add dense single-frequency response candidates around --refine-target-hz. "
            "Use --no-frequency-refinement to keep only FFT peak prefixes."
        ),
    )
    parser.add_argument("--refine-target-hz", type=float, default=DEFAULT_REFINE_TARGET_HZ)
    parser.add_argument("--refine-half-width-hz", type=float, default=DEFAULT_REFINE_HALF_WIDTH_HZ)
    parser.add_argument("--refine-step-hz", type=float, default=DEFAULT_REFINE_STEP_HZ)
    parser.add_argument(
        "--drive-channels",
        nargs="+",
        choices=["u", "du", "ddu"],
        default=DEFAULT_DRIVE_CHANNELS,
        help=(
            "Compatibility option. The mechanical response basis is displacement-only; "
            "only u is used even if du or ddu are listed."
        ),
    )
    parser.add_argument(
        "--drive-smoothing-frames",
        type=int,
        default=DEFAULT_DRIVE_SMOOTHING_FRAMES,
        help=(
            "Compatibility option retained for old commands. Ignored because derivative "
            "drive channels are disabled."
        ),
    )
    parser.add_argument(
        "--sweep-drive-channel-sets",
        action="store_true",
        help=(
            "Compatibility option retained for old commands. Ignored because the basis "
            "uses only the displacement-like u drive."
        ),
    )
    return parser.parse_args()


def mechanical_output_dir(args, records, excluded_samples):
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
        / "mechanical_response_basis_calibration"
        / label
        / f"hidden_node_{args.hidden_node}"
        / f"horizon_{args.horizon_steps}_steps"
    )
    if excluded_samples:
        parts = []
        for topology in sorted(excluded_samples):
            for sample in sorted(excluded_samples[topology]):
                parts.append(f"{topology}_{sample}".replace("/", "_"))
        base = base / ("exclude_" + "__".join(parts))
    return base


def hann_fft(signal, dt):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    signal = signal - np.mean(signal)
    window = np.hanning(len(signal))
    spectrum = np.fft.rfft(signal * window)
    freq = np.fft.rfftfreq(len(signal), dt)
    amplitude = 2.0 * np.abs(spectrum) / max(float(np.sum(window)), np.finfo(float).eps)
    power = amplitude**2
    return freq, amplitude, power


def normalize_for_plot(values):
    values = np.asarray(values, dtype=float)
    max_value = np.nanmax(values) if len(values) else 0.0
    if max_value <= 0:
        return values
    return values / max_value


def hidden_and_source_signals(record, args, rows):
    sample_dir = record["sample_dir"]
    time, positions = load_positions(sample_dir / "experiment.h5")
    hidden_relative = positions[:, args.hidden_node, :] - positions[:, args.reference_node, :]
    Y = hidden_relative[rows + args.horizon_steps]
    if args.frequency_source == "hidden_target":
        source_xy = Y
    else:
        n_nodes = positions.shape[1]
        visible_nodes = [n for n in range(n_nodes) if n not in {args.hidden_node, args.reference_node}]
        visible_y = positions[rows + args.horizon_steps][:, visible_nodes, 1] - positions[
            rows + args.horizon_steps, args.reference_node, 1
        ][:, None]
        if args.frequency_source == "visible_mean_y":
            y = np.mean(visible_y, axis=1)
        else:
            visible_y = visible_y - np.mean(visible_y, axis=0, keepdims=True)
            U, _, _ = np.linalg.svd(visible_y, full_matrices=False)
            y = U[:, 0]
        source_xy = np.column_stack([Y[:, 0], y])

    if args.target_component == "x":
        signal = source_xy[:, 0]
    elif args.target_component == "y":
        signal = source_xy[:, 1]
    else:
        x = source_xy[:, 0] / max(np.std(source_xy[:, 0]), np.finfo(float).eps)
        y = source_xy[:, 1] / max(np.std(source_xy[:, 1]), np.finfo(float).eps)
        signal = x + y
    return time, positions, hidden_relative, Y, signal


def simple_find_peaks(amplitude, min_distance_bins, prominence):
    peaks = []
    last_peak = -min_distance_bins
    for i in range(1, len(amplitude) - 1):
        if amplitude[i] > amplitude[i - 1] and amplitude[i] >= amplitude[i + 1]:
            if amplitude[i] < prominence:
                continue
            if i - last_peak < min_distance_bins:
                if peaks and amplitude[i] > amplitude[peaks[-1]]:
                    peaks[-1] = i
                    last_peak = i
                continue
            peaks.append(i)
            last_peak = i
    return np.asarray(peaks, dtype=int)


def select_frequencies(record, args, loader, positions, hidden_relative, train_rows, drives_z):
    dt = loader.dt
    _, _, _, _, target_train_signal = hidden_and_source_signals(record, args, train_rows)
    freq_t, amp_t, power_t = hann_fft(target_train_signal, dt)
    drive_fft = {}
    for drive_name, drive_signal in drives_z.items():
        freq_d, amp_d, power_d = hann_fft(drive_signal[train_rows], dt)
        drive_fft[drive_name] = {"freq": freq_d, "amp": amp_d, "power": power_d}
    nyquist = 0.5 / dt
    max_freq = 0.45 * nyquist if str(args.max_frequency_hz).lower() == "auto" else float(args.max_frequency_hz)

    coherence_available = SCIPY_SIGNAL_AVAILABLE
    coherence_by_drive = {}
    if coherence_available:
        nperseg = min(args.coherence_nperseg, len(target_train_signal))
        if nperseg >= 8:
            for drive_name, drive_signal in drives_z.items():
                coh_freq, coh_values = scipy_coherence(
                    drive_signal[train_rows],
                    target_train_signal,
                    fs=1.0 / dt,
                    window="hann",
                    nperseg=nperseg,
                    noverlap=nperseg // 2,
                )
                coherence_by_drive[drive_name] = {"freq": coh_freq, "coherence": coh_values}
        else:
            coherence_available = False

    valid = (
        (freq_t > 0)
        & (freq_t >= args.min_frequency_hz)
        & (freq_t <= max_freq)
        & np.isfinite(amp_t)
    )
    prominence = args.peak_prominence_frac * max(float(np.max(amp_t[valid])) if np.any(valid) else 0.0, 1e-12)
    df = float(freq_t[1] - freq_t[0]) if len(freq_t) > 1 else max(args.min_peak_distance_hz, 1.0)
    min_distance_bins = max(1, int(round(args.min_peak_distance_hz / max(df, np.finfo(float).eps))))
    if SCIPY_SIGNAL_AVAILABLE:
        peak_idx, _ = scipy_find_peaks(amp_t, distance=min_distance_bins, prominence=prominence)
    else:
        peak_idx = simple_find_peaks(amp_t, min_distance_bins, prominence)
    peak_idx = np.asarray([idx for idx in peak_idx if valid[idx]], dtype=int)
    if len(peak_idx) == 0 and np.any(valid):
        peak_idx = np.asarray(np.where(valid)[0], dtype=int)

    max_amp = max(float(np.max(amp_t[peak_idx])) if len(peak_idx) else 0.0, np.finfo(float).eps)
    rows = []
    for idx in peak_idx:
        freq = float(freq_t[idx])
        drive_amp_u = float(np.interp(freq, drive_fft["u"]["freq"], drive_fft["u"]["amp"]))
        if coherence_available and "u" in coherence_by_drive:
            coherence_u = float(
                np.interp(
                    freq,
                    coherence_by_drive["u"]["freq"],
                    coherence_by_drive["u"]["coherence"],
                )
            )
        else:
            coherence_u = np.nan
        best_drive_channel = "u" if np.isfinite(coherence_u) else ""
        best_coherence = coherence_u
        norm_amp = float(amp_t[idx] / max_amp)
        score = norm_amp * best_coherence if np.isfinite(best_coherence) else norm_amp
        rows.append(
            {
                "topology": record["topology"],
                "amplitude": record["amplitude"],
                "sample": record["sample"],
                "frequency_hz": freq,
                "target_fft_amplitude": float(amp_t[idx]),
                "drive_fft_amplitude_u": drive_amp_u,
                "coherence_u": coherence_u,
                "best_drive_channel": best_drive_channel,
                "best_coherence": best_coherence,
                "score": float(score),
                "selected": False,
            }
        )
    coherent_rows = [
        r
        for r in rows
        if np.isfinite(r["best_coherence"]) and r["best_coherence"] >= args.coherence_threshold
    ]
    primary = coherent_rows if len(coherent_rows) >= args.max_frequencies else rows
    ranked = sorted(primary, key=lambda r: r["score"], reverse=True)
    if len(ranked) < args.max_frequencies:
        fill = sorted(rows, key=lambda r: r["target_fft_amplitude"], reverse=True)
        seen = {r["frequency_hz"] for r in ranked}
        ranked.extend([r for r in fill if r["frequency_hz"] not in seen])
    selected = ranked[: args.max_frequencies]
    selected_freqs = [float(r["frequency_hz"]) for r in selected]
    selected_set = set(selected_freqs)
    for row in rows:
        row["selected"] = row["frequency_hz"] in selected_set
    fft_info = {
        "freq_target": freq_t,
        "amp_target": amp_t,
        "power_target": power_t,
        "drive_fft": drive_fft,
        "coherence_by_drive": coherence_by_drive,
        "coherence_available": bool(coherence_available),
        "target_train_signal": target_train_signal,
        "drive_train_signals": {name: signal[train_rows] for name, signal in drives_z.items()},
    }
    return selected_freqs, rows, fft_info


def standardize_input_from_train(u, train_rows):
    mean = float(np.mean(u[train_rows]))
    std = float(np.std(u[train_rows]))
    if std < 1e-12:
        raise ValueError("Measured input has near-zero training-window standard deviation.")
    return (u - mean) / std, mean, std


def standardize_drive_channels_from_train(drives, train_rows, selected_channels):
    out = {}
    stats = {}
    for name in selected_channels:
        x = np.asarray(drives[name], dtype=float)
        mean = float(np.mean(x[train_rows]))
        std = float(np.std(x[train_rows]))
        if std < 1e-12:
            print(f"Warning: drive channel {name} has near-zero training std and will be skipped.")
            continue
        out[name] = (x - mean) / std
        stats[name] = {"mean": mean, "std": std}
    if not out:
        raise ValueError("No valid drive channels after standardization.")
    return out, stats


def drive_channel_sets(args):
    return [("u",)]


def refinement_frequency_grid(args):
    if not args.frequency_refinement:
        return []
    if args.refine_step_hz <= 0:
        raise ValueError("--refine-step-hz must be positive.")
    start = args.refine_target_hz - args.refine_half_width_hz
    stop = args.refine_target_hz + args.refine_half_width_hz
    n_steps = int(np.floor((stop - start) / args.refine_step_hz + 0.5))
    grid = start + args.refine_step_hz * np.arange(n_steps + 1)
    grid = grid[(grid > 0) & np.isfinite(grid)]
    return [float(f) for f in grid]


def candidate_frequency_sets(selected_freqs, args):
    sets = []
    for K in range(1, min(args.max_frequencies, len(selected_freqs)) + 1):
        sets.append(
            {
                "frequency_set_type": "fft_prefix",
                "frequency_set_label": f"fft_prefix_K{K}",
                "K": K,
                "frequencies": [float(f) for f in selected_freqs[:K]],
                "refined_frequency_hz": np.nan,
            }
        )
    for freq in refinement_frequency_grid(args):
        sets.append(
            {
                "frequency_set_type": "refined_single",
                "frequency_set_label": f"refined_{freq:.4f}Hz",
                "K": 1,
                "frequencies": [float(freq)],
                "refined_frequency_hz": float(freq),
            }
        )
    return sets


def build_mechanical_features(drives_z, dt, frequencies, decay_times_s, filter_length_s):
    L_frames = int(round(filter_length_s / dt))
    tau = np.arange(L_frames + 1) * dt
    features = []
    metadata = []
    for drive_name, drive_signal in drives_z.items():
        for freq in frequencies:
            for decay in decay_times_s:
                envelope = np.exp(-tau / decay)
                for phase in ("cos", "sin"):
                    if phase == "cos":
                        kernel = envelope * np.cos(2.0 * np.pi * freq * tau)
                    else:
                        kernel = envelope * np.sin(2.0 * np.pi * freq * tau)
                    kernel = kernel / np.sqrt(np.sum(kernel**2) + np.finfo(float).eps)
                    feature = np.convolve(drive_signal, kernel, mode="full")[: len(drive_signal)]
                    features.append(feature)
                    metadata.append(
                        {
                            "raw_feature_name": f"{drive_name}_f={freq:.4f}Hz_Td={decay:.3f}s_{phase}",
                            "drive_channel": drive_name,
                            "frequency_hz": float(freq),
                            "decay_time_s": float(decay),
                            "phase": phase,
                        }
                    )
    if not features:
        raise ValueError("No mechanical response features were created.")
    return np.column_stack(features), metadata, L_frames


def standardize_features(Psi_raw, train_rows):
    mean = np.mean(Psi_raw[train_rows], axis=0, keepdims=True)
    std = np.std(Psi_raw[train_rows], axis=0, keepdims=True)
    keep = std.reshape(-1) > 1e-10
    if not np.any(keep):
        raise ValueError("All mechanical response features have near-zero variance.")
    return (Psi_raw[:, keep] - mean[:, keep]) / std[:, keep], keep


def effective_mechanical_degree(args):
    if args.mechanical_degree < 1:
        raise ValueError("--mechanical-degree must be >= 1.")
    degree = int(args.mechanical_degree)
    if args.include_nonlinear_products:
        degree = max(2, degree)
    return degree


def product_metadata(combo, metadata):
    factors = [metadata[i] for i in combo]
    if len(combo) == 1:
        meta = dict(factors[0])
    else:
        factor_names = [m["raw_feature_name"] for m in factors]
        meta = {
            "raw_feature_name": " * ".join(factor_names),
            "drive_channel": "*".join(m["drive_channel"] for m in factors),
            "frequency_hz": "*".join(str(m["frequency_hz"]) for m in factors),
            "decay_time_s": "*".join(str(m["decay_time_s"]) for m in factors),
            "phase": "*".join(m["phase"] for m in factors),
        }
    meta["product_degree"] = int(len(combo))
    meta["factor_indices"] = " ".join(str(i) for i in combo)
    meta["number_factors"] = int(len(combo))
    return meta


def expand_mechanical_products(Psi_linear, linear_meta, train_rows, args):
    degree = effective_mechanical_degree(args)
    if degree == 1:
        metadata = []
        for i, meta in enumerate(linear_meta):
            item = dict(meta)
            item["product_degree"] = 1
            item["factor_indices"] = str(i)
            item["number_factors"] = 1
            metadata.append(item)
        return Psi_linear, metadata, len(metadata), len(metadata)

    n_base = Psi_linear.shape[1]
    combos = []
    for product_degree in range(1, degree + 1):
        combos.extend(itertools.combinations_with_replacement(range(n_base), product_degree))
        if len(combos) > args.max_nonlinear_features:
            raise ValueError(
                f"Nonlinear product expansion would create {len(combos)} terms "
                f"(limit {args.max_nonlinear_features}). Reduce --mechanical-degree, "
                "reduce the frequency/decay grid, or increase --max-nonlinear-features."
            )

    products = np.empty((Psi_linear.shape[0], len(combos)), dtype=np.float32)
    metadata = []
    for col, combo in enumerate(combos):
        term = np.ones(Psi_linear.shape[0], dtype=np.float32)
        for idx in combo:
            term *= Psi_linear[:, idx].astype(np.float32, copy=False)
        products[:, col] = term
        metadata.append(product_metadata(combo, linear_meta))

    Psi_products, keep = standardize_features(products, train_rows)
    kept_meta = [meta for meta, use in zip(metadata, keep) if use]
    return Psi_products, kept_meta, n_base, len(combos)


def svd_project(Psi_train, Psi_val, args):
    U, singular_values, Vt = np.linalg.svd(Psi_train, full_matrices=False)
    if len(singular_values) == 0:
        raise ValueError("SVD returned no singular values.")
    rcond_rank = int(np.sum(singular_values > args.svd_rcond * singular_values[0]))
    energy = np.cumsum(singular_values**2) / max(np.sum(singular_values**2), np.finfo(float).eps)
    energy_rank = int(np.searchsorted(energy, args.svd_energy) + 1)
    rank = max(1, min(rcond_rank, energy_rank, args.max_svd_modes, len(singular_values)))
    Q_train = U[:, :rank]
    Q_val = Psi_val @ Vt[:rank, :].T / singular_values[:rank]
    condition_number = float(singular_values[0] / max(singular_values[rank - 1], np.finfo(float).eps))
    return Q_train, Q_val, singular_values, Vt, rank, condition_number


def fit_response_basis(Q_train, Q_val, Y_train, Y_val, ridge_alpha):
    y_mean = np.mean(Y_train, axis=0, keepdims=True)
    model = Ridge(alpha=ridge_alpha, fit_intercept=False, solver="svd")
    model.fit(Q_train, Y_train - y_mean)
    train_pred = model.predict(Q_train) + y_mean
    val_pred = model.predict(Q_val) + y_mean
    train_x, train_y, train_2d = nmse_components(Y_train, train_pred)
    val_x, val_y, val_2d = nmse_components(Y_val, val_pred)
    return model, y_mean, train_pred, val_pred, train_x, train_y, train_2d, val_x, val_y, val_2d


def evaluate_candidate(record, args, frequency_set, filter_length_s, drive_set):
    sample_dir = record["sample_dir"]
    loader = StateLoader(sample_dir / "experiment.h5")
    _, positions = load_positions(sample_dir / "experiment.h5")
    u = loader.get_actuation_signal(actuator_idx=0, dof=0)
    hidden_relative = positions[:, args.hidden_node, :] - positions[:, args.reference_node, :]
    train_start, train_stop, validation_stop = frame_windows(loader, args)
    base_train_rows = np.arange(train_start, train_stop)
    drives = {"u": np.asarray(u, dtype=float)}
    drives_z, drive_stats = standardize_drive_channels_from_train(
        drives, base_train_rows, drive_set
    )
    frequencies = [float(f) for f in frequency_set["frequencies"]]
    K = int(frequency_set["K"])
    Psi_raw, raw_meta, L_frames = build_mechanical_features(
        drives_z, loader.dt, frequencies, args.decay_times_s, filter_length_s
    )
    train_rows = np.arange(max(train_start, L_frames), train_stop)
    val_rows = np.arange(max(train_stop, L_frames), validation_stop)
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise ValueError(f"No valid train/validation rows for filter length {filter_length_s}.")
    Psi_linear, keep = standardize_features(Psi_raw, train_rows)
    kept_linear_meta = [meta for meta, use in zip(raw_meta, keep) if use]
    Psi, kept_meta, number_linear_features, number_candidate_features = expand_mechanical_products(
        Psi_linear, kept_linear_meta, train_rows, args
    )
    Psi_train = Psi[train_rows]
    Psi_val = Psi[val_rows]
    Y_train = hidden_relative[train_rows + args.horizon_steps]
    Y_val = hidden_relative[val_rows + args.horizon_steps]
    Q_train, Q_val, singular_values, Vt, rank, condition_number = svd_project(Psi_train, Psi_val, args)
    model, y_mean, _, val_pred, train_x, train_y, train_2d, val_x, val_y, val_2d = fit_response_basis(
        Q_train, Q_val, Y_train, Y_val, args.ridge_alpha
    )
    row = {
        "topology": record["topology"],
        "amplitude": record["amplitude"],
        "sample": record["sample"],
        "K": int(K),
        "frequency_set_type": frequency_set["frequency_set_type"],
        "frequency_set_label": frequency_set["frequency_set_label"],
        "refined_frequency_hz": frequency_set["refined_frequency_hz"],
        "drive_channels": " ".join(drive_set),
        "drive_smoothing_frames": int(args.drive_smoothing_frames),
        "number_drive_channels": int(len(drives_z)),
        "filter_length_s": float(filter_length_s),
        "filter_length_frames": int(L_frames),
        "selected_frequencies_hz": " ".join(f"{f:.8g}" for f in frequencies),
        "decay_times_s": " ".join(f"{d:.8g}" for d in args.decay_times_s),
        "mechanical_degree": int(effective_mechanical_degree(args)),
        "number_linear_features": int(number_linear_features),
        "number_candidate_features": int(number_candidate_features),
        "number_raw_features": int(len(kept_meta)),
        "svd_rank": int(rank),
        "condition_number": condition_number,
        "train_nmse_x": train_x,
        "train_nmse_y": train_y,
        "train_nmse_2d": train_2d,
        "validation_nmse_x": val_x,
        "validation_nmse_y": val_y,
        "validation_nmse_2d": val_2d,
        "selection_metric_value": {"validation_nmse_x": val_x, "validation_nmse_y": val_y, "validation_nmse_2d": val_2d}[
            args.selection_metric
        ],
    }
    cache = {
        "loader": loader,
        "positions": positions,
        "hidden_relative": hidden_relative,
        "drive_stats": drive_stats,
        "drive_set": tuple(drives_z.keys()),
        "raw_meta": kept_meta,
        "Psi": Psi,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "Y_train": Y_train,
        "Y_val": Y_val,
        "Q_train": Q_train,
        "Q_val": Q_val,
        "singular_values": singular_values,
        "Vt": Vt,
        "rank": rank,
        "model": model,
        "y_mean": y_mean,
        "val_pred": val_pred,
    }
    return row, cache


def summarize_candidates(rows):
    summary = []
    keys = sorted(
        {
            (
                r["frequency_set_label"],
                int(r["K"]),
                float(r["filter_length_s"]),
                r["drive_channels"],
            )
            for r in rows
        },
        key=lambda item: (len(item[3].split()), item[1], item[2], item[0]),
    )
    for frequency_set_label, K, filter_length_s, drive_channels in keys:
        group = [
            r
            for r in rows
            if r["frequency_set_label"] == frequency_set_label
            and int(r["K"]) == K
            and float(r["filter_length_s"]) == filter_length_s
            and r["drive_channels"] == drive_channels
        ]
        row = {
            "frequency_set_type": group[0]["frequency_set_type"],
            "frequency_set_label": frequency_set_label,
            "refined_frequency_hz": group[0]["refined_frequency_hz"],
            "drive_channels": drive_channels,
            "drive_smoothing_frames": int(group[0]["drive_smoothing_frames"]),
            "number_drive_channels": int(group[0]["number_drive_channels"]),
            "K": K,
            "filter_length_s": filter_length_s,
            "filter_length_frames": int(group[0]["filter_length_frames"]),
            "num_topology_sample_pairs": len(group),
            "selected_frequencies_hz": group[0]["selected_frequencies_hz"],
            "decay_times_s": group[0]["decay_times_s"],
            "mechanical_degree": int(group[0]["mechanical_degree"]),
        }
        for key in (
            "number_linear_features",
            "number_candidate_features",
            "number_raw_features",
            "number_drive_channels",
            "svd_rank",
            "condition_number",
            "train_nmse_x",
            "train_nmse_y",
            "train_nmse_2d",
            "validation_nmse_x",
            "validation_nmse_y",
            "validation_nmse_2d",
            "selection_metric_value",
        ):
            values = np.asarray([float(r[key]) for r in group], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(values))
            row[f"{key}_std"] = float(np.nanstd(values, ddof=0))
        summary.append(row)
    return summary


def summarize_by_topology(rows):
    summary = []
    keys = sorted(
        {
            (
                r["topology"],
                r["frequency_set_label"],
                int(r["K"]),
                float(r["filter_length_s"]),
                r["drive_channels"],
            )
            for r in rows
        }
    )
    for topology, frequency_set_label, K, filter_length_s, drive_channels in keys:
        group = [
            r
            for r in rows
            if r["topology"] == topology
            and r["frequency_set_label"] == frequency_set_label
            and int(r["K"]) == K
            and float(r["filter_length_s"]) == filter_length_s
            and r["drive_channels"] == drive_channels
        ]
        row = {
            "topology": topology,
            "amplitude": group[0]["amplitude"],
            "frequency_set_type": group[0]["frequency_set_type"],
            "frequency_set_label": frequency_set_label,
            "refined_frequency_hz": group[0]["refined_frequency_hz"],
            "drive_channels": drive_channels,
            "number_drive_channels": int(group[0]["number_drive_channels"]),
            "K": K,
            "filter_length_s": filter_length_s,
            "num_samples": len(group),
            "selected_frequencies_hz": group[0]["selected_frequencies_hz"],
            "mechanical_degree": int(group[0]["mechanical_degree"]),
        }
        for key in (
            "number_linear_features",
            "number_candidate_features",
            "number_raw_features",
            "validation_nmse_x",
            "validation_nmse_y",
            "validation_nmse_2d",
            "condition_number",
        ):
            values = np.asarray([float(r[key]) for r in group], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(values))
            row[f"{key}_std"] = float(np.nanstd(values, ddof=0))
        summary.append(row)
    return summary


def choose_candidate(summary, args):
    metric = "selection_metric_value_mean"
    best = min(summary, key=lambda r: float(r[metric]))
    if args.frequency_refinement:
        return best, best, float(best[metric])
    tolerance = (1.0 + args.tolerance_frac) * float(best[metric])
    eligible = [r for r in summary if float(r[metric]) <= tolerance]
    non_ill = [r for r in eligible if float(r["condition_number_mean"]) <= args.condition_threshold]
    if non_ill:
        eligible = non_ill
    selected = min(
        eligible,
        key=lambda r: (
            int(r["number_drive_channels"]),
            0 if r["frequency_set_type"] == "fft_prefix" else 1,
            int(r["K"]),
            float(r["filter_length_s"]),
            int(round(float(r["svd_rank_mean"]))),
            float(r["number_raw_features_mean"]),
        ),
    )
    return selected, best, tolerance


def selected_frequency_set_for_record(record, selected, selected_freqs_by_record):
    K = int(selected["K"])
    if selected["frequency_set_type"] == "refined_single":
        frequencies = [float(selected["refined_frequency_hz"])]
    else:
        frequencies = selected_freqs_by_record[(record["topology"], record["sample"])][:K]
    return {
        "frequency_set_type": selected["frequency_set_type"],
        "frequency_set_label": selected["frequency_set_label"],
        "K": K,
        "frequencies": [float(f) for f in frequencies],
        "refined_frequency_hz": float(selected["refined_frequency_hz"])
        if str(selected["refined_frequency_hz"]) != "nan"
        else np.nan,
    }


def selected_drive_set(selected):
    return tuple(str(selected["drive_channels"]).split())


def selected_validation_trace(record, args, selected, selected_freqs_by_record):
    frequency_set = selected_frequency_set_for_record(record, selected, selected_freqs_by_record)
    drive_set = selected_drive_set(selected)
    row, cache = evaluate_candidate(
        record, args, frequency_set, float(selected["filter_length_s"]), drive_set
    )
    loader = cache["loader"]
    time, _ = load_positions(record["sample_dir"] / "experiment.h5")
    target_time = time[cache["val_rows"] + args.horizon_steps]
    target_time = target_time - target_time[0]
    return {
        "topology": record["topology"],
        "amplitude": record["amplitude"],
        "sample": record["sample"],
        "time": target_time,
        "target": cache["Y_val"],
        "predicted": cache["val_pred"],
        "validation_nmse_x": row["validation_nmse_x"],
        "validation_nmse_y": row["validation_nmse_y"],
        "validation_nmse_2d": row["validation_nmse_2d"],
    }


def final_weights(records, args, selected_freqs_by_record, selected):
    weights = []
    reliability = []
    loadings = []
    K = int(selected["K"])
    filter_length_s = float(selected["filter_length_s"])
    drive_set = selected_drive_set(selected)
    for record in records:
        sample_dir = record["sample_dir"]
        loader = StateLoader(sample_dir / "experiment.h5")
        _, positions = load_positions(sample_dir / "experiment.h5")
        u = loader.get_actuation_signal(actuator_idx=0, dof=0)
        hidden_relative = positions[:, args.hidden_node, :] - positions[:, args.reference_node, :]
        train_start, train_stop, validation_stop = frame_windows(loader, args)
        train_rows_base = np.arange(train_start, train_stop)
        drives = {"u": np.asarray(u, dtype=float)}
        drives_z, drive_stats = standardize_drive_channels_from_train(
            drives, train_rows_base, drive_set
        )
        freqs = selected_frequency_set_for_record(record, selected, selected_freqs_by_record)[
            "frequencies"
        ]
        Psi_raw, raw_meta, L_frames = build_mechanical_features(
            drives_z, loader.dt, freqs, args.decay_times_s, filter_length_s
        )
        rows = np.arange(max(train_start, L_frames), validation_stop)
        Psi_linear, keep = standardize_features(Psi_raw, rows)
        kept_linear_meta = [meta for meta, use in zip(raw_meta, keep) if use]
        Psi, kept_meta, number_linear_features, number_candidate_features = expand_mechanical_products(
            Psi_linear, kept_linear_meta, rows, args
        )
        Y = hidden_relative[rows + args.horizon_steps]
        U, singular_values, Vt = np.linalg.svd(Psi[rows], full_matrices=False)
        rcond_rank = int(np.sum(singular_values > args.svd_rcond * singular_values[0]))
        energy = np.cumsum(singular_values**2) / max(np.sum(singular_values**2), np.finfo(float).eps)
        energy_rank = int(np.searchsorted(energy, args.svd_energy) + 1)
        rank = max(1, min(rcond_rank, energy_rank, args.max_svd_modes, len(singular_values)))
        Q = U[:, :rank]
        y_mean = np.mean(Y, axis=0, keepdims=True)
        model = Ridge(alpha=args.ridge_alpha, fit_intercept=False, solver="svd")
        model.fit(Q, Y - y_mean)
        pred = model.predict(Q) + y_mean
        nmse_x, nmse_y, nmse_2d = nmse_components(Y, pred)
        y_centered = Y - np.mean(Y, axis=0, keepdims=True)
        denom_x = float(np.sum(y_centered[:, 0] ** 2))
        denom_y = float(np.sum(y_centered[:, 1] ** 2))
        denom_2d = float(np.sum(y_centered**2))
        reliability.append(
            {
                "topology": record["topology"],
                "amplitude": record["amplitude"],
                "sample": record["sample"],
                "basis_nmse_x": nmse_x,
                "basis_nmse_y": nmse_y,
                "basis_nmse_2d": nmse_2d,
                "basis_R2_x": 1.0 - nmse_x,
                "basis_R2_y": 1.0 - nmse_y,
                "basis_R2_2d": 1.0 - nmse_2d,
                "selected_drive_channels": " ".join(drive_set),
                "drive_smoothing_frames": int(args.drive_smoothing_frames),
                "selected_K": K,
                "selected_filter_length_s": filter_length_s,
                "selected_mechanical_degree": int(effective_mechanical_degree(args)),
                "number_linear_features": int(number_linear_features),
                "number_candidate_features": int(number_candidate_features),
                "number_kept_features": int(len(kept_meta)),
                "selected_svd_rank": rank,
            }
        )
        for mode_index in range(rank):
            loading_vec = Vt[mode_index]
            dominant_idx = int(np.argmax(np.abs(loading_vec)))
            dominant = kept_meta[dominant_idx]
            q = Q[:, mode_index]
            q_norm_sq = float(np.sum(q**2))
            coef_x = float(model.coef_[0, mode_index])
            coef_y = float(model.coef_[1, mode_index])
            weights.append(
                {
                    "topology": record["topology"],
                    "amplitude": record["amplitude"],
                    "sample": record["sample"],
                    "mode_index": mode_index,
                    "coefficient_x": coef_x,
                    "coefficient_y": coef_y,
                    "q_norm_sq": q_norm_sq,
                    "c_mech_x": coef_x**2 * q_norm_sq / max(denom_x, np.finfo(float).eps),
                    "c_mech_y": coef_y**2 * q_norm_sq / max(denom_y, np.finfo(float).eps),
                    "c_mech_2d": (coef_x**2 + coef_y**2) * q_norm_sq / max(denom_2d, np.finfo(float).eps),
                    "dominant_raw_feature_name": dominant["raw_feature_name"],
                    "dominant_product_degree": dominant["product_degree"],
                    "dominant_factor_indices": dominant["factor_indices"],
                    "dominant_drive_channel": dominant["drive_channel"],
                    "dominant_frequency_hz": dominant["frequency_hz"],
                    "dominant_decay_time_s": dominant["decay_time_s"],
                    "dominant_phase": dominant["phase"],
                    "dominant_raw_feature_loading": float(loading_vec[dominant_idx]),
                }
            )
            for raw_index, meta in enumerate(kept_meta):
                loadings.append(
                    {
                        "topology": record["topology"],
                        "amplitude": record["amplitude"],
                        "sample": record["sample"],
                        "candidate_id": f"K{K}_L{filter_length_s:g}",
                        "mode_index": mode_index,
                        **meta,
                        "loading": float(loading_vec[raw_index]),
                    }
                )
    return weights, reliability, loadings


def save_with_companions(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if out_path.suffix.lower() == ".pdf":
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def panel_label(ax, label):
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=9)


def save_summary_plot(summary, selected, trace, fft_info, selected_freqs, args, out_path):
    metric_key = f"{args.selection_metric}_mean"
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.8, 7.4),
        sharex=False,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.25]},
    )
    selected_K = int(selected["K"])
    selected_filter_length = float(selected["filter_length_s"])
    selected_metric = float(selected[metric_key])
    selected_rank = int(round(float(selected["svd_rank_mean"])))
    selected_condition = float(selected["condition_number_mean"])
    selected_freq_text = selected["selected_frequencies_hz"]
    selected_drives = selected["drive_channels"]
    selected_text = (
        "Selected response basis\n"
        f"drives = {selected_drives.replace(' ', '+')}\n"
        f"degree = {int(selected['mechanical_degree'])}\n"
        f"K = {selected_K}\n"
        f"freq = {selected_freq_text} Hz\n"
        f"filter length = {selected_filter_length:g} s\n"
        f"terms = {int(round(float(selected['number_raw_features_mean'])))}\n"
        f"rank = {selected_rank}\n"
        f"val NMSE_y = {float(selected['validation_nmse_y_mean']):.4f}"
    )
    fft_rows = [r for r in summary if r["frequency_set_type"] == "fft_prefix"]
    refined_rows = [r for r in summary if r["frequency_set_type"] == "refined_single"]
    line_keys = sorted(
        {(float(r["filter_length_s"]), r["drive_channels"]) for r in fft_rows},
        key=lambda item: (item[1], item[0]),
    )
    for i, (filter_length, drive_channels) in enumerate(line_keys):
        group = sorted(
            [
                r
                for r in fft_rows
                if float(r["filter_length_s"]) == filter_length
                and r["drive_channels"] == drive_channels
            ],
            key=lambda r: int(r["K"]),
        )
        axes[0].plot(
            [int(r["K"]) for r in group],
            [float(r[metric_key]) for r in group],
            marker="o",
            lw=1.2,
            color=PALETTE[i % len(PALETTE)],
            label=f"L={filter_length:g}s, drives={drive_channels.replace(' ', '+')}",
        )
        refined_group = [
            r
            for r in refined_rows
            if float(r["filter_length_s"]) == filter_length and r["drive_channels"] == drive_channels
        ]
        if refined_group:
            best_refined = min(refined_group, key=lambda r: float(r[metric_key]))
            axes[0].scatter(
                1,
                float(best_refined[metric_key]),
                marker="D",
                s=34,
                color=PALETTE[i % len(PALETTE)],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
            )
    axes[0].scatter(
        selected_K,
        selected_metric,
        marker="*",
        s=140,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    axes[0].annotate(
        f"K={selected_K}, L={selected_filter_length:g}s\nNMSE={selected_metric:.4f}",
        xy=(selected_K, selected_metric),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=7,
        color="#111827",
        arrowprops={"arrowstyle": "->", "color": "#6B7280", "lw": 0.7},
    )
    axes[0].set_xlabel("number of selected frequencies K")
    axes[0].set_ylabel(args.selection_metric.replace("_", " "))
    axes[0].set_title("Mechanics-informed response holdout calibration")
    axes[0].legend(fontsize=6)
    axes[0].grid(axis="y", color="#E5E7EB", lw=0.6)
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
    panel_label(axes[0], "a")

    colors = [float(r["filter_length_s"]) for r in summary]
    marker_by_drives = {
        "u": "o",
    }
    for drive_channels in sorted({r["drive_channels"] for r in summary}, key=lambda s: (len(s.split()), s)):
        group = [r for r in summary if r["drive_channels"] == drive_channels]
        sc = axes[1].scatter(
            [float(r["condition_number_mean"]) for r in group],
            [float(r[metric_key]) for r in group],
            c=[float(r["filter_length_s"]) for r in group],
            cmap="viridis",
            s=36,
            marker=marker_by_drives.get(drive_channels, "o"),
            edgecolor="white",
            linewidth=0.5,
            label=drive_channels.replace(" ", "+"),
        )
    axes[1].scatter(
        selected_condition,
        selected_metric,
        marker="*",
        s=140,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    axes[1].annotate(
        f"K={selected_K}, L={selected_filter_length:g}s\nNMSE={selected_metric:.4f}",
        xy=(selected_condition, selected_metric),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=7,
        color="#111827",
        arrowprops={"arrowstyle": "->", "color": "#6B7280", "lw": 0.7},
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("mean condition number")
    axes[1].set_ylabel(args.selection_metric.replace("_", " "))
    axes[1].set_title("Conditioning check")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02, label="filter length (s)")
    axes[1].legend(fontsize=6, title="drives", title_fontsize=6)
    axes[1].grid(axis="y", color="#E5E7EB", lw=0.6)
    panel_label(axes[1], "b")

    t = trace["time"]
    axes[2].plot(t, trace["target"][:, 0], color="#8FBAD9", lw=1.2, label="x measured")
    axes[2].plot(t, trace["predicted"][:, 0], color="#1F4E79", lw=1.25, ls="--", label="x response fit")
    axes[2].plot(t, trace["target"][:, 1], color="#8DD3C7", lw=1.2, label="y measured")
    axes[2].plot(t, trace["predicted"][:, 1], color="#006D5B", lw=1.25, ls="--", label="y response fit")
    axes[2].set_xlabel("validation time (s)")
    axes[2].set_ylabel("relative hidden-node position")
    axes[2].set_title("Selected mechanical-response reconstruction on validation window")
    axes[2].grid(axis="y", color="#E5E7EB", lw=0.6)
    axes[2].legend(
        fontsize=7,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        handlelength=1.8,
    )
    axes[2].text(
        0.98,
        0.96,
        f"{trace['topology']}/{trace['amplitude']}/{trace['sample']}\n"
        f"NMSE_x = {trace['validation_nmse_x']:.3f}\n"
        f"NMSE_y = {trace['validation_nmse_y']:.3f}\n"
        f"NMSE_2D = {trace['validation_nmse_2d']:.3f}",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )
    panel_label(axes[2], "c")
    save_with_companions(fig, out_path)
    plt.close(fig)


def save_fft_plot(fft_info, selected_freqs, out_path):
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    ax.plot(fft_info["freq_target"], normalize_for_plot(fft_info["amp_target"]), lw=1.2, label="target FFT", color="#D55E00")
    drive_styles = {
        "u": {"ls": "-", "color": "#0072B2", "label": "u FFT"},
    }
    for drive_name, style in drive_styles.items():
        if drive_name not in fft_info["drive_fft"]:
            continue
        drive_fft = fft_info["drive_fft"][drive_name]
        ax.plot(
            drive_fft["freq"],
            normalize_for_plot(drive_fft["amp"]),
            lw=1.0,
            ls=style["ls"],
            color=style["color"],
            label=style["label"],
        )
    for freq in selected_freqs:
        ax.axvline(freq, color="#4B5563", ls="--", lw=0.8)
        ax.text(freq, 1.02, f"{freq:.2f} Hz", rotation=90, ha="center", va="bottom", fontsize=6)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalized Hann FFT amplitude")
    ax.set_title("Mechanical response frequency selection (train window only)")
    ax.legend(fontsize=7)
    if fft_info["coherence_available"] and fft_info["coherence_by_drive"]:
        all_freqs = None
        all_coh = []
        for drive_name, coh_info in fft_info["coherence_by_drive"].items():
            if all_freqs is None:
                all_freqs = coh_info["freq"]
            all_coh.append(np.interp(all_freqs, coh_info["freq"], coh_info["coherence"]))
        best_coherence = np.max(np.vstack(all_coh), axis=0)
        ax2 = ax.twinx()
        ax2.plot(all_freqs, best_coherence, color="#111827", lw=0.9, alpha=0.65)
        ax2.set_ylabel("best coherence")
        ax2.set_ylim(0, 1)
    ax.text(
        0.98,
        0.96,
        "Hann window\ntrain window only\ndisplacement drive",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )
    save_with_companions(fig, out_path)
    plt.close(fig)


def comparison_template(
    records,
    reliability,
    args,
    out_dir,
    excluded_samples,
    selected_freqs_by_record,
    selected,
):
    legendre_dir = legendre_output_dir(args, records, excluded_samples)
    legendre_json = legendre_dir / "selected_dictionary.json"
    legendre = {}
    if legendre_json.exists():
        legendre = json.loads(legendre_json.read_text())
    rows = []
    rel_by_key = {(r["topology"], r["sample"]): r for r in reliability}
    for record in records:
        rel = rel_by_key[(record["topology"], record["sample"])]
        rep = legendre.get("representative_trace", {})
        rows.append(
            {
                "topology": record["topology"],
                "sample": record["sample"],
                "mechanical_validation_nmse_x": rel["basis_nmse_x"],
                "mechanical_validation_nmse_y": rel["basis_nmse_y"],
                "mechanical_validation_nmse_2d": rel["basis_nmse_2d"],
                "selected_drive_channels": selected["drive_channels"],
                "drive_smoothing_frames": args.drive_smoothing_frames,
                "selected_K": rel["selected_K"],
                "selected_frequencies_hz": " ".join(
                    f"{f:.8g}"
                    for f in selected_frequency_set_for_record(
                        record, selected, selected_freqs_by_record
                    )["frequencies"]
                ),
                "selected_filter_length_s": rel["selected_filter_length_s"],
                "selected_mechanical_degree": rel["selected_mechanical_degree"],
                "number_linear_features": rel["number_linear_features"],
                "number_candidate_features": rel["number_candidate_features"],
                "number_kept_features": rel["number_kept_features"],
                "selected_svd_rank": rel["selected_svd_rank"],
                "legendre_validation_nmse_x": rep.get("validation_nmse_x", ""),
                "legendre_validation_nmse_y": rep.get("validation_nmse_y", ""),
                "legendre_validation_nmse_2d": rep.get("validation_nmse_2d", ""),
                "legendre_H_cut": legendre.get("H_cut", ""),
                "legendre_D_cut": legendre.get("D_cut", ""),
            }
        )
    write_csv(rows, out_dir / "mechanical_vs_legendre_comparison_template.csv")


def main():
    configure_matplotlib()
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    args = parse_args()
    degree = effective_mechanical_degree(args)
    if args.max_nonlinear_features < 1:
        raise ValueError("--max-nonlinear-features must be >= 1.")
    records, skipped, excluded_samples = discover_records(args)
    out_dir = mechanical_output_dir(args, records, excluded_samples)
    print("Mechanical-response basis calibration:")
    print("  frequency selection uses train window only")
    print(f"  mechanical response product degree: {degree}")
    print(f"  scipy coherence used: {SCIPY_SIGNAL_AVAILABLE}")
    print("  selected topology/sample records:")
    for record in records:
        print(f"    {record['topology']}/{record['amplitude']}/{record['sample']}")

    frequency_rows = []
    candidates = []
    selected_freqs_by_record = {}
    fft_info_by_record = {}
    cache = {}
    for record in records:
        loader = StateLoader(record["sample_dir"] / "experiment.h5")
        _, positions = load_positions(record["sample_dir"] / "experiment.h5")
        u = loader.get_actuation_signal(actuator_idx=0, dof=0)
        train_start, train_stop, _ = frame_windows(loader, args)
        train_rows = np.arange(train_start, train_stop)
        drives = {"u": np.asarray(u, dtype=float)}
        drives_z, _ = standardize_drive_channels_from_train(
            drives, train_rows, ("u",)
        )
        hidden_relative = positions[:, args.hidden_node, :] - positions[:, args.reference_node, :]
        selected_freqs, freq_rows, fft_info = select_frequencies(
            record, args, loader, positions, hidden_relative, train_rows, drives_z
        )
        if not selected_freqs:
            print(f"Warning: no selected frequencies for {record['topology']}/{record['sample']}.")
            continue
        print(
            f"-> {record['topology']}/{record['sample']} selected frequencies: "
            + ", ".join(f"{f:.3f} Hz" for f in selected_freqs)
        )
        selected_freqs_by_record[(record["topology"], record["sample"])] = selected_freqs
        fft_info_by_record[(record["topology"], record["sample"])] = fft_info
        frequency_rows.extend(freq_rows)
        frequency_sets = candidate_frequency_sets(selected_freqs, args)
        if args.frequency_refinement:
            print(
                f"   refinement grid: {args.refine_target_hz - args.refine_half_width_hz:.3f}--"
                f"{args.refine_target_hz + args.refine_half_width_hz:.3f} Hz "
                f"step {args.refine_step_hz:g} Hz"
            )
        for frequency_set in frequency_sets:
            for drive_set in drive_channel_sets(args):
                for filter_length_s in args.filter_lengths_s:
                    try:
                        row, row_cache = evaluate_candidate(
                            record, args, frequency_set, filter_length_s, drive_set
                        )
                    except ValueError as exc:
                        if "Nonlinear product expansion" not in str(exc):
                            raise
                        print(
                            f"Warning: skipping {record['topology']}/{record['sample']} "
                            f"{frequency_set['frequency_set_label']} L={filter_length_s:g}s: {exc}"
                        )
                        continue
                    candidates.append(row)
                    cache[
                        (
                            record["topology"],
                            record["sample"],
                            frequency_set["frequency_set_label"],
                            float(filter_length_s),
                            " ".join(drive_set),
                        )
                    ] = row_cache

    if not candidates:
        raise RuntimeError("No valid mechanical-response candidates were evaluated.")
    summary = summarize_candidates(candidates)
    topology_summary = summarize_by_topology(candidates)
    selected, best, tolerance = choose_candidate(summary, args)
    trace_record = records[0]
    trace_frequency_set = selected_frequency_set_for_record(
        trace_record, selected, selected_freqs_by_record
    )
    trace_freqs = trace_frequency_set["frequencies"]
    trace = selected_validation_trace(trace_record, args, selected, selected_freqs_by_record)
    weights, reliability, loadings = final_weights(records, args, selected_freqs_by_record, selected)
    trace_rows = [
        {
            "topology": trace["topology"],
            "amplitude": trace["amplitude"],
            "sample": trace["sample"],
            "time_s": float(t),
            "target_x": float(target[0]),
            "target_y": float(target[1]),
            "mechanical_fit_x": float(pred[0]),
            "mechanical_fit_y": float(pred[1]),
        }
        for t, target, pred in zip(trace["time"], trace["target"], trace["predicted"])
    ]

    selected_reliability = [r for r in reliability]
    mean_nmse_x = float(np.mean([r["basis_nmse_x"] for r in selected_reliability]))
    mean_nmse_y = float(np.mean([r["basis_nmse_y"] for r in selected_reliability]))
    mean_nmse_2d = float(np.mean([r["basis_nmse_2d"] for r in selected_reliability]))
    payload = {
        "script": "run_hidden_node_mechanical_response_basis.py",
        "selection_protocol": "mechanics-informed response basis; FFT on train window only; holdout validation",
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
        "target_component": args.target_component,
        "frequency_source": args.frequency_source,
        "hann_window_used": True,
        "coherence_available": bool(SCIPY_SIGNAL_AVAILABLE),
        "frequency_refinement_enabled": bool(args.frequency_refinement),
        "refine_target_hz": float(args.refine_target_hz),
        "refine_half_width_hz": float(args.refine_half_width_hz),
        "refine_step_hz": float(args.refine_step_hz),
        "drive_channels_available": ["u"],
        "selected_drive_channels": ["u"],
        "drive_smoothing_frames": None,
        "candidate_sweep_includes_drive_sets": False,
        "selected_frequency_set_type": selected["frequency_set_type"],
        "selected_frequency_set_label": selected["frequency_set_label"],
        "selected_refined_frequency_hz": float(selected["refined_frequency_hz"])
        if str(selected["refined_frequency_hz"]) != "nan"
        else None,
        "selected_K": int(selected["K"]),
        "selected_frequencies_hz": {
            f"{record['topology']}/{record['sample']}": selected_frequency_set_for_record(
                record, selected, selected_freqs_by_record
            )["frequencies"]
            for record in records
        },
        "selected_filter_length_s": float(selected["filter_length_s"]),
        "selected_filter_length_frames": int(selected["filter_length_frames"]),
        "decay_times_s": [float(v) for v in args.decay_times_s],
        "mechanical_degree": int(selected["mechanical_degree"]),
        "include_nonlinear_products": int(selected["mechanical_degree"]) > 1,
        "number_drive_channels": int(selected["number_drive_channels"]),
        "number_linear_features": float(selected["number_linear_features_mean"]),
        "number_candidate_features": float(selected["number_candidate_features_mean"]),
        "number_raw_features": float(selected["number_raw_features_mean"]),
        "selected_svd_rank": float(selected["svd_rank_mean"]),
        "condition_number": float(selected["condition_number_mean"]),
        "selection_metric": args.selection_metric,
        "selected_validation_nmse_x": float(selected["validation_nmse_x_mean"]),
        "selected_validation_nmse_y": float(selected["validation_nmse_y_mean"]),
        "selected_validation_nmse_2d": float(selected["validation_nmse_2d_mean"]),
        "basis_adequacy_label_x": basis_label(mean_nmse_x),
        "basis_adequacy_label_y": basis_label(mean_nmse_y),
        "basis_adequacy_label_2d": basis_label(mean_nmse_2d),
        "skipped": skipped,
        "excluded_samples": {topology: sorted(samples) for topology, samples in excluded_samples.items()},
    }

    write_csv(frequency_rows, out_dir / "frequency_candidates.csv")
    write_csv(candidates, out_dir / "mechanical_response_candidates_per_sample.csv")
    write_csv(summary, out_dir / "mechanical_response_summary.csv")
    write_csv(topology_summary, out_dir / "mechanical_response_by_topology.csv")
    write_csv(loadings, out_dir / "mechanical_response_feature_loadings.csv")
    write_csv(weights, out_dir / "mechanical_response_task_weights.csv")
    write_csv(reliability, out_dir / "selected_mechanical_response_reliability.csv")
    write_csv(trace_rows, out_dir / "selected_mechanical_response_validation_trace.csv")
    write_json(payload, out_dir / "selected_mechanical_response_dictionary.json")
    comparison_template(
        records,
        reliability,
        args,
        out_dir,
        excluded_samples,
        selected_freqs_by_record,
        selected,
    )

    trace_fft = fft_info_by_record[(trace_record["topology"], trace_record["sample"])]
    save_summary_plot(
        summary,
        selected,
        trace,
        trace_fft,
        trace_freqs[: int(selected["K"])],
        args,
        out_dir / "mechanical_response_basis_summary.pdf",
    )
    save_fft_plot(
        trace_fft,
        trace_freqs[: int(selected["K"])],
        out_dir / "mechanical_response_fft_selection.pdf",
    )

    print("Candidate table summary:")
    printable_rows = [r for r in summary if r["frequency_set_type"] == "fft_prefix"]
    refined_print_rows = sorted(
        [r for r in summary if r["frequency_set_type"] == "refined_single"],
        key=lambda r: float(r[args.selection_metric + "_mean"]),
    )[:10]
    if refined_print_rows:
        print("  coarse FFT-prefix candidates:")
    for row in printable_rows:
        print(
            f"  K={int(row['K'])}, L={float(row['filter_length_s']):g}s, "
            f"drives={row['drive_channels'].replace(' ', '+')}, "
            f"degree={int(row['mechanical_degree'])}, "
            f"terms={float(row['number_raw_features_mean']):.0f}, "
            f"{args.selection_metric}={float(row[args.selection_metric + '_mean']):.4f}, "
            f"rank={float(row['svd_rank_mean']):.1f}"
        )
    if refined_print_rows:
        print("  best dense-refinement candidates:")
    for row in refined_print_rows:
        print(
            f"  f={float(row['refined_frequency_hz']):.3f}Hz, "
            f"L={float(row['filter_length_s']):g}s, "
            f"drives={row['drive_channels'].replace(' ', '+')}, "
            f"degree={int(row['mechanical_degree'])}, "
            f"terms={float(row['number_raw_features_mean']):.0f}, "
            f"{args.selection_metric}={float(row[args.selection_metric + '_mean']):.4f}, "
            f"rank={float(row['svd_rank_mean']):.1f}"
        )
    print(
        f"Selected mechanical response basis: {selected['frequency_set_type']}, "
        f"drives={selected['drive_channels'].replace(' ', '+')}, "
        f"degree={int(selected['mechanical_degree'])}, "
        f"frequencies={selected['selected_frequencies_hz']} Hz, K={int(selected['K'])}, "
        f"filter length={float(selected['filter_length_s']):g}s, "
        f"terms~{float(selected['number_raw_features_mean']):.0f}, "
        f"rank~{float(selected['svd_rank_mean']):.1f}."
    )
    print(
        f"Validation NMSE x/y/2d: {float(selected['validation_nmse_x_mean']):.4f}, "
        f"{float(selected['validation_nmse_y_mean']):.4f}, "
        f"{float(selected['validation_nmse_2d_mean']):.4f}."
    )
    if float(selected["validation_nmse_y_mean"]) > 0.7:
        print(
            "Mechanical response basis still fails for y. This suggests hidden y is not "
            "compactly represented by scalar input-driven response features."
        )
    if float(selected["condition_number_mean"]) > args.condition_threshold:
        print("Warning: selected mechanical response basis is ill-conditioned.")
    if float(selected["svd_rank_mean"]) < 0.5 * float(selected["number_raw_features_mean"]):
        print(
            "Mechanical response features are highly redundant; interpretation should use "
            "orthogonal SVD modes."
        )
    print(
        "Interpretation: if mechanical_response_basis NMSE_y is much lower than Legendre "
        "NMSE_y, the hidden y-coordinate is input-driven but raw delayed Legendre "
        "coordinates were a poor basis. If NMSE_y remains high while actual visible-state "
        "readout NMSE_y is low, prediction is likely dominated by spatial observability "
        "from visible reservoir states."
    )
    print("Do not call this standard IPC; use mechanics-informed response basis adequacy.")
    print(f"Saved output directory: {out_dir}")


if __name__ == "__main__":
    main()
