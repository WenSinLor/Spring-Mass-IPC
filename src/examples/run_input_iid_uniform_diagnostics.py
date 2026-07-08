import argparse
import csv
import os
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


DEFAULT_TOPOLOGY = "topology_17_prestress"
DEFAULT_AMPLITUDE = "amp=1"
DEFAULT_SOURCE = "actuation"
DEFAULT_MAX_LAG = 60
DEFAULT_BINS = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the input trajectory is approximately IID "
            "Uniform(-1, 1) after normalization."
        )
    )
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    parser.add_argument("--amplitude", default=DEFAULT_AMPLITUDE)
    parser.add_argument(
        "--source",
        choices=("actuation", "node0_x"),
        default=DEFAULT_SOURCE,
        help=(
            "'actuation' reads time_series/actuation_signals/0. "
            "'node0_x' reconstructs -(node0_x - node0_x[0]) from positions."
        ),
    )
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    return parser.parse_args()


def normalize_to_unit_interval(u):
    u = np.asarray(u, dtype=float).reshape(-1)
    finite = np.isfinite(u)
    if not np.all(finite):
        u = u[finite]

    u_min = float(np.min(u))
    u_max = float(np.max(u))
    if abs(u_max - u_min) < 1e-12:
        raise ValueError("Input is constant; cannot normalize to [-1, 1].")

    u_norm = 2.0 * (u - u_min) / (u_max - u_min) - 1.0
    return u, u_norm, u_min, u_max


def load_input_from_h5(h5_path: Path, source: str):
    with h5py.File(h5_path, "r") as f:
        time = f["time_series/time"][:]

        if source == "actuation":
            u = f["time_series/actuation_signals/0"][:]
            if u.ndim == 2:
                u = u[:, 0]
        elif source == "node0_x":
            pos = f["time_series/nodes/positions"][:, :, :]
            node0_x = pos[:, 0, 0]
            u = -(node0_x - node0_x[0])
        else:
            raise ValueError(f"Unknown source: {source}")

    min_len = min(len(time), len(u))
    return time[:min_len], np.asarray(u[:min_len], dtype=float)


def autocorrelation(x, max_lag):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.full(max_lag + 1, np.nan)

    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = float(np.dot(x[:-lag], x[lag:]) / denom)
    return acf


def runs_test_around_median(x):
    x = np.asarray(x, dtype=float)
    median = float(np.median(x))
    signs = x >= median

    n1 = int(np.sum(signs))
    n2 = int(len(signs) - n1)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, 0

    runs = int(1 + np.sum(signs[1:] != signs[:-1]))
    expected = 1.0 + (2.0 * n1 * n2) / (n1 + n2)
    variance = (
        2.0
        * n1
        * n2
        * (2.0 * n1 * n2 - n1 - n2)
        / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    )
    z = (runs - expected) / np.sqrt(variance)
    p = 2.0 * stats.norm.sf(abs(z))
    return z, p, runs


def analyze_sequence(u_norm, max_lag, bins):
    n = len(u_norm)
    uniform = stats.uniform(loc=-1.0, scale=2.0)

    ks_stat, ks_p = stats.kstest(u_norm, uniform.cdf)

    counts, edges = np.histogram(u_norm, bins=bins, range=(-1.0, 1.0))
    expected = np.full(bins, n / bins, dtype=float)
    chi2_stat, chi2_p = stats.chisquare(counts, expected)

    acf = autocorrelation(u_norm, max_lag=max_lag)
    conf = 1.96 / np.sqrt(n)
    significant_lags = int(np.sum(np.abs(acf[1:]) > conf))
    max_abs_acf = float(np.nanmax(np.abs(acf[1:]))) if max_lag > 0 else np.nan

    runs_z, runs_p, runs = runs_test_around_median(u_norm)

    return {
        "n": n,
        "mean": float(np.mean(u_norm)),
        "variance": float(np.var(u_norm)),
        "skew": float(stats.skew(u_norm)),
        "excess_kurtosis": float(stats.kurtosis(u_norm)),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "chi2_stat": float(chi2_stat),
        "chi2_p": float(chi2_p),
        "runs": runs,
        "runs_z": float(runs_z),
        "runs_p": float(runs_p),
        "acf_conf95": float(conf),
        "max_abs_acf_lag1_to_max": max_abs_acf,
        "significant_acf_lags": significant_lags,
        "acf": acf,
        "hist_counts": counts,
        "hist_edges": edges,
    }


def verdict(metrics):
    distribution_ok = metrics["ks_p"] >= 0.05 and metrics["chi2_p"] >= 0.05
    independence_ok = metrics["runs_p"] >= 0.05 and metrics["significant_acf_lags"] <= 3

    if distribution_ok and independence_ok:
        return "approx_iid_uniform"
    if distribution_ok:
        return "uniform_like_but_correlated"
    if independence_ok:
        return "independent_like_but_not_uniform"
    return "not_iid_uniform"


def save_summary_csv(rows, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "source",
        "n",
        "raw_min",
        "raw_max",
        "mean",
        "variance",
        "skew",
        "excess_kurtosis",
        "ks_p",
        "chi2_p",
        "runs_p",
        "max_abs_acf_lag1_to_max",
        "significant_acf_lags",
        "verdict",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def plot_diagnostics(sample_results, out_path, max_lag):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(sample_results)
    fig, axes = plt.subplots(
        n_samples,
        3,
        figsize=(15, max(3.0 * n_samples, 6)),
        constrained_layout=False,
    )
    axes = np.atleast_2d(axes)

    for row_idx, result in enumerate(sample_results):
        sample = result["sample"]
        time = result["time"]
        u_norm = result["u_norm"]
        metrics = result["metrics"]
        color = "#D55E00" if sample == "sample_1" else "#0072B2"

        ax_time, ax_hist, ax_acf = axes[row_idx]

        ax_time.plot(time, u_norm, color=color, linewidth=1.0)
        ax_time.set_ylim(-1.08, 1.08)
        ax_time.set_title(f"{sample}: normalized input")
        ax_time.set_ylabel("u norm")
        ax_time.grid(alpha=0.25)

        ax_hist.hist(
            u_norm,
            bins=metrics["hist_edges"],
            density=True,
            color=color,
            alpha=0.65,
            edgecolor="white",
        )
        ax_hist.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
        ax_hist.set_xlim(-1.05, 1.05)
        ax_hist.set_title(
            f"distribution: KS p={metrics['ks_p']:.3g}, chi2 p={metrics['chi2_p']:.3g}"
        )
        ax_hist.set_ylabel("density")
        ax_hist.grid(alpha=0.2)

        lags = np.arange(max_lag + 1)
        ax_acf.axhspan(
            -metrics["acf_conf95"],
            metrics["acf_conf95"],
            color="#BBBBBB",
            alpha=0.35,
            label="95% iid band",
        )
        ax_acf.vlines(lags[1:], 0, metrics["acf"][1:], color=color, linewidth=1.0)
        ax_acf.axhline(0, color="black", linewidth=0.8)
        ax_acf.set_ylim(-1.0, 1.0)
        ax_acf.set_title(
            f"autocorrelation: runs p={metrics['runs_p']:.3g}, "
            f"sig lags={metrics['significant_acf_lags']}"
        )
        ax_acf.set_ylabel("ACF")
        ax_acf.grid(alpha=0.2)

        if row_idx == n_samples - 1:
            ax_time.set_xlabel("time in slice (s)")
            ax_hist.set_xlabel("u normalized")
            ax_acf.set_xlabel("lag")

    fig.suptitle(
        "IID Uniform(-1, 1) diagnostics for the IPC input\n"
        "Uniformity: histogram/KS/chi-square; independence: autocorrelation/runs test",
        fontsize=13,
        y=0.99,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.91, bottom=0.06, hspace=0.55, wspace=0.24)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    current_script_dir = Path(__file__).parent.resolve()
    amp_dir = (
        current_script_dir.parent.parent
        / "data"
        / "experiment_data"
        / args.topology
        / args.amplitude
    )

    sample_dirs = sorted(
        p for p in amp_dir.glob("sample_*") if (p / "experiment.h5").exists()
    )
    if not sample_dirs:
        print(f"[Error] No sample_*/experiment.h5 files found in {amp_dir}")
        return

    sample_results = []
    summary_rows = []
    for sample_dir in sample_dirs:
        time, u_raw = load_input_from_h5(sample_dir / "experiment.h5", args.source)
        u_raw, u_norm, raw_min, raw_max = normalize_to_unit_interval(u_raw)
        metrics = analyze_sequence(u_norm, max_lag=args.max_lag, bins=args.bins)
        row = {
            "sample": sample_dir.name,
            "source": args.source,
            "raw_min": raw_min,
            "raw_max": raw_max,
            **{k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)},
        }
        row["verdict"] = verdict(metrics)
        summary_rows.append(row)
        sample_results.append(
            {
                "sample": sample_dir.name,
                "time": time[: len(u_norm)],
                "u_norm": u_norm,
                "metrics": metrics,
            }
        )

    output_dir = amp_dir / "plots"
    csv_path = output_dir / f"input_iid_uniform_summary_{args.source}.csv"
    svg_path = output_dir / f"input_iid_uniform_diagnostics_{args.source}.svg"
    save_summary_csv(summary_rows, csv_path)
    plot_diagnostics(sample_results, svg_path, max_lag=args.max_lag)

    print(f"[Saved] CSV -> {csv_path}")
    print(f"[Saved] SVG -> {svg_path}")
    print("\nIID Uniform(-1, 1) diagnostic summary:")
    print("Expected for ideal Uniform(-1,1): mean=0, variance=0.333, skew=0, excess kurtosis=-1.2")
    for row in summary_rows:
        print(
            f"  {row['sample']}: verdict={row['verdict']}, "
            f"KS p={row['ks_p']:.3g}, chi2 p={row['chi2_p']:.3g}, "
            f"runs p={row['runs_p']:.3g}, "
            f"sig ACF lags={row['significant_acf_lags']}, "
            f"max |ACF|={row['max_abs_acf_lag1_to_max']:.3f}"
        )


if __name__ == "__main__":
    main()
