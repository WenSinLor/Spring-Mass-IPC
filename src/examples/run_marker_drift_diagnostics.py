import argparse
import csv
import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


DEFAULT_TOPOLOGY = "topology_15_prestress"
DEFAULT_AMPLITUDE = "amp=2.5"
DEFAULT_NODES = [3, 12, 15]
DEFAULT_WASHOUT_S = 5.0
DEFAULT_TRAIN_S = 10.0
DEFAULT_TEST_S = 10.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Quantify and visualize train/test marker-coordinate drift in "
            "experiment.h5 samples."
        )
    )
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    parser.add_argument("--amplitude", default=DEFAULT_AMPLITUDE)
    parser.add_argument(
        "--nodes",
        nargs="+",
        type=int,
        default=DEFAULT_NODES,
        help="Marker/node IDs to emphasize in the plot.",
    )
    parser.add_argument("--washout", type=float, default=DEFAULT_WASHOUT_S)
    parser.add_argument("--train", type=float, default=DEFAULT_TRAIN_S)
    parser.add_argument("--test", type=float, default=DEFAULT_TEST_S)
    return parser.parse_args()


def load_sample(sample_dir: Path):
    h5_path = sample_dir / "experiment.h5"
    with h5py.File(h5_path, "r") as f:
        time = f["time_series/time"][:]
        positions = f["time_series/nodes/positions"][:, :, :2]
        fps = float(f.attrs.get("fps", 1.0 / np.median(np.diff(time))))

    return {
        "sample": sample_dir.name,
        "path": h5_path,
        "time": time,
        "positions": positions,
        "fps": fps,
    }


def frame_windows(time, fps, washout_s, train_s, test_s):
    if len(time) < 2:
        raise ValueError("Time series is too short.")

    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0 / fps

    washout_frames = int(washout_s / dt)
    train_frames = int(train_s / dt)
    test_frames = int(test_s / dt)

    train_start = washout_frames
    train_stop = washout_frames + train_frames
    test_stop = train_stop + test_frames

    if test_stop > len(time):
        raise ValueError(
            f"Need {test_stop} frames for configured windows, "
            f"but sample has {len(time)}."
        )

    return train_start, train_stop, test_stop


def outside_train_range(values, train_values):
    train_min = float(np.min(train_values))
    train_max = float(np.max(train_values))

    below = np.maximum(train_min - values, 0.0)
    above = np.maximum(values - train_max, 0.0)
    unsigned = np.maximum(below, above)

    signed = np.zeros_like(values, dtype=float)
    signed[values < train_min] = values[values < train_min] - train_min
    signed[values > train_max] = values[values > train_max] - train_max
    return signed, unsigned, train_min, train_max


def compute_records(samples, nodes, washout_s, train_s, test_s):
    records = []
    summary = []

    for sample in samples:
        time = sample["time"]
        pos = sample["positions"]
        train_start, train_stop, test_stop = frame_windows(
            time, sample["fps"], washout_s, train_s, test_s
        )

        per_sample_records = []
        for node in range(pos.shape[1]):
            for dim_idx, dim_name in enumerate(("x", "y")):
                series = pos[:, node, dim_idx]
                train_values = series[train_start:train_stop]
                test_values = series[train_stop:test_stop]

                signed, unsigned, train_min, train_max = outside_train_range(
                    test_values, train_values
                )

                final_signed = float(signed[-1])
                max_idx = int(np.argmax(unsigned))
                rec = {
                    "sample": sample["sample"],
                    "node": node,
                    "dim": dim_name,
                    "train_min_px": train_min,
                    "train_max_px": train_max,
                    "train_mean_px": float(np.mean(train_values)),
                    "test_min_px": float(np.min(test_values)),
                    "test_max_px": float(np.max(test_values)),
                    "final_test_px": float(test_values[-1]),
                    "final_outside_train_px": final_signed,
                    "max_test_outside_train_px": float(unsigned[max_idx]),
                    "max_test_outside_time_s": float(time[train_stop + max_idx]),
                    "emphasized_node": node in nodes,
                }
                records.append(rec)
                per_sample_records.append(rec)

        emphasized = [r for r in per_sample_records if r["node"] in nodes]
        all_max = max(
            per_sample_records,
            key=lambda r: abs(r["max_test_outside_train_px"]),
        )
        emphasized_max = max(
            emphasized,
            key=lambda r: abs(r["max_test_outside_train_px"]),
        )
        emphasized_final = max(
            emphasized,
            key=lambda r: abs(r["final_outside_train_px"]),
        )

        summary.append(
            {
                "sample": sample["sample"],
                "max_all": all_max,
                "max_emphasized": emphasized_max,
                "final_emphasized": emphasized_final,
            }
        )

    return records, summary


def save_csv(records, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "node",
        "dim",
        "train_min_px",
        "train_max_px",
        "train_mean_px",
        "test_min_px",
        "test_max_px",
        "final_test_px",
        "final_outside_train_px",
        "max_test_outside_train_px",
        "max_test_outside_time_s",
        "emphasized_node",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in fieldnames})


def sample_colors(samples):
    names = [s["sample"] for s in samples]
    nature_palette = {
        "sample_0": "#0072B2",  # blue
        "sample_1": "#D55E00",  # vermillion highlight
        "sample_2": "#009E73",  # bluish green
        "sample_3": "#CC79A7",  # reddish purple
        "sample_4": "#E69F00",  # orange/gold
    }
    fallback = ["#56B4E9", "#999999", "#332288", "#88CCEE", "#44AA99"]
    return {
        name: nature_palette.get(name, fallback[i % len(fallback)])
        for i, name in enumerate(names)
    }


def plot_all_node_timeseries(
    samples,
    out_path,
    washout_s,
    train_s,
    test_s,
    center_on_train_mean=True,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = sample_colors(samples)
    n_nodes = samples[0]["positions"].shape[1]
    n_cols = 4
    n_rows_per_dim = int(np.ceil(n_nodes / n_cols))

    fig, axes = plt.subplots(
        n_rows_per_dim * 2,
        n_cols,
        figsize=(18, 21),
        sharex=True,
        constrained_layout=False,
    )
    axes = np.asarray(axes)

    for dim_idx, dim_name in enumerate(("x", "y")):
        for node in range(n_nodes):
            row = dim_idx * n_rows_per_dim + node // n_cols
            col = node % n_cols
            ax = axes[row, col]

            panel_values = []
            sample_1_train_band = None

            for sample in samples:
                time = sample["time"]
                pos = sample["positions"]
                train_start, train_stop, test_stop = frame_windows(
                    time, sample["fps"], washout_s, train_s, test_s
                )
                series = pos[:, node, dim_idx]
                train_series = series[train_start:train_stop]
                train_mean = float(np.mean(train_series))

                if center_on_train_mean:
                    y = series - train_mean
                    if sample["sample"] == "sample_1":
                        sample_1_train_band = (
                            float(np.min(train_series) - train_mean),
                            float(np.max(train_series) - train_mean),
                        )
                else:
                    y = series
                    if sample["sample"] == "sample_1":
                        sample_1_train_band = (
                            float(np.min(train_series)),
                            float(np.max(train_series)),
                        )

                panel_values.append(y)
                lw = 1.9 if sample["sample"] == "sample_1" else 0.9
                alpha = 0.95 if sample["sample"] == "sample_1" else 0.58
                zorder = 4 if sample["sample"] == "sample_1" else 2
                ax.plot(
                    time,
                    y,
                    color=colors[sample["sample"]],
                    linewidth=lw,
                    alpha=alpha,
                    zorder=zorder,
                    label=sample["sample"],
                )

            if sample_1_train_band is not None:
                ax.axhspan(
                    sample_1_train_band[0],
                    sample_1_train_band[1],
                    color="#D62728",
                    alpha=0.07,
                    zorder=1,
                )

            ax.axvspan(washout_s, washout_s + train_s, color="#59A14F", alpha=0.06)
            ax.axvspan(
                washout_s + train_s,
                washout_s + train_s + test_s,
                color="#F28E2B",
                alpha=0.06,
            )
            if center_on_train_mean:
                ax.axhline(0, color="black", linewidth=0.6, alpha=0.55)

            if panel_values:
                all_values = np.concatenate(panel_values)
                lo, hi = np.nanpercentile(all_values, [1, 99])
                pad = max((hi - lo) * 0.15, 0.5)
                ax.set_ylim(lo - pad, hi + pad)

            ax.set_title(f"node {node} {dim_name}", fontsize=9)
            ax.grid(alpha=0.22, linewidth=0.6)
            ax.tick_params(labelsize=7)

            if col == 0:
                ylabel = "coord - train mean (px)" if center_on_train_mean else "raw coord (px)"
                ax.set_ylabel(ylabel, fontsize=8)
            if row in (n_rows_per_dim - 1, 2 * n_rows_per_dim - 1):
                ax.set_xlabel("time in slice (s)", fontsize=8)

    for idx in range(n_nodes, n_rows_per_dim * n_cols):
        for dim_idx in range(2):
            row = dim_idx * n_rows_per_dim + idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
    )

    mode = "centered on each sample's training mean" if center_on_train_mean else "raw pixel coordinates"
    fig.suptitle(
        "All-node marker time series across samples\n"
        f"{mode}; green=train window, orange=test window; sample_1 highlighted red",
        fontsize=13,
        y=0.985,
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.925, bottom=0.055, hspace=0.55, wspace=0.22)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(samples, records, summary, nodes, out_path, washout_s, train_s, test_s):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = sample_colors(samples)

    fig, axes = plt.subplots(
        len(nodes),
        2,
        figsize=(14, max(7, 2.7 * len(nodes))),
        sharex=True,
        constrained_layout=False,
    )
    axes = np.atleast_2d(axes)

    def plot_node_dim(ax, node, dim_idx):
        dim_name = "x" if dim_idx == 0 else "y"
        for sample in samples:
            time = sample["time"]
            pos = sample["positions"]
            train_start, train_stop, test_stop = frame_windows(
                time, sample["fps"], washout_s, train_s, test_s
            )
            series = pos[:, node, dim_idx]
            train_series = series[train_start:train_stop]
            train_mean = float(np.mean(train_series))
            y = series - train_mean
            train_min = float(np.min(train_series) - train_mean)
            train_max = float(np.max(train_series) - train_mean)

            lw = 2.2 if sample["sample"] == "sample_1" else 1.1
            alpha = 0.95 if sample["sample"] == "sample_1" else 0.55
            ax.plot(
                time,
                y,
                color=colors[sample["sample"]],
                linewidth=lw,
                alpha=alpha,
                label=sample["sample"],
            )
            if sample["sample"] == "sample_1":
                ax.axhspan(train_min, train_max, color="tab:red", alpha=0.08)

        ax.axvspan(washout_s, washout_s + train_s, color="tab:green", alpha=0.08)
        ax.axvspan(
            washout_s + train_s,
            washout_s + train_s + test_s,
            color="tab:orange",
            alpha=0.08,
        )
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.6)
        ax.set_title(f"Node {node} {dim_name} drift")
        ax.set_ylabel(f"node {node} {dim_name} - train mean (px)")
        ax.grid(alpha=0.3)

    for row, node in enumerate(nodes):
        plot_node_dim(axes[row, 0], node=node, dim_idx=0)
        plot_node_dim(axes[row, 1], node=node, dim_idx=1)
        axes[row, 0].set_title(f"Node {node} x drift")
        axes[row, 1].set_title(f"Node {node} y drift")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time in slice (s)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )

    fig.suptitle(
        "Focused drift view for support/corner markers",
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.105, hspace=0.45, wspace=0.18)
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

    samples = [load_sample(p) for p in sample_dirs]
    records, summary = compute_records(
        samples,
        nodes=args.nodes,
        washout_s=args.washout,
        train_s=args.train,
        test_s=args.test,
    )

    output_dir = amp_dir / "plots"
    csv_path = output_dir / "marker_drift_summary.csv"
    focused_svg_path = output_dir / "marker_drift_focused_nodes.svg"
    centered_svg_path = output_dir / "marker_timeseries_all_nodes_centered.svg"
    raw_svg_path = output_dir / "marker_timeseries_all_nodes_raw.svg"

    save_csv(records, csv_path)
    plot_all_node_timeseries(
        samples,
        out_path=centered_svg_path,
        washout_s=args.washout,
        train_s=args.train,
        test_s=args.test,
        center_on_train_mean=True,
    )
    plot_all_node_timeseries(
        samples,
        out_path=raw_svg_path,
        washout_s=args.washout,
        train_s=args.train,
        test_s=args.test,
        center_on_train_mean=False,
    )
    plot_diagnostics(
        samples,
        records,
        summary,
        nodes=args.nodes,
        out_path=focused_svg_path,
        washout_s=args.washout,
        train_s=args.train,
        test_s=args.test,
    )

    print(f"[Saved] CSV -> {csv_path}")
    print(f"[Saved] Centered all-node SVG -> {centered_svg_path}")
    print(f"[Saved] Raw all-node SVG -> {raw_svg_path}")
    print(f"[Saved] Focused SVG -> {focused_svg_path}")
    print("\nLargest emphasized-node drift per sample:")
    for item in summary:
        rec = item["max_emphasized"]
        final_rec = item["final_emphasized"]
        print(
            f"  {item['sample']}: node {rec['node']} {rec['dim']} "
            f"max={rec['max_test_outside_train_px']:.2f}px at "
            f"t={rec['max_test_outside_time_s']:.2f}s; "
            f"largest final drift=node {final_rec['node']} {final_rec['dim']} "
            f"{final_rec['final_outside_train_px']:.2f}px"
        )


if __name__ == "__main__":
    main()
