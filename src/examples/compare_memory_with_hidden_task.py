import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt


current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))


DEFAULT_AMPLITUDE = "amp=1"
DEFAULT_HIDDEN_NODE = 10
DEFAULT_REFERENCE_NODE = 0
DEFAULT_HORIZON_STEPS = 5

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
        description="Compare fixed delayed-input memory with actual hidden-node prediction NMSE."
    )
    parser.add_argument("--amplitude", default=DEFAULT_AMPLITUDE)
    parser.add_argument("--hidden-node", type=int, default=DEFAULT_HIDDEN_NODE)
    parser.add_argument("--reference-node", type=int, default=DEFAULT_REFERENCE_NODE)
    parser.add_argument("--horizon-steps", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument("--memory-metrics-csv", default=None)
    parser.add_argument(
        "--hidden-metrics-csv",
        default=None,
        help="Optional comma-separated hidden-node metrics CSV paths. If omitted, auto-discover.",
    )
    parser.add_argument("--output-dir", default=None)
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


def as_float(row, key):
    return float(row[key])


def discover_hidden_metrics(data_root, amplitude, hidden_node, horizon_steps):
    paths = sorted(
        data_root.glob(
            f"*/{amplitude}/plots/hidden_node_{hidden_node}_prediction/"
            f"horizon_{horizon_steps}_steps/"
            f"hidden_node_{hidden_node}_h{horizon_steps}_prediction_metrics.csv"
        )
    )
    return paths


def load_hidden_metrics(paths):
    rows = []
    for path in paths:
        for row in read_csv(path):
            if "topology" not in row or not row["topology"]:
                row["topology"] = Path(path).parents[4].name
            if "amplitude" not in row or not row["amplitude"]:
                row["amplitude"] = Path(path).parents[3].name
            rows.append(row)
    return rows


def merge_rows(memory_rows, hidden_rows, hidden_node, reference_node, horizon_steps):
    hidden_by_key = {}
    for row in hidden_rows:
        key = (
            row["topology"],
            row["sample"],
            int(float(row["hidden_node"])),
            int(float(row["reference_node"])),
            int(float(row["horizon_steps"])),
        )
        hidden_by_key[key] = row

    merged = []
    for m in memory_rows:
        key = (
            m["topology"],
            m["sample"],
            int(float(m["hidden_node"])),
            int(float(m["reference_node"])),
            int(float(m["horizon_steps"])),
        )
        if key not in hidden_by_key:
            continue
        h = hidden_by_key[key]
        if int(float(h["hidden_node"])) != hidden_node:
            continue
        if int(float(h["reference_node"])) != reference_node:
            continue
        if int(float(h["horizon_steps"])) != horizon_steps:
            continue
        merged.append(
            {
                "topology": m["topology"],
                "amplitude": m.get("amplitude", h.get("amplitude", "")),
                "sample": m["sample"],
                "hidden_node": hidden_node,
                "reference_node": reference_node,
                "horizon_steps": horizon_steps,
                "M_0_5": float(m["M_0_5"]),
                "M_0_10": float(m["M_0_10"]),
                "M_0_relevant": float(m["M_0_relevant"]),
                "M_total": float(m["M_total"]),
                "peak_tau": float(m["peak_tau"]),
                "memory_length_threshold_0p05": float(m["memory_length_threshold_0p05"]),
                "test_nmse_2d": float(h["test_nmse_2d"]),
                "test_nmse_x": float(h["test_nmse_x"]),
                "test_nmse_y": float(h["test_nmse_y"]),
            }
        )
    return merged


def summarize_topology(merged):
    summary = []
    for topology in sorted({r["topology"] for r in merged}):
        group = [r for r in merged if r["topology"] == topology]
        row = {"topology": topology, "num_samples": len(group)}
        for key in ("test_nmse_2d", "M_0_5", "M_0_10", "M_total"):
            vals = np.asarray([r[key] for r in group], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals))
            row[f"{key}_std"] = float(np.nanstd(vals, ddof=0))
        summary.append(row)
    return summary


def correlation_or_ranking(summary):
    topologies = [r["topology"] for r in summary]
    if len(summary) >= 3:
        lines = []
        for metric in ("M_0_5", "M_0_10", "M_total"):
            x = np.asarray([r[f"{metric}_mean"] for r in summary], dtype=float)
            y = -np.asarray([r["test_nmse_2d_mean"] for r in summary], dtype=float)
            rho, p = spearmanr(x, y)
            lines.append(f"Spearman({metric}, -test_nmse_2d): rho={rho:.3f}, p={p:.3g}")
        return lines
    if len(summary) == 2:
        a, b = summary
        lines = []
        for metric in ("M_0_5", "M_0_10"):
            higher_memory = a if a[f"{metric}_mean"] > b[f"{metric}_mean"] else b
            lower_nmse = a if a["test_nmse_2d_mean"] < b["test_nmse_2d_mean"] else b
            lines.append(
                f"Ranking consistency for {metric}: "
                f"{higher_memory['topology']} has higher memory; "
                f"{lower_nmse['topology']} has lower hidden-node NMSE; "
                f"consistent={higher_memory['topology'] == lower_nmse['topology']}"
            )
        return lines
    return ["Need at least two topologies for ranking comparison."]


def plot_comparison(summary, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("M_0_5", "M_0_10"):
        fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
        for i, row in enumerate(summary):
            ax.scatter(
                row[f"{metric}_mean"],
                row["test_nmse_2d_mean"],
                s=44,
                color=PALETTE[i % len(PALETTE)],
            )
            ax.text(
                row[f"{metric}_mean"],
                row["test_nmse_2d_mean"],
                row["topology"],
                fontsize=6,
                ha="left",
                va="bottom",
            )
        ax.set_xlabel(metric)
        ax.set_ylabel("hidden-node test NMSE, 2D")
        ax.set_title(f"Memory versus hidden-node error ({metric})")
        ax.grid(axis="y", color="#E5E7EB", lw=0.6)
        fig.savefig(out_dir / f"memory_vs_hidden_nmse_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)

    topologies = [r["topology"] for r in summary]
    x = np.arange(len(topologies))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), constrained_layout=True)
    axes[0].bar(x, [r["M_0_5_mean"] for r in summary], color="#0072B2")
    axes[0].set_title("Forecast-relevant memory")
    axes[0].set_ylabel("M_0_5")
    axes[1].bar(x, [r["test_nmse_2d_mean"] for r in summary], color="#D55E00")
    axes[1].set_title("Hidden-node error")
    axes[1].set_ylabel("test NMSE, 2D")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(topologies, rotation=25, ha="right")
    fig.savefig(out_dir / "memory_hidden_task_bar_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def interpretation(summary):
    if len(summary) < 2:
        return "Need multiple topologies to interpret memory-performance ranking."
    best_memory = max(summary, key=lambda r: r["M_0_5_mean"])
    best_task = min(summary, key=lambda r: r["test_nmse_2d_mean"])
    if best_memory["topology"] == best_task["topology"]:
        return (
            f"{best_memory['topology']} has stronger fixed delayed-input memory over the "
            "forecast-relevant delay range and also achieves lower hidden-node prediction "
            "NMSE. This supports the interpretation that memory-enhanced visible-state "
            "encoding contributes to better hidden-node prediction."
        )
    return (
        "The hidden-node performance difference is not explained by linear delayed-input "
        "memory alone. Other factors such as hidden-state observability, spatial coupling, "
        "noise, or nonlinear state encoding may dominate."
    )


def main():
    configure_matplotlib()
    args = parse_args()
    data_root = src_dir.parent / "data" / "experiment_data"
    default_out_dir = (
        data_root
        / "fixed_memory_analysis"
        / args.amplitude
        / f"hidden_node_{args.hidden_node}"
        / f"horizon_{args.horizon_steps}_steps"
    )
    memory_csvs = (
        [Path(p.strip()) for p in args.memory_metrics_csv.split(",") if p.strip()]
        if args.memory_metrics_csv
        else [default_out_dir / "linear_memory_capacity_metrics.csv"]
    )
    if args.hidden_metrics_csv:
        hidden_paths = [Path(p.strip()) for p in args.hidden_metrics_csv.split(",") if p.strip()]
    else:
        hidden_paths = discover_hidden_metrics(
            data_root, args.amplitude, args.hidden_node, args.horizon_steps
        )

    memory_rows = []
    for memory_csv in memory_csvs:
        memory_rows.extend(read_csv(memory_csv))
    hidden_rows = load_hidden_metrics(hidden_paths)
    merged = merge_rows(
        memory_rows,
        hidden_rows,
        args.hidden_node,
        args.reference_node,
        args.horizon_steps,
    )
    if not merged:
        raise ValueError("No matching memory/hidden-node rows found. Rerun hidden prediction metrics if needed.")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif len(memory_csvs) == 1:
        out_dir = memory_csvs[0].parent / "memory_hidden_task_comparison"
    else:
        out_dir = (
            data_root
            / "fixed_memory_analysis"
            / "topology_17_vs_topology_10"
            / f"hidden_node_{args.hidden_node}"
            / f"horizon_{args.horizon_steps}_steps"
        )
    summary = summarize_topology(merged)
    write_csv(merged, out_dir / "memory_hidden_task_merged_per_sample.csv")
    write_csv(summary, out_dir / "memory_hidden_task_summary_by_topology.csv")
    plot_comparison(summary, out_dir)

    lines = correlation_or_ranking(summary)
    interp = interpretation(summary)
    report = "\n".join(["# Memory vs Hidden-Node Task Comparison", "", *lines, "", interp, ""])
    (out_dir / "memory_hidden_task_interpretation.txt").write_text(report)

    print(f"Saved comparison outputs to: {out_dir}")
    for line in lines:
        print(line)
    print(interp)


if __name__ == "__main__":
    main()
