import argparse
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from task_specific_ipc_common import (
    PALETTE,
    configure_matplotlib,
    data_root,
    parse_csv_paths,
    read_csv,
    topology_index,
    write_csv,
)


DEFAULT_AMPLITUDE = "amp=1"
DEFAULT_HIDDEN_NODE = 10
DEFAULT_REFERENCE_NODE = 0
DEFAULT_HORIZON_STEPS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare task-specific IPC scores with actual hidden-node prediction NMSE."
    )
    parser.add_argument("--twipc-csv", required=True)
    parser.add_argument(
        "--hidden-metrics-csv",
        default=None,
        help="Comma-separated hidden-node prediction metrics CSV paths. Auto-discovered if omitted.",
    )
    parser.add_argument("--grouped-contributions-csv", default=None)
    parser.add_argument("--amplitude", default=DEFAULT_AMPLITUDE)
    parser.add_argument("--hidden-node", type=int, default=DEFAULT_HIDDEN_NODE)
    parser.add_argument("--reference-node", type=int, default=DEFAULT_REFERENCE_NODE)
    parser.add_argument("--horizon-steps", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def discover_hidden_metrics(root, amplitude, hidden_node, horizon_steps):
    return sorted(
        Path(root).glob(
            f"*/{amplitude}/plots/hidden_node_{hidden_node}_prediction/"
            f"horizon_{horizon_steps}_steps/"
            f"hidden_node_{hidden_node}_h{horizon_steps}_prediction_metrics.csv"
        )
    )


def load_hidden_metrics(paths):
    rows = []
    for path in paths:
        for row in read_csv(path):
            if not row.get("topology"):
                row["topology"] = Path(path).parents[4].name
            if not row.get("amplitude"):
                row["amplitude"] = Path(path).parents[3].name
            rows.append(row)
    return rows


def merge_rows(twipc_rows, hidden_rows, hidden_node, reference_node, horizon_steps):
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
    for row in twipc_rows:
        key = (
            row["topology"],
            row["sample"],
            int(float(row["hidden_node"])),
            int(float(row["reference_node"])),
            int(float(row["horizon_steps"])),
        )
        if key not in hidden_by_key:
            continue
        hidden = hidden_by_key[key]
        if key[2] != hidden_node or key[3] != reference_node or key[4] != horizon_steps:
            continue
        merged.append(
            {
                "topology": row["topology"],
                "amplitude": row.get("amplitude", hidden.get("amplitude", "")),
                "sample": row["sample"],
                "hidden_node": hidden_node,
                "reference_node": reference_node,
                "horizon_steps": horizon_steps,
                "Q_twIPC_raw": float(row["Q_twIPC_raw"]),
                "Q_twIPC_clipped": float(row["Q_twIPC_clipped"]),
                "generic_total_ipc_clipped": float(row.get("generic_total_ipc_clipped", np.nan)),
                "basis_nmse_2d": float(row.get("basis_nmse_2d", np.nan)),
                "basis_adequacy_label": row.get("basis_adequacy_label", ""),
                "actual_test_nmse_2d": float(hidden["test_nmse_2d"]),
                "actual_test_nmse_x": float(hidden["test_nmse_x"]),
                "actual_test_nmse_y": float(hidden["test_nmse_y"]),
            }
        )
    return merged


def summarize_topology(merged):
    summary = []
    for topology in sorted({r["topology"] for r in merged}, key=topology_index):
        group = [r for r in merged if r["topology"] == topology]
        row = {
            "topology": topology,
            "num_samples": len(group),
            "basis_adequacy_label": group[0].get("basis_adequacy_label", ""),
        }
        for key in (
            "actual_test_nmse_2d",
            "Q_twIPC_clipped",
            "Q_twIPC_raw",
            "generic_total_ipc_clipped",
            "basis_nmse_2d",
        ):
            values = np.asarray([r[key] for r in group], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(values))
            row[f"{key}_std"] = float(np.nanstd(values, ddof=0))
        summary.append(row)
    return summary


def correlation_or_ranking(summary):
    if len(summary) >= 3:
        y = -np.asarray([r["actual_test_nmse_2d_mean"] for r in summary], dtype=float)
        q = np.asarray([r["Q_twIPC_clipped_mean"] for r in summary], dtype=float)
        rho_q, p_q = spearmanr(q, y)
        lines = [f"Spearman(Q_twIPC_clipped, -test_nmse_2d): rho={rho_q:.3f}, p={p_q:.3g}"]
        total = np.asarray([r["generic_total_ipc_clipped_mean"] for r in summary], dtype=float)
        if np.all(np.isfinite(total)):
            rho_total, p_total = spearmanr(total, y)
            lines.append(
                f"Spearman(generic_total_ipc, -test_nmse_2d): rho={rho_total:.3f}, p={p_total:.3g}"
            )
        return lines
    if len(summary) == 2:
        a, b = summary
        higher_q = a if a["Q_twIPC_clipped_mean"] > b["Q_twIPC_clipped_mean"] else b
        lower_nmse = a if a["actual_test_nmse_2d_mean"] < b["actual_test_nmse_2d_mean"] else b
        return [
            "Only two topologies are available, so correlation is not meaningful.",
            f"Ranking consistency: {higher_q['topology']} has higher Q_twIPC; "
            f"{lower_nmse['topology']} has lower actual NMSE; "
            f"consistent={higher_q['topology'] == lower_nmse['topology']}",
        ]
    return ["Need at least two topologies for comparison."]


def interpretation(summary):
    if len(summary) < 2:
        return "Need multiple topologies to interpret Q_twIPC-performance ranking."
    best_q = max(summary, key=lambda r: r["Q_twIPC_clipped_mean"])
    best_task = min(summary, key=lambda r: r["actual_test_nmse_2d_mean"])
    if best_q["topology"] == best_task["topology"]:
        return (
            "The lower hidden-node prediction NMSE is associated with a higher "
            "task-specific IPC score under the common dictionary. This indicates "
            "a better match between the topology-conditioned task demand and the "
            "topology's linearly decodable memory/nonlinear capacity."
        )
    return (
        "The task-specific IPC under the scalar input-history dictionary does not "
        "explain the observed task ranking. This suggests that hidden-state "
        "observability, spatial coupling, unmeasured dynamics, or non-input-driven "
        "state information dominate."
    )


def panel_label(ax, label):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=8, fontweight="bold")


def plot_comparison(merged, summary, grouped_rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
    topologies = sorted({r["topology"] for r in merged}, key=topology_index)
    for i, topology in enumerate(topologies):
        group = [r for r in merged if r["topology"] == topology]
        ax.scatter(
            [r["Q_twIPC_clipped"] for r in group],
            [r["actual_test_nmse_2d"] for r in group],
            s=32,
            color=PALETTE[i % len(PALETTE)],
            alpha=0.75,
            label=topology,
        )
    ax.set_xlabel("Q_twIPC, clipped")
    ax.set_ylabel("actual hidden-node test NMSE, 2D")
    ax.set_title("Task-specific IPC versus task error")
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    ax.legend(fontsize=5)
    fig.savefig(out_dir / "twipc_vs_hidden_nmse_per_sample.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True, constrained_layout=True)
    x = np.arange(len(summary))
    labels = [r["topology"] for r in summary]
    axes[0].bar(
        x,
        [r["actual_test_nmse_2d_mean"] for r in summary],
        yerr=[r["actual_test_nmse_2d_std"] for r in summary],
        color="#D55E00",
    )
    axes[0].set_ylabel("test NMSE, 2D")
    axes[0].set_title("Actual hidden-node prediction error")
    axes[1].bar(
        x,
        [r["Q_twIPC_clipped_mean"] for r in summary],
        yerr=[r["Q_twIPC_clipped_std"] for r in summary],
        color="#0072B2",
    )
    axes[1].set_ylabel("Q_twIPC")
    axes[1].set_title("Task-specific IPC")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    for ax in axes:
        ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    fig.savefig(out_dir / "twipc_hidden_nmse_topology_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    if grouped_rows:
        delay_rows = [r for r in grouped_rows if r["group_type"] == "readout_time_delay"]
        chosen = sorted({r["topology"] for r in delay_rows}, key=topology_index)[:2]
        if chosen:
            fig, ax = plt.subplots(figsize=(6.4, 3.0), constrained_layout=True)
            width = 0.8 / max(len(chosen), 1)
            for i, topology in enumerate(chosen):
                group = [r for r in delay_rows if r["topology"] == topology]
                delays = sorted({int(float(r["group_value"])) for r in group})
                means = []
                for delay in delays:
                    vals = [
                        float(r["weighted_contribution_sum"])
                        for r in group
                        if int(float(r["group_value"])) == delay
                    ]
                    means.append(float(np.nanmean(vals)))
                x = np.asarray(delays, dtype=float) + (i - (len(chosen) - 1) / 2) * width
                ax.bar(x, means, width=width, color=PALETTE[i % len(PALETTE)], label=topology)
            ax.set_xlabel("readout-time delay")
            ax.set_ylabel("mean c_alpha * IPC_alpha")
            ax.set_title("Delay-resolved task-specific IPC contribution")
            ax.legend(fontsize=6)
            ax.grid(axis="y", color="#E5E7EB", lw=0.6)
            fig.savefig(out_dir / "twipc_delay_resolved_contribution.pdf", bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.0), sharex=True, constrained_layout=True)
            first = chosen[0]
            group = [r for r in delay_rows if r["topology"] == first]
            delays = sorted({int(float(r["group_value"])) for r in group})
            for ax, key, title, color in (
                (axes[0], "c_alpha_sum", "Task demand c_alpha", "#009E73"),
                (axes[1], "IPC_alpha_clipped_sum", "Reservoir supply IPC_alpha", "#0072B2"),
                (axes[2], "weighted_contribution_sum", "Demand-supply product", "#D55E00"),
            ):
                means = []
                for delay in delays:
                    vals = [
                        float(r[key])
                        for r in group
                        if int(float(r["group_value"])) == delay
                    ]
                    means.append(float(np.nanmean(vals)))
                ax.bar(delays, means, color=color)
                ax.set_ylabel(title)
                ax.grid(axis="y", color="#E5E7EB", lw=0.6)
            axes[-1].set_xlabel("readout-time delay")
            fig.suptitle(f"Demand-supply decomposition: {first}", fontsize=8)
            fig.savefig(out_dir / "twipc_demand_supply_decomposition.pdf", bbox_inches="tight")
            plt.close(fig)


def main():
    configure_matplotlib()
    args = parse_args()
    root = data_root()
    twipc_path = Path(args.twipc_csv)
    twipc_rows = read_csv(twipc_path)
    hidden_paths = (
        parse_csv_paths(args.hidden_metrics_csv)
        if args.hidden_metrics_csv
        else discover_hidden_metrics(root, args.amplitude, args.hidden_node, args.horizon_steps)
    )
    hidden_rows = load_hidden_metrics(hidden_paths)
    merged = merge_rows(
        twipc_rows,
        hidden_rows,
        args.hidden_node,
        args.reference_node,
        args.horizon_steps,
    )
    if not merged:
        raise ValueError("No matching task-specific IPC and hidden-node metric rows found.")

    if args.grouped_contributions_csv:
        grouped_path = Path(args.grouped_contributions_csv)
    else:
        grouped_path = twipc_path.parent / "task_specific_ipc_grouped_contributions.csv"
    grouped_rows = read_csv(grouped_path) if grouped_path.exists() else []

    out_dir = Path(args.output_dir) if args.output_dir else twipc_path.parent / "twipc_hidden_nmse_comparison"
    summary = summarize_topology(merged)
    write_csv(merged, out_dir / "twipc_hidden_nmse_merged_per_sample.csv")
    write_csv(summary, out_dir / "twipc_hidden_nmse_summary_by_topology.csv")
    plot_comparison(merged, summary, grouped_rows, out_dir)

    lines = correlation_or_ranking(summary)
    interp = interpretation(summary)
    report = "\n".join(["# Task-Specific IPC vs Hidden-Node NMSE", "", *lines, "", interp, ""])
    (out_dir / "twipc_hidden_nmse_interpretation.txt").write_text(report)

    print(f"Saved task-specific IPC comparison outputs to: {out_dir}")
    for line in lines:
        print(line)
    print(interp)


if __name__ == "__main__":
    main()
