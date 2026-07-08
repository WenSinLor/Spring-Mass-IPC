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

current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(src_dir))

from compute_task_specific_ipc import evaluate_sample
from run_hidden_node_prediction import train_hidden_node_readout
from task_specific_ipc_common import PALETTE, data_root, write_csv


HIDDEN_NODE = 10
REFERENCE_NODE = 0
HORIZON_STEPS = 5
WASHOUT_S = 5.0
TRAIN_S = 10.0
TEST_S = 10.0
RIDGE_ALPHA = 1e-6
H_CUT = 27
D_CUT = 1
LAG_STRIDE_FRAMES = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal mixed-amplitude topology 10-17 task-specific IPC summary "
            "with optional topology/sample exclusions."
        )
    )
    parser.add_argument(
        "--exclude-sample",
        action="append",
        default=[],
        metavar="TOPOLOGY:SAMPLE",
        help=(
            "Exclude one sample from the summary, e.g. "
            "topology_17_prestress:sample_1. Can be repeated."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Optional suffix appended to the output folder name.",
    )
    return parser.parse_args()


def parse_exclusions(values):
    excluded = {}
    for value in values:
        if ":" not in value:
            raise ValueError(
                f"Invalid --exclude-sample value '{value}'. Use TOPOLOGY:SAMPLE."
            )
        topology, sample = [part.strip() for part in value.split(":", 1)]
        if not topology or not sample:
            raise ValueError(
                f"Invalid --exclude-sample value '{value}'. Use TOPOLOGY:SAMPLE."
            )
        excluded.setdefault(topology, set()).add(sample)
    return excluded


def exclusion_suffix(excluded):
    parts = []
    for topology in sorted(excluded):
        for sample in sorted(excluded[topology]):
            parts.append(f"exclude_{topology}_{sample}")
    return "_".join(parts)


class Args:
    amplitude = ""
    hidden_node = HIDDEN_NODE
    reference_node = REFERENCE_NODE
    horizon_steps = HORIZON_STEPS
    washout = WASHOUT_S
    train = TRAIN_S
    test = TEST_S
    cv_blocks = 4
    global_standardize = True


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


def topology_index(name):
    return int(str(name).split("_")[1])


def choose_topology_folder(root, idx):
    for name in (f"topology_{idx}_prestress", f"topology_{idx}"):
        folder = root / name
        if folder.exists():
            return folder
    return None


def choose_amplitude_folder(topology_folder):
    amp_dirs = sorted(
        p
        for p in topology_folder.glob("amp=*")
        if p.is_dir() and any((s / "experiment.h5").exists() for s in p.glob("sample_*"))
    )
    if not amp_dirs:
        return None
    for preferred in ("amp=1", "amp=2.5"):
        for amp_dir in amp_dirs:
            if amp_dir.name == preferred:
                return amp_dir
    return amp_dirs[-1]


def node_count(sample_dir):
    with h5py.File(sample_dir / "experiment.h5", "r") as f:
        return int(f["time_series/nodes/positions"].shape[1])


def discover_mixed_family(root, excluded_samples):
    records = []
    skipped = []
    excluded_rows = []
    for idx in range(10, 18):
        topology_folder = choose_topology_folder(root, idx)
        if topology_folder is None:
            skipped.append(
                {
                    "topology_index": idx,
                    "reason": "no topology folder found",
                    "chosen_topology": "",
                    "chosen_amplitude": "",
                }
            )
            continue
        amp_dir = choose_amplitude_folder(topology_folder)
        if amp_dir is None:
            skipped.append(
                {
                    "topology_index": idx,
                    "reason": "no amp=*/sample_*/experiment.h5 found",
                    "chosen_topology": topology_folder.name,
                    "chosen_amplitude": "",
                }
            )
            continue
        sample_dirs = sorted(
            s for s in amp_dir.glob("sample_*") if s.is_dir() and (s / "experiment.h5").exists()
        )
        requested_exclusions = excluded_samples.get(topology_folder.name, set())
        if requested_exclusions:
            kept_sample_dirs = []
            found_samples = {s.name for s in sample_dirs}
            for sample_dir in sample_dirs:
                if sample_dir.name in requested_exclusions:
                    excluded_rows.append(
                        {
                            "topology_index": idx,
                            "topology": topology_folder.name,
                            "amplitude": amp_dir.name,
                            "sample": sample_dir.name,
                            "reason": "user excluded",
                        }
                    )
                else:
                    kept_sample_dirs.append(sample_dir)
            for sample in sorted(requested_exclusions - found_samples):
                excluded_rows.append(
                    {
                        "topology_index": idx,
                        "topology": topology_folder.name,
                        "amplitude": amp_dir.name,
                        "sample": sample,
                        "reason": "requested exclusion sample not found",
                    }
                )
            sample_dirs = kept_sample_dirs
        if not sample_dirs:
            skipped.append(
                {
                    "topology_index": idx,
                    "reason": "all samples excluded",
                    "chosen_topology": topology_folder.name,
                    "chosen_amplitude": amp_dir.name,
                }
            )
            continue
        n_nodes = node_count(sample_dirs[0])
        if HIDDEN_NODE >= n_nodes:
            skipped.append(
                {
                    "topology_index": idx,
                    "reason": f"hidden_node {HIDDEN_NODE} outside node range 0..{n_nodes - 1}",
                    "chosen_topology": topology_folder.name,
                    "chosen_amplitude": amp_dir.name,
                }
            )
            continue
        records.append(
            {
                "topology_index": idx,
                "topology": topology_folder.name,
                "amplitude": amp_dir.name,
                "sample_dirs": sample_dirs,
                "num_nodes": n_nodes,
            }
        )
    return records, skipped, excluded_rows


def dictionary_payload():
    return {
        "script": "run_mixed_topology_minimal_ipc_summary.py",
        "purpose": "minimal mixed-amplitude topology-family task-specific IPC summary",
        "H_cut": H_CUT,
        "D_cut": D_CUT,
        "H_max_initial": H_CUT,
        "D_max_initial": D_CUT,
        "horizon_steps": HORIZON_STEPS,
        "lag_stride_frames": LAG_STRIDE_FRAMES,
        "readout_time_delays": list(range(H_CUT + 1)),
        "target_time_delays": [HORIZON_STEPS + q for q in range(H_CUT + 1)],
        "adequacy_label": "fixed common H27 dictionary",
        "num_terms_selected": H_CUT + 1,
        "hidden_node": HIDDEN_NODE,
        "reference_node": REFERENCE_NODE,
        "washout_s": WASHOUT_S,
        "train_s": TRAIN_S,
        "test_s": TEST_S,
        "basis_window_s": TRAIN_S + TEST_S,
    }


def summarize(per_sample_rows):
    summary = []
    for topology in sorted({r["topology"] for r in per_sample_rows}, key=topology_index):
        group = [r for r in per_sample_rows if r["topology"] == topology]
        row = {
            "topology_index": int(group[0]["topology_index"]),
            "topology": topology,
            "amplitude": group[0]["amplitude"],
            "num_samples": len(group),
            "num_nodes": int(group[0]["num_nodes"]),
        }
        for key in (
            "actual_test_nmse_2d",
            "Q_twIPC_clipped",
            "generic_total_ipc_clipped",
            "basis_nmse_2d",
        ):
            values = np.asarray([float(r[key]) for r in group], dtype=float)
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values, ddof=0))
        summary.append(row)
    return summary


def save_minimal_plot(summary, out_path):
    labels = [str(r["topology_index"]) for r in summary]
    x = np.arange(len(summary))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(summary))]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.3), sharex=True, constrained_layout=True)
    axes[0].bar(
        x,
        [r["actual_test_nmse_2d_mean"] for r in summary],
        yerr=[r["actual_test_nmse_2d_std"] for r in summary],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[0].set_ylabel("test NMSE, 2D")
    axes[0].set_title("Actual hidden-node prediction error; lower is better")
    axes[0].grid(axis="y", color="#E5E7EB", lw=0.6)

    axes[1].bar(
        x,
        [r["basis_nmse_2d_mean"] for r in summary],
        yerr=[r["basis_nmse_2d_std"] for r in summary],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[1].set_ylabel("basis NMSE, 2D")
    axes[1].set_title("Input-history basis fit to hidden-node target; lower is better")
    axes[1].grid(axis="y", color="#E5E7EB", lw=0.6)

    axes[2].bar(
        x,
        [r["Q_twIPC_clipped_mean"] for r in summary],
        yerr=[r["Q_twIPC_clipped_std"] for r in summary],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[2].set_ylabel("Q_twIPC")
    axes[2].set_title("Task-specific IPC on fixed H=27, D=1 dictionary; higher is better")
    axes[2].set_xlabel("topology index")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].grid(axis="y", color="#E5E7EB", lw=0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_text_summary(summary, skipped, excluded_rows, out_path):
    best_nmse = min(summary, key=lambda r: r["actual_test_nmse_2d_mean"])
    best_basis = min(summary, key=lambda r: r["basis_nmse_2d_mean"])
    best_q = max(summary, key=lambda r: r["Q_twIPC_clipped_mean"])
    lines = [
        "Mixed-amplitude minimal topology summary",
        "========================================",
        "",
        "Included topology folders use their available amplitude folders. "
        "All included samples use hidden_node=10, reference_node=0, horizon_steps=5.",
        "",
        "The task-specific IPC uses one fixed common dictionary for every included topology:",
        "  H_cut=27, D_cut=1, terms=[P1(u_m), ..., P1(u_{m-27})].",
        "",
        f"Lowest actual hidden-node NMSE: topology {best_nmse['topology_index']} "
        f"({best_nmse['topology']}, {best_nmse['amplitude']}), "
        f"NMSE={best_nmse['actual_test_nmse_2d_mean']:.4f}.",
        f"Lowest basis NMSE: topology {best_basis['topology_index']} "
        f"({best_basis['topology']}, {best_basis['amplitude']}), "
        f"basis NMSE={best_basis['basis_nmse_2d_mean']:.4f}.",
        f"Highest Q_twIPC: topology {best_q['topology_index']} "
        f"({best_q['topology']}, {best_q['amplitude']}), "
        f"Q_twIPC={best_q['Q_twIPC_clipped_mean']:.4f}.",
        "",
        "Excluded samples:",
    ]
    if excluded_rows:
        for row in excluded_rows:
            lines.append(
                f"  {row['topology']}/{row['amplitude']}/{row['sample']}: {row['reason']}"
            )
    else:
        lines.append("  none")
    lines.extend(
        [
            "",
        "Skipped topology indices:",
        ]
    )
    for row in skipped:
        lines.append(
            f"  {row['topology_index']}: {row['reason']} "
            f"({row['chosen_topology']} {row['chosen_amplitude']})"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main():
    cli_args = parse_args()
    excluded_samples = parse_exclusions(cli_args.exclude_sample)
    configure_matplotlib()
    root = data_root()
    suffix = cli_args.output_suffix
    if suffix is None and excluded_samples:
        suffix = exclusion_suffix(excluded_samples)
    folder_name = "mixed_topology_10_to_17_H27_minimal"
    if suffix:
        folder_name = f"{folder_name}_{suffix}"
    out_dir = root / "task_specific_ipc" / folder_name
    records, skipped, excluded_rows = discover_mixed_family(root, excluded_samples)
    dictionary = dictionary_payload()
    (out_dir / "fixed_common_H27_dictionary.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "fixed_common_H27_dictionary.json").write_text(json.dumps(dictionary, indent=2))

    per_sample_rows = []
    coefficient_rows = []
    task_weight_rows = []
    ipc_rows = []
    term_rows = []
    grouped_rows = []

    for record in records:
        args = Args()
        args.amplitude = record["amplitude"]
        print(f"-> Mixed summary: {record['topology']}/{record['amplitude']}")
        for sample_dir in record["sample_dirs"]:
            hidden = train_hidden_node_readout(
                sample_dir,
                HIDDEN_NODE,
                REFERENCE_NODE,
                WASHOUT_S,
                TRAIN_S,
                TEST_S,
                RIDGE_ALPHA,
                HORIZON_STEPS,
            )
            coeffs, weights, ipc, terms, grouped, score = evaluate_sample(
                sample_dir, record["topology"], args, dictionary
            )
            coefficient_rows.extend(coeffs)
            task_weight_rows.extend(weights)
            ipc_rows.extend(ipc)
            term_rows.extend(terms)
            grouped_rows.extend(grouped)
            per_sample_rows.append(
                {
                    "topology_index": record["topology_index"],
                    "topology": record["topology"],
                    "amplitude": record["amplitude"],
                    "sample": sample_dir.name,
                    "num_nodes": record["num_nodes"],
                    "hidden_node": HIDDEN_NODE,
                    "reference_node": REFERENCE_NODE,
                    "horizon_steps": HORIZON_STEPS,
                    "H_cut": H_CUT,
                    "D_cut": D_CUT,
                    "actual_test_nmse_2d": hidden["metrics"]["test_nmse_2d"],
                    "actual_test_nmse_x": hidden["metrics"]["test_nmse_x"],
                    "actual_test_nmse_y": hidden["metrics"]["test_nmse_y"],
                    "basis_nmse_2d": score["basis_nmse_2d"],
                    "Q_twIPC_raw": score["Q_twIPC_raw"],
                    "Q_twIPC_clipped": score["Q_twIPC_clipped"],
                    "generic_total_ipc_clipped": score["generic_total_ipc_clipped"],
                }
            )

    summary = summarize(per_sample_rows)
    write_csv(per_sample_rows, out_dir / "mixed_topology_minimal_per_sample.csv")
    write_csv(summary, out_dir / "mixed_topology_minimal_summary.csv")
    if skipped:
        write_csv(skipped, out_dir / "mixed_topology_minimal_skipped.csv")
    else:
        (out_dir / "mixed_topology_minimal_skipped.csv").write_text(
            "topology_index,reason,chosen_topology,chosen_amplitude\n"
        )
    if excluded_rows:
        write_csv(excluded_rows, out_dir / "mixed_topology_minimal_excluded_samples.csv")
    else:
        (out_dir / "mixed_topology_minimal_excluded_samples.csv").write_text(
            "topology_index,topology,amplitude,sample,reason\n"
        )
    write_csv(coefficient_rows, out_dir / "task_coefficients_all_samples.csv")
    write_csv(task_weight_rows, out_dir / "task_weights_per_sample.csv")
    write_csv(ipc_rows, out_dir / "ipc_per_term_per_sample.csv")
    write_csv(term_rows, out_dir / "task_specific_ipc_per_term_per_sample.csv")
    write_csv(grouped_rows, out_dir / "task_specific_ipc_grouped_contributions.csv")
    save_minimal_plot(summary, out_dir / "mixed_topology_minimal_summary.pdf")
    write_text_summary(
        summary,
        skipped,
        excluded_rows,
        out_dir / "mixed_topology_minimal_interpretation.txt",
    )

    print(f"Saved mixed minimal outputs to: {out_dir}")
    for row in summary:
        print(
            f"  topology {row['topology_index']:>2} ({row['amplitude']}): "
            f"NMSE={row['actual_test_nmse_2d_mean']:.4f}, "
            f"basis NMSE={row['basis_nmse_2d_mean']:.4f}, "
            f"Q_twIPC={row['Q_twIPC_clipped_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
