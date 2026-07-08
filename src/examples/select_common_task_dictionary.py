import argparse
import os
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from task_specific_ipc_common import (
    ALPHA_GRID,
    PALETTE,
    StateLoader,
    basis_adequacy_label,
    basis_window,
    build_legendre_design,
    configure_matplotlib,
    data_root,
    discover_sample_dirs,
    load_positions,
    make_blocked_folds,
    nmse_components,
    normalize_to_unit_interval,
    select_alpha_by_blocked_cv,
    target_time_delays,
    topology_names,
    valid_basis_rows,
    write_csv,
    write_json,
)


DEFAULT_TOPOLOGY_START = 10
DEFAULT_TOPOLOGY_STOP = 17
DEFAULT_TOPOLOGY_FORMAT = "topology_{i}_prestress"
DEFAULT_AMPLITUDE = "amp=1"
DEFAULT_HIDDEN_NODE = 10
DEFAULT_REFERENCE_NODE = 0
DEFAULT_HORIZON_STEPS = 5
DEFAULT_WASHOUT_S = 5.0
DEFAULT_TRAIN_S = 10.0
DEFAULT_TEST_S = 10.0
DEFAULT_H_MAX = 30
DEFAULT_D_MAX = 2
DEFAULT_CV_BLOCKS = 4
DEFAULT_LAG_STRIDE_FRAMES = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select one common input-history Legendre dictionary cutoff for "
            "task-specific IPC across a prestressed topology family."
        )
    )
    parser.add_argument("--topology-start", type=int, default=DEFAULT_TOPOLOGY_START)
    parser.add_argument("--topology-stop", type=int, default=DEFAULT_TOPOLOGY_STOP)
    parser.add_argument("--topology-name-format", default=DEFAULT_TOPOLOGY_FORMAT)
    parser.add_argument("--amplitude", default=DEFAULT_AMPLITUDE)
    parser.add_argument("--sample", default="all")
    parser.add_argument("--hidden-node", type=int, default=DEFAULT_HIDDEN_NODE)
    parser.add_argument("--reference-node", type=int, default=DEFAULT_REFERENCE_NODE)
    parser.add_argument("--horizon-steps", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument("--washout", type=float, default=DEFAULT_WASHOUT_S)
    parser.add_argument("--train", type=float, default=DEFAULT_TRAIN_S)
    parser.add_argument("--test", type=float, default=DEFAULT_TEST_S)
    parser.add_argument("--H-max", type=int, default=DEFAULT_H_MAX)
    parser.add_argument("--D-max", type=int, default=DEFAULT_D_MAX)
    parser.add_argument("--cv-blocks", type=int, default=DEFAULT_CV_BLOCKS)
    parser.add_argument("--lag-stride-frames", type=int, default=DEFAULT_LAG_STRIDE_FRAMES)
    parser.add_argument(
        "--skip-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip topology/amplitude folders that are not present instead of failing.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def output_dir(args):
    if args.output_dir:
        return Path(args.output_dir)
    family = f"topology_{args.topology_start}_to_{args.topology_stop}"
    return (
        data_root()
        / "task_specific_ipc"
        / family
        / args.amplitude
        / f"hidden_node_{args.hidden_node}"
        / f"horizon_{args.horizon_steps}_steps"
        / "common_dictionary_selection"
    )


def evaluate_candidate(sample_dir, topology, args, H, D):
    loader = StateLoader(sample_dir / "experiment.h5")
    _, positions = load_positions(sample_dir / "experiment.h5")
    u_norm = normalize_to_unit_interval(loader.get_actuation_signal(actuator_idx=0, dof=0))
    hidden_relative = positions[:, args.hidden_node, :] - positions[:, args.reference_node, :]

    window_start, window_stop = basis_window(
        loader, args.washout, args.train, args.test, args.horizon_steps
    )
    rows = valid_basis_rows(window_start, window_stop, H, args.lag_stride_frames)
    P_full, exps = build_legendre_design(u_norm, H, D, args.lag_stride_frames)
    P = P_full[rows]
    Y = hidden_relative[rows + args.horizon_steps]
    if np.isnan(P).any():
        raise ValueError(f"NaNs remained in basis matrix for {topology}/{sample_dir.name}, H={H}, D={D}.")

    selected_alpha, folds = select_alpha_by_blocked_cv(
        P, Y, ALPHA_GRID, args.cv_blocks
    )
    min_train_rows = min(len(train_idx) for train_idx, _ in folds)
    number_of_terms = int(P.shape[1])
    terms_per_training_row = float(number_of_terms / max(min_train_rows, 1))
    overparameterized = number_of_terms > 0.8 * min_train_rows

    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(folds):
        model = Ridge(alpha=selected_alpha, fit_intercept=True)
        model.fit(P[train_idx], Y[train_idx])
        train_pred = model.predict(P[train_idx])
        val_pred = model.predict(P[val_idx])
        train_x, train_y, train_2d = nmse_components(Y[train_idx], train_pred)
        val_x, val_y, val_2d = nmse_components(Y[val_idx], val_pred)
        fold_rows.append(
            {
                "topology": topology,
                "amplitude": args.amplitude,
                "sample": sample_dir.name,
                "H": H,
                "D": D,
                "fold": fold_id,
                "selected_alpha": selected_alpha,
                "number_of_terms": number_of_terms,
                "min_training_rows_per_fold": int(min_train_rows),
                "terms_per_training_row": terms_per_training_row,
                "overparameterized": overparameterized,
                "condition_warning": "overparameterized_terms_gt_0.8_train_rows"
                if overparameterized
                else "",
                "train_nmse_x": train_x,
                "val_nmse_x": val_x,
                "train_nmse_y": train_y,
                "val_nmse_y": val_y,
                "train_nmse_2d": train_2d,
                "val_nmse_2d": val_2d,
            }
        )
    sample_row = summarize_folds(fold_rows)
    sample_row.update(
        {
            "topology": topology,
            "amplitude": args.amplitude,
            "sample": sample_dir.name,
            "H": H,
            "D": D,
            "horizon_steps": args.horizon_steps,
            "lag_stride_frames": args.lag_stride_frames,
            "readout_time_delays": str(list(range(H + 1))),
            "target_time_delays": str(target_time_delays(H, args.horizon_steps, args.lag_stride_frames)),
            "number_of_terms": number_of_terms,
            "selected_alpha": selected_alpha,
            "min_training_rows_per_fold": int(min_train_rows),
            "terms_per_training_row": terms_per_training_row,
            "overparameterized": overparameterized,
            "condition_warning": "overparameterized_terms_gt_0.8_train_rows"
            if overparameterized
            else "",
            "num_usable_rows": int(len(P)),
        }
    )
    return fold_rows, sample_row, len(exps)


def summarize_folds(fold_rows):
    out = {}
    for key in (
        "train_nmse_x",
        "val_nmse_x",
        "train_nmse_y",
        "val_nmse_y",
        "train_nmse_2d",
        "val_nmse_2d",
    ):
        values = np.asarray([r[key] for r in fold_rows], dtype=float)
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values, ddof=0))
    out["train_validation_gap_2d"] = out["val_nmse_2d_mean"] - out["train_nmse_2d_mean"]
    return out


def summarize_candidates(sample_rows):
    summary = []
    for D in sorted({int(r["D"]) for r in sample_rows}):
        for H in sorted({int(r["H"]) for r in sample_rows if int(r["D"]) == D}):
            group = [r for r in sample_rows if int(r["H"]) == H and int(r["D"]) == D]
            row = {
                "H": H,
                "D": D,
                "horizon_steps": int(group[0]["horizon_steps"]),
                "lag_stride_frames": int(group[0]["lag_stride_frames"]),
                "readout_time_delays": str(list(range(H + 1))),
                "target_time_delays": str(target_time_delays(H, int(group[0]["horizon_steps"]), int(group[0]["lag_stride_frames"]))),
                "number_of_terms": int(group[0]["number_of_terms"]),
                "num_topology_sample_pairs": len(group),
                "overparameterized_fraction": float(np.mean([bool(r["overparameterized"]) for r in group])),
                "condition_warning": "some_samples_overparameterized"
                if any(bool(r["overparameterized"]) for r in group)
                else "",
            }
            for key in (
                "selected_alpha",
                "min_training_rows_per_fold",
                "terms_per_training_row",
                "train_nmse_x_mean",
                "val_nmse_x_mean",
                "train_nmse_y_mean",
                "val_nmse_y_mean",
                "train_nmse_2d_mean",
                "val_nmse_2d_mean",
                "train_validation_gap_2d",
            ):
                values = np.asarray([r[key] for r in group], dtype=float)
                row[key] = float(np.mean(values))
                row[f"{key}_std_across_samples"] = float(np.std(values, ddof=0))
            summary.append(row)
    return summary


def select_common_cutoff(summary):
    valid = [r for r in summary if np.isfinite(float(r["val_nmse_2d_mean"]))]
    if not valid:
        raise ValueError("No valid common dictionary candidates.")
    global_best = min(valid, key=lambda r: float(r["val_nmse_2d_mean"]))
    tolerance = 1.05 * float(global_best["val_nmse_2d_mean"])
    gap_ok = [
        r
        for r in valid
        if float(r["val_nmse_2d_mean"]) <= tolerance
        and float(r["train_validation_gap_2d"]) < max(0.10, 0.5 * float(r["val_nmse_2d_mean"]))
    ]
    if not gap_ok:
        gap_ok = [r for r in valid if float(r["val_nmse_2d_mean"]) <= tolerance]

    non_over = [r for r in gap_ok if float(r["overparameterized_fraction"]) == 0.0]
    eligible = non_over if non_over else gap_ok
    selected = min(
        eligible,
        key=lambda r: (int(r["D"]), int(r["H"]), int(r["number_of_terms"])),
    )
    return selected, global_best, tolerance


def write_selected_json(args, selected, global_best, out_dir, sample_rows, missing):
    label = basis_adequacy_label(float(global_best["val_nmse_2d_mean"]))
    H_cut = int(selected["H"])
    D_cut = int(selected["D"])
    selected_sample_rows = [
        r for r in sample_rows if int(r["H"]) == H_cut and int(r["D"]) == D_cut
    ]
    alpha_by_sample = {
        f"{r['topology']}/{r['sample']}": float(r["selected_alpha"])
        for r in selected_sample_rows
    }
    payload = {
        "script": "select_common_task_dictionary.py",
        "purpose": "common frozen dictionary for topology-family task-specific IPC",
        "H_cut": H_cut,
        "D_cut": D_cut,
        "H_max_initial": int(args.H_max),
        "D_max_initial": int(args.D_max),
        "horizon_steps": int(args.horizon_steps),
        "lag_stride_frames": int(args.lag_stride_frames),
        "readout_time_delays": list(range(H_cut + 1)),
        "target_time_delays": target_time_delays(H_cut, args.horizon_steps, args.lag_stride_frames),
        "selected_mean_cv_nmse_2d": float(selected["val_nmse_2d_mean"]),
        "selected_mean_train_nmse_2d": float(selected["train_nmse_2d_mean"]),
        "selected_train_validation_gap_2d": float(selected["train_validation_gap_2d"]),
        "global_best_H": int(global_best["H"]),
        "global_best_D": int(global_best["D"]),
        "global_best_mean_cv_nmse_2d": float(global_best["val_nmse_2d_mean"]),
        "adequacy_label": label,
        "num_terms_selected": int(selected["number_of_terms"]),
        "selected_alpha_mean": float(selected["selected_alpha"]),
        "selected_alpha_by_topology_sample": alpha_by_sample,
        "topology_start": int(args.topology_start),
        "topology_stop": int(args.topology_stop),
        "topology_name_format": args.topology_name_format,
        "amplitude": args.amplitude,
        "hidden_node": int(args.hidden_node),
        "reference_node": int(args.reference_node),
        "washout_s": float(args.washout),
        "train_s": float(args.train),
        "test_s": float(args.test),
        "basis_window_s": float(args.train + args.test),
        "missing_topology_amplitude_dirs": missing,
    }
    json_path = out_dir / "selected_common_dictionary.json"
    write_json(payload, json_path)
    return payload, json_path


def panel_label(ax, label):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=8, fontweight="bold")


def plot_summary(summary, sample_rows, selected, out_path):
    rows = sorted(summary, key=lambda r: (int(r["D"]), int(r["H"])))
    topologies = sorted({r["topology"] for r in sample_rows})
    fig = plt.figure(figsize=(7.4, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax_delay = fig.add_subplot(gs[0, 0])
    ax_terms = fig.add_subplot(gs[0, 1])
    ax_gap = fig.add_subplot(gs[1, 0])
    ax_heat1 = fig.add_subplot(gs[1, 1])
    ax_heat2 = fig.add_subplot(gs[2, :])

    for j, D in enumerate(sorted({int(r["D"]) for r in rows})):
        group = [r for r in rows if int(r["D"]) == D]
        color = PALETTE[j % len(PALETTE)]
        ax_delay.plot(
            [int(r["H"]) for r in group],
            [float(r["val_nmse_2d_mean"]) for r in group],
            marker="o",
            lw=1.2,
            color=color,
            label=f"D={D}",
        )
    ax_delay.scatter(
        int(selected["H"]),
        float(selected["val_nmse_2d_mean"]),
        marker="*",
        s=120,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    ax_delay.set_xlabel("extra readout-time history H (frames)")
    ax_delay.set_ylabel("mean CV NMSE, 2D")
    ax_delay.set_title("Common cutoff sweep")
    ax_delay.grid(axis="y", color="#E5E7EB", lw=0.6)
    ax_delay.legend(fontsize=6)
    panel_label(ax_delay, "a")

    colors = [PALETTE[(int(r["D"]) - 1) % len(PALETTE)] for r in rows]
    ax_terms.scatter(
        [int(r["number_of_terms"]) for r in rows],
        [float(r["val_nmse_2d_mean"]) for r in rows],
        c=colors,
        s=32,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_terms.scatter(
        int(selected["number_of_terms"]),
        float(selected["val_nmse_2d_mean"]),
        marker="*",
        s=120,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    ax_terms.set_xscale("log")
    ax_terms.set_xlabel("number of nonconstant terms")
    ax_terms.set_ylabel("mean CV NMSE, 2D")
    ax_terms.set_title("Basis size versus validation error")
    ax_terms.grid(axis="y", color="#E5E7EB", lw=0.6)
    panel_label(ax_terms, "b")

    ax_gap.scatter(
        [float(r["train_nmse_2d_mean"]) for r in rows],
        [float(r["val_nmse_2d_mean"]) for r in rows],
        c=colors,
        s=32,
        edgecolor="white",
        linewidth=0.5,
    )
    lim = 1.05 * max(
        max(float(r["train_nmse_2d_mean"]) for r in rows),
        max(float(r["val_nmse_2d_mean"]) for r in rows),
    )
    ax_gap.plot([0, lim], [0, lim], color="#4B5563", lw=0.8, ls=":")
    ax_gap.set_xlim(left=0)
    ax_gap.set_ylim(bottom=0)
    ax_gap.set_xlabel("train NMSE, 2D")
    ax_gap.set_ylabel("validation NMSE, 2D")
    ax_gap.set_title("Train-validation gap")
    ax_gap.grid(axis="y", color="#E5E7EB", lw=0.6)
    panel_label(ax_gap, "c")

    heat = np.full((len(topologies), max(int(r["H"]) for r in rows) + 1), np.nan)
    for i, topology in enumerate(topologies):
        for H in range(heat.shape[1]):
            vals = [
                float(r["val_nmse_2d_mean"])
                for r in sample_rows
                if r["topology"] == topology and int(r["H"]) == H and int(r["D"]) == int(selected["D"])
            ]
            if vals:
                heat[i, H] = float(np.mean(vals))
    im = ax_heat1.imshow(heat, aspect="auto", cmap="viridis")
    ax_heat1.set_yticks(np.arange(len(topologies)))
    ax_heat1.set_yticklabels(topologies, fontsize=6)
    ax_heat1.set_xlabel("H")
    ax_heat1.set_title(f"Per-topology CV NMSE at D={int(selected['D'])}")
    fig.colorbar(im, ax=ax_heat1, fraction=0.046, pad=0.02)
    panel_label(ax_heat1, "d")

    matrix_rows = []
    matrix_labels = []
    for topology in topologies:
        for D in sorted({int(r["D"]) for r in rows}):
            vals = []
            for H in range(max(int(r["H"]) for r in rows) + 1):
                group = [
                    float(r["val_nmse_2d_mean"])
                    for r in sample_rows
                    if r["topology"] == topology and int(r["D"]) == D and int(r["H"]) == H
                ]
                vals.append(float(np.mean(group)) if group else np.nan)
            matrix_rows.append(vals)
            matrix_labels.append(f"{topology}, D={D}")
    im2 = ax_heat2.imshow(np.asarray(matrix_rows), aspect="auto", cmap="viridis")
    ax_heat2.set_yticks(np.arange(len(matrix_labels)))
    ax_heat2.set_yticklabels(matrix_labels, fontsize=5)
    ax_heat2.set_xlabel("H")
    ax_heat2.set_title("Per-topology CV NMSE heatmap over H,D")
    fig.colorbar(im2, ax=ax_heat2, fraction=0.02, pad=0.01)
    panel_label(ax_heat2, "e")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore", message="Singular matrix in solving dual problem.*")
    configure_matplotlib()
    args = parse_args()
    if args.H_max < 0 or args.D_max < 1:
        raise ValueError("Require H_max >= 0 and D_max >= 1.")
    if args.lag_stride_frames < 1:
        raise ValueError("--lag-stride-frames must be at least 1.")

    root = data_root()
    topologies = topology_names(args.topology_start, args.topology_stop, args.topology_name_format)
    samples, missing = discover_sample_dirs(
        root, topologies, args.amplitude, args.sample, args.skip_missing
    )
    if missing:
        print("Warning: missing topology/amplitude directories were skipped:")
        for path in missing:
            print(f"  {path}")

    out_dir = output_dir(args)
    candidates = [(H, D) for D in range(1, args.D_max + 1) for H in range(args.H_max + 1)]
    print(
        f"Evaluating {len(candidates)} nested dictionaries across "
        f"{len(samples)} topology/sample combinations."
    )

    fold_rows = []
    sample_rows = []
    for topology, sample_dir in samples:
        print(f"-> Common dictionary CV: {topology}/{args.amplitude}/{sample_dir.name}")
        for H, D in candidates:
            f_rows, s_row, _ = evaluate_candidate(sample_dir, topology, args, H, D)
            fold_rows.extend(f_rows)
            sample_rows.append(s_row)

    summary = summarize_candidates(sample_rows)
    selected, global_best, tolerance = select_common_cutoff(summary)
    selected_payload, json_path = write_selected_json(
        args, selected, global_best, out_dir, sample_rows, missing
    )

    write_csv(fold_rows, out_dir / "common_dictionary_cutoff_per_fold.csv")
    write_csv(sample_rows, out_dir / "common_dictionary_cutoff_per_sample.csv")
    write_csv(summary, out_dir / "common_dictionary_cutoff_summary.csv")
    plot_summary(summary, sample_rows, selected, out_dir / "common_dictionary_cutoff_summary.pdf")

    print(f"Saved common dictionary selection outputs to: {out_dir}")
    print(
        f"Recommended common dictionary: H_cut={selected_payload['H_cut']}, "
        f"D_cut={selected_payload['D_cut']}."
    )
    print(f"Readout-time delays: {selected_payload['readout_time_delays']}")
    print(f"Target-time delays: {selected_payload['target_time_delays']}")
    print(
        f"Global best CV NMSE_2D={selected_payload['global_best_mean_cv_nmse_2d']:.4f}; "
        f"adequacy={selected_payload['adequacy_label']}."
    )
    print(f"Selected dictionary JSON: {json_path}")
    if selected_payload["adequacy_label"] == "weak":
        print(
            "The selected dictionary is a common compact dictionary for task-weighted IPC. "
            "The scalar input-history basis gives only a weak explanation of the hidden-node "
            "targets, so Q_twIPC should be interpreted as a partial input-driven explanation."
        )


if __name__ == "__main__":
    main()
