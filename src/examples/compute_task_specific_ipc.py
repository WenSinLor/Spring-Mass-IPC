import argparse
from pathlib import Path

import numpy as np

from sklearn.linear_model import Ridge

from task_specific_ipc_common import (
    ALPHA_GRID,
    StateLoader,
    basis_window,
    build_legendre_design,
    data_root,
    discover_sample_dirs,
    frame_counts,
    load_json,
    load_positions,
    make_blocked_folds,
    nmse_components,
    normalize_to_unit_interval,
    scalar_r2,
    select_alpha_by_blocked_cv,
    target_time_delays,
    term_metadata,
    topology_names,
    valid_basis_rows,
    visible_state_design,
    write_csv,
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
DEFAULT_CV_BLOCKS = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute task-specific IPC from a frozen common input-history dictionary."
    )
    parser.add_argument("--selected-dictionary", required=True)
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
    parser.add_argument("--cv-blocks", type=int, default=DEFAULT_CV_BLOCKS)
    parser.add_argument(
        "--skip-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip topology/amplitude folders that are not present instead of failing.",
    )
    parser.add_argument(
        "--global-standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use lab convention: z-score full visible-state matrix before adding bias.",
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
        / "task_specific_ipc"
    )


def fit_task_coefficients(P, Y, cv_blocks):
    selected_alpha, _ = select_alpha_by_blocked_cv(P, Y, ALPHA_GRID, cv_blocks)
    model = Ridge(alpha=selected_alpha, fit_intercept=True)
    model.fit(P, Y)
    pred = model.predict(P)
    nmse_x, nmse_y, nmse_2d = nmse_components(Y, pred)
    return model, selected_alpha, nmse_x, nmse_y, nmse_2d


def task_weight_rows(topology, amplitude, sample, args, dictionary, metadata, P, Y, model):
    centered = Y - np.mean(Y, axis=0, keepdims=True)
    denom_2d = float(np.sum(centered**2))
    denom_x = float(np.sum(centered[:, 0] ** 2))
    denom_y = float(np.sum(centered[:, 1] ** 2))
    rows = []
    for meta in metadata:
        idx = int(meta["basis_index"])
        p = P[:, idx]
        p_norm_sq = float(np.sum(p**2))
        coef_x = float(model.coef_[0, idx])
        coef_y = float(model.coef_[1, idx])
        coeff_mag_sq = coef_x**2 + coef_y**2
        c_alpha = coeff_mag_sq * p_norm_sq / max(denom_2d, np.finfo(float).eps)
        row = {
            "topology": topology,
            "amplitude": amplitude,
            "sample": sample,
            "hidden_node": args.hidden_node,
            "reference_node": args.reference_node,
            "horizon_steps": args.horizon_steps,
            "H_cut": dictionary["H_cut"],
            "D_cut": dictionary["D_cut"],
            **meta,
            "coefficient_x": coef_x,
            "coefficient_y": coef_y,
            "p_alpha_norm_sq": p_norm_sq,
            "c_alpha_x": coef_x**2 * p_norm_sq / max(denom_x, np.finfo(float).eps),
            "c_alpha_y": coef_y**2 * p_norm_sq / max(denom_y, np.finfo(float).eps),
            "c_alpha": c_alpha,
        }
        rows.append(row)
    return rows


def coefficient_rows(topology, amplitude, sample, args, dictionary, metadata, model):
    rows = [
        {
            "topology": topology,
            "amplitude": amplitude,
            "sample": sample,
            "hidden_node": args.hidden_node,
            "reference_node": args.reference_node,
            "horizon_steps": args.horizon_steps,
            "H_cut": dictionary["H_cut"],
            "D_cut": dictionary["D_cut"],
            "basis_index": -1,
            "basis_term": "intercept",
            "total_degree": 0,
            "max_readout_time_delay": "",
            "max_target_time_delay": "",
            "max_lag_frames": "",
            "coefficient_x": float(model.intercept_[0]),
            "coefficient_y": float(model.intercept_[1]),
        }
    ]
    for meta in metadata:
        idx = int(meta["basis_index"])
        rows.append(
            {
                "topology": topology,
                "amplitude": amplitude,
                "sample": sample,
                "hidden_node": args.hidden_node,
                "reference_node": args.reference_node,
                "horizon_steps": args.horizon_steps,
                "H_cut": dictionary["H_cut"],
                "D_cut": dictionary["D_cut"],
                **meta,
                "coefficient_x": float(model.coef_[0, idx]),
                "coefficient_y": float(model.coef_[1, idx]),
            }
        )
    return rows


def compute_ipc_rows(sample_dir, topology, args, dictionary, metadata, P_full):
    loader, X_design, state_nodes, num_state_features = visible_state_design(
        sample_dir, args.hidden_node, args.reference_node, args.global_standardize
    )
    H = int(dictionary["H_cut"])
    _, _, _, train_start, train_stop, test_stop = frame_counts(
        loader, args.washout, args.train, args.test
    )
    valid_train_start = max(train_start, H * int(dictionary["lag_stride_frames"]))
    if valid_train_start >= train_stop:
        raise ValueError(f"No valid training rows for IPC in {topology}/{sample_dir.name}.")
    train_rows = np.arange(valid_train_start, train_stop)
    test_rows = np.arange(max(train_stop, H * int(dictionary["lag_stride_frames"])), test_stop)
    rows = []
    for meta in metadata:
        idx = int(meta["basis_index"])
        y_train = P_full[train_rows, idx]
        y_test = P_full[test_rows, idx]
        model = Ridge(alpha=1e-6, fit_intercept=False)
        model.fit(X_design[train_rows], y_train)
        train_r2, train_nmse = scalar_r2(y_train, model.predict(X_design[train_rows]))
        test_r2, test_nmse = scalar_r2(y_test, model.predict(X_design[test_rows]))
        rows.append(
            {
                "topology": topology,
                "amplitude": args.amplitude,
                "sample": sample_dir.name,
                "hidden_node": args.hidden_node,
                "reference_node": args.reference_node,
                "horizon_steps": args.horizon_steps,
                "H_cut": dictionary["H_cut"],
                "D_cut": dictionary["D_cut"],
                **meta,
                "num_state_nodes": len(state_nodes),
                "num_state_features": int(num_state_features),
                "train_r2_raw": train_r2,
                "test_r2_raw": test_r2,
                "train_nmse": train_nmse,
                "test_nmse": test_nmse,
                "IPC_alpha_raw": test_r2,
                "IPC_alpha_clipped": max(test_r2, 0.0),
            }
        )
    return rows


def grouped_contributions(task_weight_rows_sample, ipc_rows_sample, basis_nmse_2d, dictionary_adequacy):
    ipc_by_index = {int(r["basis_index"]): r for r in ipc_rows_sample}
    groups = {}
    for w in task_weight_rows_sample:
        idx = int(w["basis_index"])
        ipc = ipc_by_index[idx]
        for group_type, group_value in (
            ("degree", int(w["total_degree"])),
            ("readout_time_delay", int(w["max_readout_time_delay"])),
            ("target_time_delay", int(w["max_target_time_delay"])),
        ):
            key = (
                w["topology"],
                w["amplitude"],
                w["sample"],
                int(w["hidden_node"]),
                int(w["reference_node"]),
                int(w["horizon_steps"]),
                group_type,
                group_value,
            )
            if key not in groups:
                groups[key] = {
                    "topology": w["topology"],
                    "amplitude": w["amplitude"],
                    "sample": w["sample"],
                    "hidden_node": int(w["hidden_node"]),
                    "reference_node": int(w["reference_node"]),
                    "horizon_steps": int(w["horizon_steps"]),
                    "group_type": group_type,
                    "group_value": group_value,
                    "c_alpha_sum": 0.0,
                    "IPC_alpha_clipped_sum": 0.0,
                    "weighted_contribution_sum": 0.0,
                    "num_terms": 0,
                    "basis_nmse_2d": basis_nmse_2d,
                    "basis_adequacy_label": dictionary_adequacy,
                }
            groups[key]["c_alpha_sum"] += float(w["c_alpha"])
            groups[key]["IPC_alpha_clipped_sum"] += float(ipc["IPC_alpha_clipped"])
            groups[key]["weighted_contribution_sum"] += float(w["c_alpha"]) * float(
                ipc["IPC_alpha_clipped"]
            )
            groups[key]["num_terms"] += 1
    return list(groups.values())


def evaluate_sample(sample_dir, topology, args, dictionary):
    loader = StateLoader(sample_dir / "experiment.h5")
    _, positions = load_positions(sample_dir / "experiment.h5")
    u_norm = normalize_to_unit_interval(loader.get_actuation_signal(actuator_idx=0, dof=0))
    hidden_relative = positions[:, args.hidden_node, :] - positions[:, args.reference_node, :]

    H = int(dictionary["H_cut"])
    D = int(dictionary["D_cut"])
    lag_stride = int(dictionary["lag_stride_frames"])
    P_full, exps = build_legendre_design(u_norm, H, D, lag_stride)
    metadata = term_metadata(exps, args.horizon_steps, lag_stride)

    window_start, window_stop = basis_window(
        loader, args.washout, args.train, args.test, args.horizon_steps
    )
    basis_rows = valid_basis_rows(window_start, window_stop, H, lag_stride)
    P = P_full[basis_rows]
    Y = hidden_relative[basis_rows + args.horizon_steps]
    task_model, selected_alpha, basis_nmse_x, basis_nmse_y, basis_nmse_2d = fit_task_coefficients(
        P, Y, args.cv_blocks
    )

    coeff_rows = coefficient_rows(
        topology, args.amplitude, sample_dir.name, args, dictionary, metadata, task_model
    )
    weights = task_weight_rows(
        topology, args.amplitude, sample_dir.name, args, dictionary, metadata, P, Y, task_model
    )
    ipc = compute_ipc_rows(sample_dir, topology, args, dictionary, metadata, P_full)
    ipc_by_index = {int(r["basis_index"]): r for r in ipc}

    term_rows = []
    q_raw = 0.0
    q_clipped = 0.0
    total_ipc_raw = 0.0
    total_ipc_clipped = 0.0
    for w in weights:
        ipc_row = ipc_by_index[int(w["basis_index"])]
        contribution_raw = float(w["c_alpha"]) * float(ipc_row["IPC_alpha_raw"])
        contribution_clipped = float(w["c_alpha"]) * float(ipc_row["IPC_alpha_clipped"])
        q_raw += contribution_raw
        q_clipped += contribution_clipped
        total_ipc_raw += float(ipc_row["IPC_alpha_raw"])
        total_ipc_clipped += float(ipc_row["IPC_alpha_clipped"])
        term_rows.append(
            {
                **w,
                "IPC_alpha_raw": ipc_row["IPC_alpha_raw"],
                "IPC_alpha_clipped": ipc_row["IPC_alpha_clipped"],
                "weighted_contribution_raw": contribution_raw,
                "weighted_contribution_clipped": contribution_clipped,
                "basis_nmse_x": basis_nmse_x,
                "basis_nmse_y": basis_nmse_y,
                "basis_nmse_2d": basis_nmse_2d,
                "task_alpha_selected": selected_alpha,
                "basis_adequacy_label": dictionary.get("adequacy_label", ""),
            }
        )

    score_row = {
        "topology": topology,
        "amplitude": args.amplitude,
        "sample": sample_dir.name,
        "hidden_node": args.hidden_node,
        "reference_node": args.reference_node,
        "horizon_steps": args.horizon_steps,
        "H_cut": H,
        "D_cut": D,
        "lag_stride_frames": lag_stride,
        "readout_time_delays": str(list(range(H + 1))),
        "target_time_delays": str(target_time_delays(H, args.horizon_steps, lag_stride)),
        "number_of_terms": len(metadata),
        "task_alpha_selected": selected_alpha,
        "basis_nmse_x": basis_nmse_x,
        "basis_nmse_y": basis_nmse_y,
        "basis_nmse_2d": basis_nmse_2d,
        "basis_adequacy_label": dictionary.get("adequacy_label", ""),
        "Q_twIPC_raw": q_raw,
        "Q_twIPC_clipped": q_clipped,
        "generic_total_ipc_raw": total_ipc_raw,
        "generic_total_ipc_clipped": total_ipc_clipped,
    }
    grouped = grouped_contributions(
        weights, ipc, basis_nmse_2d, dictionary.get("adequacy_label", "")
    )
    return coeff_rows, weights, ipc, term_rows, grouped, score_row


def summarize_scores(score_rows):
    summary = []
    for topology in sorted({r["topology"] for r in score_rows}):
        group = [r for r in score_rows if r["topology"] == topology]
        row = {"topology": topology, "num_samples": len(group)}
        for key in (
            "basis_nmse_2d",
            "Q_twIPC_raw",
            "Q_twIPC_clipped",
            "generic_total_ipc_clipped",
        ):
            values = np.asarray([r[key] for r in group], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(values))
            row[f"{key}_std"] = float(np.nanstd(values, ddof=0))
        summary.append(row)
    return summary


def write_readme(out_dir):
    text = """# Task-Specific IPC Pipeline

This output uses one frozen input-history Legendre dictionary selected across a
topology family. The hidden-node target changes with topology, so the common
dictionary keeps the task-weighted IPC scores comparable.

H_cut and D_cut are the truncation of the input-history polynomial dictionary.
They are not a direct measurement of the exact memory depth or nonlinearity of
any individual topology.

c_alpha measures how strongly the topology-conditioned hidden-node target uses
basis term alpha. IPC_alpha measures how well the visible reservoir state
linearly reconstructs that same basis term. The task-specific score is the
demand-supply match:

    Q_twIPC = sum_alpha c_alpha * IPC_alpha

If the basis adequacy label is weak, Q_twIPC is only a partial input-driven
explanation of hidden-node prediction performance.
"""
    (out_dir / "README_task_specific_ipc_pipeline.txt").write_text(text)


def main():
    args = parse_args()
    dictionary = load_json(args.selected_dictionary)
    if args.horizon_steps != int(dictionary["horizon_steps"]):
        raise ValueError("The requested --horizon-steps does not match the selected dictionary.")
    if args.hidden_node != int(dictionary["hidden_node"]):
        raise ValueError("The requested --hidden-node does not match the selected dictionary.")
    if args.reference_node != int(dictionary["reference_node"]):
        raise ValueError("The requested --reference-node does not match the selected dictionary.")

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
    coefficient_rows_all = []
    task_weight_rows_all = []
    ipc_rows_all = []
    term_rows_all = []
    grouped_rows_all = []
    score_rows = []

    print(
        f"Frozen dictionary: H_cut={dictionary['H_cut']}, D_cut={dictionary['D_cut']}, "
        f"target-time delays={dictionary['target_time_delays']}"
    )
    for topology, sample_dir in samples:
        print(f"-> Task-specific IPC: {topology}/{args.amplitude}/{sample_dir.name}")
        coeff_rows, weights, ipc, terms, grouped, score = evaluate_sample(
            sample_dir, topology, args, dictionary
        )
        coefficient_rows_all.extend(coeff_rows)
        task_weight_rows_all.extend(weights)
        ipc_rows_all.extend(ipc)
        term_rows_all.extend(terms)
        grouped_rows_all.extend(grouped)
        score_rows.append(score)
        coeff_path = out_dir / f"task_coefficients_{topology}_{sample_dir.name}.csv"
        write_csv(coeff_rows, coeff_path)

    summary = summarize_scores(score_rows)
    write_csv(coefficient_rows_all, out_dir / "task_coefficients_all_samples.csv")
    write_csv(task_weight_rows_all, out_dir / "task_weights_per_sample.csv")
    write_csv(ipc_rows_all, out_dir / "ipc_per_term_per_sample.csv")
    write_csv(term_rows_all, out_dir / "task_specific_ipc_per_term_per_sample.csv")
    write_csv(grouped_rows_all, out_dir / "task_specific_ipc_grouped_contributions.csv")
    write_csv(score_rows, out_dir / "task_specific_ipc_per_sample.csv")
    write_csv(summary, out_dir / "task_specific_ipc_summary.csv")
    write_readme(out_dir)

    print(f"Saved task-specific IPC outputs to: {out_dir}")
    for row in summary:
        print(
            f"  {row['topology']}: Q_twIPC_clipped="
            f"{row['Q_twIPC_clipped_mean']:.4f} +/- {row['Q_twIPC_clipped_std']:.4f}; "
            f"basis NMSE={row['basis_nmse_2d_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
