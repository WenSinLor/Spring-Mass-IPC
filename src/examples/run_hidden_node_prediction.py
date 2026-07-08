import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from openprc.reservoir.features.node_features import NodeDisplacements
from openprc.reservoir.io.state_loader import StateLoader


DEFAULT_TOPOLOGY = "topology_17_prestress"
DEFAULT_AMPLITUDE = "amp=1"
DEFAULT_HIDDEN_NODE = 10
DEFAULT_REFERENCE_NODE = 0
DEFAULT_WASHOUT_S = 5.0
DEFAULT_TRAIN_S = 10.0
DEFAULT_TEST_S = 10.0
DEFAULT_RIDGE = 1e-6
DEFAULT_HORIZON_STEPS = 5


PALETTE = {
    "actual": "#1f2937",
    "pred": "#0072B2",
    "residual": "#D55E00",
    "train": "#E8EEF7",
    "test": "#F7EDE2",
    "reference": "#CC79A7",
    "hidden": "#D55E00",
    "state": "#8C8C8C",
    "x": "#0072B2",
    "y": "#009E73",
}


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a linear readout to predict one hidden marker trajectory "
            "from the remaining marker displacements relative to node 0."
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
    parser.add_argument("--washout", type=float, default=DEFAULT_WASHOUT_S)
    parser.add_argument("--train", type=float, default=DEFAULT_TRAIN_S)
    parser.add_argument("--test", type=float, default=DEFAULT_TEST_S)
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=DEFAULT_HORIZON_STEPS,
        help=(
            "Prediction horizon in frames. A value of 5 trains X[t] to predict "
            "the hidden node at t+5."
        ),
    )
    parser.add_argument(
        "--save-readout",
        action="store_true",
        help="Save fitted readout weights and preprocessing parameters for each sample.",
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


def frame_windows(loader: StateLoader, washout_s: float, train_s: float, test_s: float):
    washout_frames = int(washout_s / loader.dt)
    train_frames = int(train_s / loader.dt)
    test_frames = int(test_s / loader.dt)
    train_start = washout_frames
    train_stop = train_start + train_frames
    test_stop = train_stop + test_frames

    if test_stop > loader.total_frames:
        raise ValueError(
            f"Need {test_stop} frames for washout/train/test windows, "
            f"but only {loader.total_frames} frames are available."
        )

    return train_start, train_stop, test_stop


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, np.finfo(float).eps)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2, axis=0))


def nmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    variance = np.var(y_true, axis=0)
    return mse / np.maximum(variance, np.finfo(float).eps)


def nmse_2d(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    err = y_true - y_pred
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    return float(
        np.sum(err**2) / np.maximum(np.sum(centered**2), np.finfo(float).eps)
    )


def load_positions(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        positions = f["time_series/nodes/positions"][:, :, :2]
        time = f["time_series/time"][:]
    return time, positions


def train_hidden_node_readout(
    sample_dir: Path,
    hidden_node: int,
    reference_node: int,
    washout_s: float,
    train_s: float,
    test_s: float,
    ridge_alpha: float,
    horizon_steps: int,
):
    h5_path = sample_dir / "experiment.h5"
    loader = StateLoader(h5_path)
    time, positions = load_positions(h5_path)
    n_nodes = positions.shape[1]

    if hidden_node < 0 or hidden_node >= n_nodes:
        raise ValueError(f"Hidden node {hidden_node} is outside available node range 0..{n_nodes - 1}.")
    if reference_node < 0 or reference_node >= n_nodes:
        raise ValueError(
            f"Reference node {reference_node} is outside available node range 0..{n_nodes - 1}."
        )
    if hidden_node == reference_node:
        raise ValueError("The hidden node cannot also be the reference node.")
    if horizon_steps < 0:
        raise ValueError("Prediction horizon must be non-negative.")

    state_nodes = [node for node in range(n_nodes) if node not in {reference_node, hidden_node}]
    features = NodeDisplacements(reference_node=reference_node, node_ids=state_nodes, dims=[0, 1])
    X_full = features.transform(loader)

    reference_xy = positions[:, reference_node, :]
    hidden_relative_xy = positions[:, hidden_node, :] - reference_xy

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_full)
    design = np.hstack([np.ones((X_std.shape[0], 1)), X_std])

    if horizon_steps > 0:
        design = design[:-horizon_steps]
        y_full = hidden_relative_xy[horizon_steps:]
        target_time = time[horizon_steps:]
    else:
        y_full = hidden_relative_xy
        target_time = time

    train_start, train_stop, test_stop = frame_windows(loader, washout_s, train_s, test_s)
    if test_stop > len(y_full):
        raise ValueError(
            f"Need target frame {test_stop + horizon_steps} for a {horizon_steps}-step "
            f"forecast, but only {loader.total_frames} frames are available."
        )
    train_slice = slice(train_start, train_stop)
    test_slice = slice(train_stop, test_stop)

    model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    model.fit(design[train_slice], y_full[train_slice])
    y_pred = model.predict(design)

    metrics = {
        "topology": sample_dir.parent.parent.name,
        "amplitude": sample_dir.parent.name,
        "sample": sample_dir.name,
        "hidden_node": hidden_node,
        "reference_node": reference_node,
        "horizon_steps": horizon_steps,
        "horizon_s": float(horizon_steps * loader.dt),
        "num_state_nodes": len(state_nodes),
        "num_state_features": X_full.shape[1],
        "train_r2_x": float(r2_score(y_full[train_slice], y_pred[train_slice])[0]),
        "train_r2_y": float(r2_score(y_full[train_slice], y_pred[train_slice])[1]),
        "test_r2_x": float(r2_score(y_full[test_slice], y_pred[test_slice])[0]),
        "test_r2_y": float(r2_score(y_full[test_slice], y_pred[test_slice])[1]),
        "train_nmse_x": float(nmse(y_full[train_slice], y_pred[train_slice])[0]),
        "train_nmse_y": float(nmse(y_full[train_slice], y_pred[train_slice])[1]),
        "test_nmse_x": float(nmse(y_full[test_slice], y_pred[test_slice])[0]),
        "test_nmse_y": float(nmse(y_full[test_slice], y_pred[test_slice])[1]),
        "train_nmse_2d": nmse_2d(y_full[train_slice], y_pred[train_slice]),
        "test_nmse_2d": nmse_2d(y_full[test_slice], y_pred[test_slice]),
        "train_rmse_x_px": float(rmse(y_full[train_slice], y_pred[train_slice])[0]),
        "train_rmse_y_px": float(rmse(y_full[train_slice], y_pred[train_slice])[1]),
        "test_rmse_x_px": float(rmse(y_full[test_slice], y_pred[test_slice])[0]),
        "test_rmse_y_px": float(rmse(y_full[test_slice], y_pred[test_slice])[1]),
    }

    return {
        "sample": sample_dir.name,
        "sample_dir": sample_dir,
        "loader": loader,
        "time": target_time,
        "state_time": time[: len(target_time)],
        "positions": positions,
        "state_nodes": state_nodes,
        "feature_info": features.get_feature_info(loader),
        "target": y_full,
        "prediction": y_pred,
        "residual": y_full - y_pred,
        "train_start": train_start,
        "train_stop": train_stop,
        "test_stop": test_stop,
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
    }


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


def save_pub_figure(fig, out_stem: Path):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")


def plot_sample_result(result, hidden_node, reference_node, out_dir: Path):
    time = result["time"]
    positions = result["positions"]
    y = result["target"]
    yhat = result["prediction"]
    train_start = result["train_start"]
    train_stop = result["train_stop"]
    test_stop = result["test_stop"]
    metrics = result["metrics"]

    t_rel = time - time[train_start]
    train_t = t_rel[train_start:train_stop]
    test_t = t_rel[train_stop:test_stop]

    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.9, 0.9], width_ratios=[1.0, 1.25])
    ax_layout = fig.add_subplot(gs[0, 0])
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_x = fig.add_subplot(gs[1, :])
    ax_y = fig.add_subplot(gs[2, :], sharex=ax_x)

    mean_xy = np.mean(positions[train_start:train_stop], axis=0)
    ax_layout.scatter(
        mean_xy[result["state_nodes"], 0],
        mean_xy[result["state_nodes"], 1],
        s=24,
        color=PALETTE["state"],
        alpha=0.75,
        linewidth=0,
        label="readout nodes",
    )
    ax_layout.scatter(
        mean_xy[reference_node, 0],
        mean_xy[reference_node, 1],
        s=44,
        color=PALETTE["reference"],
        edgecolor="white",
        linewidth=0.6,
        label=f"reference {reference_node}",
        zorder=3,
    )
    ax_layout.scatter(
        mean_xy[hidden_node, 0],
        mean_xy[hidden_node, 1],
        s=52,
        color=PALETTE["hidden"],
        edgecolor="white",
        linewidth=0.6,
        label=f"hidden {hidden_node}",
        zorder=4,
    )
    ax_layout.text(mean_xy[reference_node, 0], mean_xy[reference_node, 1], "0", ha="center", va="center", color="white", fontsize=6)
    ax_layout.text(mean_xy[hidden_node, 0], mean_xy[hidden_node, 1], str(hidden_node), ha="center", va="center", color="white", fontsize=6)
    ax_layout.set_aspect("equal", adjustable="box")
    ax_layout.invert_yaxis()
    ax_layout.set_xlabel("camera x (px)")
    ax_layout.set_ylabel("camera y (px)")
    ax_layout.set_title("Readout setup")
    ax_layout.legend(loc="lower left", fontsize=6, handlelength=1.0)
    panel_label(ax_layout, "a")

    ax_xy.plot(y[train_stop:test_stop, 0], y[train_stop:test_stop, 1], color=PALETTE["actual"], lw=1.4, label="measured")
    ax_xy.plot(yhat[train_stop:test_stop, 0], yhat[train_stop:test_stop, 1], color=PALETTE["pred"], lw=1.2, ls="--", label="predicted")
    ax_xy.set_xlabel("relative x (px)")
    ax_xy.set_ylabel("relative y (px)")
    ax_xy.set_title("Test-window trajectory")
    ax_xy.legend(loc="best", fontsize=6)
    panel_label(ax_xy, "b")

    for ax, dim, dim_name, dim_color in [
        (ax_x, 0, "x", PALETTE["x"]),
        (ax_y, 1, "y", PALETTE["y"]),
    ]:
        ax.axvspan(t_rel[train_start], t_rel[train_stop - 1], color=PALETTE["train"], zorder=0)
        ax.axvspan(t_rel[train_stop], t_rel[test_stop - 1], color=PALETTE["test"], zorder=0)
        ax.plot(train_t, y[train_start:train_stop, dim], color=PALETTE["actual"], lw=1.0)
        ax.plot(test_t, y[train_stop:test_stop, dim], color=PALETTE["actual"], lw=1.15, label="measured")
        ax.plot(test_t, yhat[train_stop:test_stop, dim], color=dim_color, lw=1.15, ls="--", label="predicted")
        ax.axvline(t_rel[train_stop], color="#4B5563", lw=0.8, ls=":")
        ax.set_ylabel(f"relative {dim_name} (px)")
        ax.grid(axis="y", color="#E5E7EB", lw=0.6)
        ax.text(
            0.03,
            0.94,
            f"test NMSE = {metrics[f'test_nmse_{dim_name}']:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color=dim_color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
        )

    ax_x.set_title(
        f"{result['sample']} hidden node {hidden_node}: {metrics['horizon_steps']}-step forecast, "
        f"test R2 x={metrics['test_r2_x']:.2f}, y={metrics['test_r2_y']:.2f}"
    )
    ax_x.legend(loc="upper right", ncol=2, fontsize=6)
    panel_label(ax_x, "c")
    ax_y.set_xlabel("time from train start (s)")
    ax_x.set_xlabel("time from train start (s)")
    panel_label(ax_y, "d")

    out_stem = (
        out_dir
        / f"hidden_node_{hidden_node}_h{metrics['horizon_steps']}_prediction_{result['sample']}"
    )
    save_pub_figure(fig, out_stem)
    plt.close(fig)
    return out_stem


def plot_summary(results, hidden_node, out_dir: Path):
    samples = [r["sample"] for r in results]
    test_r2_x = np.array([r["metrics"]["test_r2_x"] for r in results])
    test_r2_y = np.array([r["metrics"]["test_r2_y"] for r in results])
    test_nmse_x = np.array([r["metrics"]["test_nmse_x"] for r in results])
    test_nmse_y = np.array([r["metrics"]["test_nmse_y"] for r in results])

    fig = plt.figure(figsize=(7.2, 3.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.35])
    ax_r2 = fig.add_subplot(gs[0, 0])
    ax_rmse = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[0, 2])

    x = np.arange(len(samples))
    width = 0.36
    ax_r2.bar(x - width / 2, test_r2_x, width, color=PALETTE["x"], label="x")
    ax_r2.bar(x + width / 2, test_r2_y, width, color=PALETTE["y"], label="y")
    ax_r2.axhline(0, color="#4B5563", lw=0.8)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(samples, rotation=35, ha="right")
    ax_r2.set_ylabel("test R2")
    ax_r2.set_title("Prediction accuracy")
    ax_r2.legend(loc="best", fontsize=6)
    panel_label(ax_r2, "a")

    ax_rmse.plot(x, test_nmse_x, marker="o", color=PALETTE["x"], lw=1.2, label="x")
    ax_rmse.plot(x, test_nmse_y, marker="o", color=PALETTE["y"], lw=1.2, label="y")
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(samples, rotation=35, ha="right")
    ax_rmse.set_ylabel("test NMSE")
    ax_rmse.set_title("Normalized prediction error")
    ax_rmse.grid(axis="y", color="#E5E7EB", lw=0.6)
    ax_rmse.legend(loc="best", fontsize=6)
    panel_label(ax_rmse, "b")

    worst_idx = int(np.argmin(np.minimum(test_r2_x, test_r2_y)))
    worst = results[worst_idx]
    train_stop = worst["train_stop"]
    test_stop = worst["test_stop"]
    t = worst["time"] - worst["time"][train_stop]
    ax_trace.plot(
        t[train_stop:test_stop],
        worst["target"][train_stop:test_stop, 0],
        color=PALETTE["actual"],
        lw=1.15,
        label="measured x",
    )
    ax_trace.plot(
        t[train_stop:test_stop],
        worst["prediction"][train_stop:test_stop, 0],
        color=PALETTE["x"],
        lw=1.15,
        ls="--",
        label="predicted x",
    )
    ax_trace.set_xlabel("test time (s)")
    ax_trace.set_ylabel("relative x (px)")
    ax_trace.set_title(f"Worst sample example: {worst['sample']}")
    ax_trace.grid(axis="y", color="#E5E7EB", lw=0.6)
    ax_trace.legend(loc="best", fontsize=6)
    panel_label(ax_trace, "c")

    horizon_steps = results[0]["metrics"]["horizon_steps"]
    out_stem = out_dir / f"hidden_node_{hidden_node}_h{horizon_steps}_prediction_summary"
    save_pub_figure(fig, out_stem)
    plt.close(fig)
    return out_stem


def save_metrics_csv(results, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0]["metrics"].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result["metrics"])


def save_prediction_csv(result, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "target_x_px",
                "target_y_px",
                "pred_x_px",
                "pred_y_px",
                "residual_x_px",
                "residual_y_px",
                "split",
            ]
        )
        train_start = result["train_start"]
        train_stop = result["train_stop"]
        test_stop = result["test_stop"]
        for idx in range(train_start, test_stop):
            split = "train" if idx < train_stop else "test"
            writer.writerow(
                [
                    result["time"][idx],
                    result["target"][idx, 0],
                    result["target"][idx, 1],
                    result["prediction"][idx, 0],
                    result["prediction"][idx, 1],
                    result["residual"][idx, 0],
                    result["residual"][idx, 1],
                    split,
                ]
            )


def save_readout_h5(result, hidden_node, reference_node, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_map = np.array(
        [[fi["node_id"], fi["dim"]] for fi in result["feature_info"]],
        dtype=np.int32,
    )
    with h5py.File(out_path, "w") as f:
        f.attrs["task"] = "hidden_node_prediction"
        f.attrs["sample"] = result["sample"]
        f.attrs["hidden_node"] = hidden_node
        f.attrs["reference_node"] = reference_node
        f.create_dataset("model/weights", data=result["model"].coef_)
        f.create_dataset("preprocessing/X_mean", data=result["scaler"].mean_)
        f.create_dataset("preprocessing/X_scale", data=result["scaler"].scale_)
        f.create_dataset("feature_map/node_dim", data=feature_map)


def main():
    configure_matplotlib()
    args = parse_args()

    data_root = src_dir.parent / "data" / "experiment_data"
    base_dir = data_root / args.topology / args.amplitude
    sample_dirs = find_sample_dirs(base_dir, args.sample)
    out_dir = (
        base_dir
        / "plots"
        / f"hidden_node_{args.hidden_node}_prediction"
        / f"horizon_{args.horizon_steps}_steps"
    )

    print(
        "Hidden-node readout setup: "
        f"target=node {args.hidden_node}, reference=node {args.reference_node}, "
        "state=all remaining node displacements, "
        f"horizon={args.horizon_steps} frame(s)."
    )

    results = []
    for sample_dir in sample_dirs:
        print(f"-> Training {sample_dir.name}")
        result = train_hidden_node_readout(
            sample_dir=sample_dir,
            hidden_node=args.hidden_node,
            reference_node=args.reference_node,
            washout_s=args.washout,
            train_s=args.train,
            test_s=args.test,
            ridge_alpha=args.ridge,
            horizon_steps=args.horizon_steps,
        )
        results.append(result)

        fig_stem = plot_sample_result(result, args.hidden_node, args.reference_node, out_dir)
        pred_csv = (
            out_dir
            / f"hidden_node_{args.hidden_node}_h{args.horizon_steps}_prediction_{result['sample']}.csv"
        )
        save_prediction_csv(result, pred_csv)

        if args.save_readout:
            readout_path = (
                out_dir
                / f"hidden_node_{args.hidden_node}_h{args.horizon_steps}_readout_{result['sample']}.h5"
            )
            save_readout_h5(result, args.hidden_node, args.reference_node, readout_path)
            print(f"   saved readout: {readout_path}")

        m = result["metrics"]
        print(
            f"   test NMSE: x={m['test_nmse_x']:.4f}, y={m['test_nmse_y']:.4f}; "
            f"test R2: x={m['test_r2_x']:.3f}, y={m['test_r2_y']:.3f}; "
            f"test RMSE: x={m['test_rmse_x_px']:.3f}px, y={m['test_rmse_y_px']:.3f}px"
        )
        print(f"   saved figure: {fig_stem}.pdf")

    metrics_csv = out_dir / f"hidden_node_{args.hidden_node}_h{args.horizon_steps}_prediction_metrics.csv"
    save_metrics_csv(results, metrics_csv)
    print(f"Saved metrics: {metrics_csv}")

    if len(results) > 1:
        summary_stem = plot_summary(results, args.hidden_node, out_dir)
        print(f"Saved summary figure: {summary_stem}.pdf")


if __name__ == "__main__":
    main()
