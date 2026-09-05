"""Create FPS-matched experiment files for topology 18 without overwriting data.

This script converts the 119.88-fps topology-18 trajectories to 29.97 fps,
the rate used by topology 10.  It writes the result under
``data/experiment_data/topology_18_prestress_30fps`` so that the existing
topology-18 experiment files and heatmaps remain unchanged.

The current actuation definition is deliberately preserved: marker 0's
x-displacement.  This makes the output useful for an FPS-only comparison.
It is not yet a replacement for a benchmark using the external actuator or
vibrometer signal as the input.
"""

from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample_poly


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "experiment_data"
SOURCE_TRACKING_H5 = (
    DATA_ROOT / "topology_18_prestress" / "amp=1" / "tracking_data.h5"
)
OUTPUT_TOPOLOGY = "topology_18_prestress_30fps"
OUTPUT_AMPLITUDE = "amp=1"

TARGET_FPS = 30000.0 / 1001.0  # 29.97002997 Hz; topology-10 recording rate
DURATION_S = 30.0
NUM_SAMPLES = 5
TRIGGER_FRACTION = 0.20
PRE_TRIGGER_S = 0.20
GAP_AFTER_SAMPLE_S = 2.0

BAR_INDICES = np.array(
    [
        # Horizontal
        [0, 1], [1, 2], [2, 3],
        [4, 5], [5, 6], [6, 7],
        [8, 9], [9, 10], [10, 11],
        [12, 13], [13, 14], [14, 15],
        # Vertical
        [0, 4], [4, 8], [8, 12],
        [1, 5], [5, 9], [9, 13],
        [2, 6], [6, 10], [10, 14],
        [3, 7], [7, 11], [11, 15],
    ],
    dtype=np.intp,
)


def _load_tracking_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Tracking data not found: {path}")

    with h5py.File(path, "r") as h5:
        trajectories = h5["trajectories"][:]
        time = h5["time"][:]
        metadata = dict(h5.attrs)

    if trajectories.ndim != 3 or trajectories.shape[1] != 16:
        raise ValueError(
            "Expected trajectories with shape (frames, 16, coordinates); "
            f"received {trajectories.shape}."
        )
    if len(time) != len(trajectories):
        raise ValueError("The trajectory and time arrays have different lengths.")
    if len(time) < 2:
        raise ValueError("At least two frames are required to determine FPS.")

    source_fps = float(metadata.get("fps", 1.0 / np.median(np.diff(time))))
    return trajectories, time, metadata, source_fps


def _integer_decimation(source_fps: float) -> int:
    decimation = int(round(source_fps / TARGET_FPS))
    if decimation < 1 or not np.isclose(
        source_fps / TARGET_FPS, decimation, rtol=1e-3, atol=1e-3
    ):
        raise ValueError(
            f"Cannot safely use integer decimation: source={source_fps:.8f} Hz, "
            f"target={TARGET_FPS:.8f} Hz."
        )
    return decimation


def _write_experiment(
    output_path: Path,
    time: np.ndarray,
    positions: np.ndarray,
    actuation: np.ndarray,
    metadata: dict,
    output_fps: float,
    sample_index: int,
    source_fps: float,
    decimation: int,
) -> None:
    p0 = positions[:, BAR_INDICES[:, 0], :]
    p1 = positions[:, BAR_INDICES[:, 1], :]
    bar_lengths = np.sqrt(np.sum((p1 - p0) ** 2, axis=2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        ts = h5.create_group("time_series")
        ts.create_dataset("time", data=time - time[0])
        ts.create_group("nodes").create_dataset("positions", data=positions)
        ts.create_group("elements/bars").create_dataset("lengths", data=bar_lengths)
        ts.create_group("actuation_signals").create_dataset("0", data=actuation)

        for key, value in metadata.items():
            h5.attrs[key] = value
        h5.attrs["fps"] = output_fps
        h5.attrs["slice_index"] = sample_index
        h5.attrs["amplitude_group"] = OUTPUT_AMPLITUDE
        h5.attrs["resampled_from_fps"] = source_fps
        h5.attrs["resample_decimation"] = decimation
        h5.attrs["actuation_definition"] = "negative marker-0 x displacement (pixels)"


def main() -> None:
    trajectories, video_time, metadata, source_fps = _load_tracking_data(
        SOURCE_TRACKING_H5
    )
    decimation = _integer_decimation(source_fps)
    output_fps = source_fps / decimation

    source_frames_per_sample = int(source_fps * DURATION_S)
    pre_trigger_frames = int(PRE_TRIGGER_S * source_fps)
    gap_frames = int(GAP_AFTER_SAMPLE_S * source_fps)

    # Preserve the original sample-selection logic exactly.  Resampling occurs
    # only after each 30-s native-rate interval has been selected.
    marker0_x = trajectories[:, 0, 0]
    displacement = -(marker0_x - marker0_x[0])
    trigger = float(np.max(np.abs(displacement)) * TRIGGER_FRACTION)
    cursor = 0
    output_root = DATA_ROOT / OUTPUT_TOPOLOGY / OUTPUT_AMPLITUDE

    print(f"Source: {SOURCE_TRACKING_H5}")
    print(
        f"Resampling {source_fps:.8f} Hz -> {output_fps:.8f} Hz "
        f"(factor {decimation}); trigger={trigger:.3f} px"
    )

    written = 0
    for sample_index in range(NUM_SAMPLES):
        matches = np.flatnonzero(np.abs(displacement[cursor:]) > trigger)
        if len(matches) == 0:
            print(f"[STOP] No trigger for sample_{sample_index}.")
            break

        start = max(0, cursor + int(matches[0]) - pre_trigger_frames)
        stop = start + source_frames_per_sample
        if stop > len(trajectories):
            print(f"[STOP] Not enough frames for sample_{sample_index}.")
            break

        native_positions = trajectories[start:stop]
        native_actuation = displacement[start:stop]

        # Polyphase resampling includes the required anti-alias low-pass filter.
        positions = resample_poly(native_positions, up=1, down=decimation, axis=0)
        actuation = resample_poly(native_actuation, up=1, down=decimation).reshape(-1, 1)
        time = video_time[start:stop:decimation][: len(positions)]
        positions = positions[: len(time)]
        actuation = actuation[: len(time)]

        output_path = output_root / f"sample_{sample_index}" / "experiment.h5"
        _write_experiment(
            output_path,
            time,
            positions,
            actuation,
            metadata,
            output_fps,
            sample_index,
            source_fps,
            decimation,
        )
        print(
            f"[SAVED] {output_path}  "
            f"shape={positions.shape}, duration={time[-1] - time[0]:.3f} s"
        )
        written += 1
        cursor = stop + gap_frames

    print(f"Done: wrote {written} FPS-matched experiment file(s).")


if __name__ == "__main__":
    main()
