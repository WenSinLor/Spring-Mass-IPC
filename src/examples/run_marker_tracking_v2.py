"""
Robust Grid Marker Tracker
==========================
Config-and-orchestration script.  All tracking logic lives in
processing/tracking.py — edit that file to change tracker behaviour.

Edit the CONFIG block below, then run:
    python tracker_grid_robust.py

Hotkeys during live preview:
    Q  —  quit
    S  —  save screenshot
    T  —  HSV tuner
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# ── library import ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processing.tracking import RobustGridTracker, VideoLoader, _morph_kernel

# ═══════════════════════════════════════════════════════════════
#  CONFIG  —  edit this block only
# ═══════════════════════════════════════════════════════════════

# --- Paths ---
current_script_dir = Path(__file__).parent.resolve()
DATA_DIR    = current_script_dir.parent.parent / "data"
VIDEO_FILE  = DATA_DIR / "camera_data"     / "topology_9_prestress" / "C1421.MP4"
OUTPUT_FILE = DATA_DIR / "experiment_data" / "topology_9_prestress" / "spring_mass_data.npz"

# --- Grid ---
GRID_ROWS = 4
GRID_COLS = 4

# --- HSV thresholds (passed to RobustGridTracker as hsv_params) ---
HSV_PARAMS = dict(
    red_lower_1=(0,   120,  70),
    red_upper_1=(10,  255, 255),
    red_lower_2=(170, 120,  70),
    red_upper_2=(180, 255, 255),
    blur_kernel=5,
    morph_size=7,
)

# --- Detection filters (passed as detection_params) ---
DETECTION_PARAMS = dict(
    min_area=9000,
    max_area=25000,
    circularity=0.1,
)

# --- Tracker parameters (passed as tracker_params) ---
TRACKER_PARAMS = dict(
    max_dist=180,
    max_lost=15,
)

# --- Display ---
SHOW_PREVIEW = True
SHOW_IDS     = True
QUAD_W       = 1280
QUAD_H       = 720
VIDEO_FPS    = 30000.0 / 1001.0   # fallback if cap.get() returns 0

# --- Reference layout ---
# Same format as tracker_fixed.py: one dict per camera angle.
# Keys are "M0".."M{GRID_ROWS*GRID_COLS - 1}", values are (x, y) pixel positions.
#
# HOW TO FILL:
#   1. Open a representative still frame from each angle.
#   2. Read off the centroid (x, y) of each marker in the order you
#      want them numbered (e.g. row-major: top-left -> top-right,
#      then next row, etc.).
#   3. Paste the coordinates below, replacing the placeholder zeros.
#
# HOW TO FIX A SWAP: rename the key (e.g. "M2" <-> "M3").
#   Coordinates stay as-is. Key order does not matter.
#
# HOW TO ADD MORE ANGLES: duplicate a dict block and add its key to
#   TRIAL_BATCHES below.

# --- Topology 9 ---
# REFERENCE_LAYOUTS = {

#     "topology_9": {
#         'M3': (2510.4, 304.5),
#         'M2': (2050.3, 335.1),
#         'M0': (1142.1, 347.0),
#         'M1': (1642.4, 371.4),
#         'M7': (2507.7, 790.3),
#         'M6': (2063.2, 799.1),
#         'M4': (1167.6, 834.4),
#         'M5': (1646.8, 836.0),
#         'M11': (2493.3, 1220.3),
#         'M10': (2043.7, 1249.4),
#         'M9': (1644.4, 1257.7),
#         'M8': (1181.7, 1259.2),
#         'M13': (1655.7, 1705.6),
#         'M14': (2062.0, 1712.4),
#         'M15': (2549.5, 1715.5),
#         'M12': (1163.1, 1739.3),
#     },

# }

# --- Topology 9 Narma ---
# REFERENCE_LAYOUTS = {

#     "topology_9": {
#         'M3': (2510.3, 305.6),
#         'M2': (2045.6, 335.2),
#         'M0': (1125.9, 348.4),
#         'M1': (1628.0, 371.3),
#         'M7': (2506.1, 790.9),
#         'M6': (2059.2, 798.9),
#         'M5': (1638.2, 835.1),
#         'M4': (1157.9, 835.4),
#         'M11': (2492.5, 1220.5),
#         'M10': (2041.8, 1248.5),
#         'M9': (1640.3, 1256.9),
#         'M8': (1176.8, 1260.3),
#         'M13': (1654.7, 1705.2),
#         'M14': (2061.2, 1711.6),
#         'M15': (2548.5, 1715.8),
#         'M12': (1162.9, 1739.6),
#     },

# }

# --- Topology 9 Prestress ---
REFERENCE_LAYOUTS = {

    "topology_9": {
        'M3': (2652.9, 264.6),
        'M0': (1115.0, 317.0),
        'M2': (2142.8, 323.4),
        'M1': (1642.9, 360.5),
        'M7': (2623.2, 798.1),
        'M6': (2130.0, 805.0),
        'M4': (1155.6, 849.5),
        'M5': (1658.5, 850.6),
        'M11': (2637.7, 1335.2),
        'M10': (2140.9, 1361.6),
        'M9': (1642.8, 1362.1),
        'M8': (1159.1, 1368.5),
        'M13': (1640.6, 1837.2),
        'M14': (2155.0, 1843.0),
        'M15': (2714.9, 1885.7),
        'M12': (1103.5, 1906.3),
    },

}

# Each tuple: (layout_key, video_file, output_npz)
# Add one entry per trial / angle you want to process.
TRIAL_BATCHES = [
    ("topology_9", VIDEO_FILE, OUTPUT_FILE),
]

# ═══════════════════════════════════════════════════════════════


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_config():
    expected = GRID_ROWS * GRID_COLS
    errors = []
    for layout_key, layout in REFERENCE_LAYOUTS.items():
        missing = [f"M{i}" for i in range(expected) if f"M{i}" not in layout]
        if missing:
            errors.append(f"  '{layout_key}' — missing keys: {missing}")
        if len(layout) != expected:
            errors.append(
                f"  '{layout_key}' — has {len(layout)} entries, expected {expected}.")
    for layout_key, _, _ in TRIAL_BATCHES:
        if layout_key not in REFERENCE_LAYOUTS:
            errors.append(
                f"  TRIAL_BATCHES references '{layout_key}' not in REFERENCE_LAYOUTS.")
    if errors:
        print("\n[ERROR] REFERENCE_LAYOUTS invalid:")
        for e in errors:
            print(e)
        sys.exit(1)
    print(f"[INFO] Grid {GRID_ROWS}x{GRID_COLS} = {expected} markers.  Config OK.\n")


def _layout_to_list(layout: dict) -> list:
    """Convert {'M0': (x,y), ...} to an ordered list by marker index."""
    n = GRID_ROWS * GRID_COLS
    return [layout[f"M{i}"] for i in range(n)]


def _draw_quad_view(original, mask, overlay, black):
    hw, hh = QUAD_W // 2, QUAD_H // 2
    def r(img):
        return cv2.resize(img, (hw, hh), interpolation=cv2.INTER_LINEAR)
    return np.vstack([
        np.hstack([r(original), r(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))]),
        np.hstack([r(overlay),  r(black)]),
    ])


def _add_labels(canvas):
    hw, hh = QUAD_W // 2, QUAD_H // 2
    for (x, y), text in [
        ((8,    hh - 8),     "Original"),
        ((hw+8, hh - 8),     "HSV Mask"),
        ((8,    QUAD_H - 8), "Tracked (overlay)"),
        ((hw+8, QUAD_H - 8), "Tracked (black bg)"),
    ]:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.line(canvas, (hw, 0),  (hw, QUAD_H), (60, 60, 60), 1)
    cv2.line(canvas, (0,  hh), (QUAD_W, hh), (60, 60, 60), 1)


def _hsv_tuner(cap):
    """Interactive HSV threshold tuner — press Q to print updated values."""
    WIN = "HSV Tuner  (Q = done)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 900, 600)

    def nothing(_): pass
    for name, val, mx in [
        ("H_lo1",   0,  30), ("S_lo1", 120, 255), ("V_lo1",  70, 255),
        ("H_hi1",  10,  30), ("S_hi1", 255, 255), ("V_hi1", 255, 255),
        ("H_lo2", 170, 180), ("S_lo2", 120, 255), ("V_lo2",  70, 255),
        ("H_hi2", 180, 180), ("S_hi2", 255, 255), ("V_hi2", 255, 255),
    ]:
        cv2.createTrackbar(name, WIN, val, mx, nothing)

    mk = _morph_kernel(3)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        def tb(n): return cv2.getTrackbarPos(n, WIN)
        lo1 = np.array([tb("H_lo1"), tb("S_lo1"), tb("V_lo1")], dtype=np.uint8)
        hi1 = np.array([tb("H_hi1"), tb("S_hi1"), tb("V_hi1")], dtype=np.uint8)
        lo2 = np.array([tb("H_lo2"), tb("S_lo2"), tb("V_lo2")], dtype=np.uint8)
        hi2 = np.array([tb("H_hi2"), tb("S_hi2"), tb("V_hi2")], dtype=np.uint8)
        mask = cv2.morphologyEx(
            cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2)),
            cv2.MORPH_CLOSE, mk, iterations=2)
        cv2.imshow(WIN, np.hstack([
            cv2.resize(frame, (450, 400)),
            cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (450, 400)),
        ]))
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("\n--- Paste into HSV_PARAMS in CONFIG ---")
            print(f"    red_lower_1 = {tuple(lo1.tolist())},")
            print(f"    red_upper_1 = {tuple(hi1.tolist())},")
            print(f"    red_lower_2 = {tuple(lo2.tolist())},")
            print(f"    red_upper_2 = {tuple(hi2.tolist())},")
            break
    cv2.destroyWindow(WIN)


# ── Per-trial processing ──────────────────────────────────────────────────────

def _process_trial(video_file, output_file, reference):
    video_file  = Path(video_file)
    output_file = Path(output_file)
    n_markers   = GRID_ROWS * GRID_COLS

    tracker = RobustGridTracker(
        reference_positions=reference,
        hsv_params=HSV_PARAMS,
        detection_params=DETECTION_PARAMS,
        tracker_params=TRACKER_PARAMS,
        show_ids=SHOW_IDS,
    )

    raw_trajectory_data = []
    valid_frames_count  = 0

    WIN = "Robust Grid Tracker  [Q=quit | S=screenshot | T=HSV tuner]"

    with VideoLoader(str(video_file)) as loader:
        fps          = loader.cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
        total_frames = int(loader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] Video : {video_file.name}   FPS={fps:.4f}   Frames={total_frames}")

        if SHOW_PREVIEW:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, QUAD_W, QUAD_H)

        try:
            for frame_idx, frame in enumerate(loader.stream_frames()):
                original, mask, overlay, black = tracker.process_frame(frame)

                # Always emit a row; inactive markers fall back to reference.
                centroids   = tracker.get_centroids_ordered()
                frame_block = [[cx, cy, 0] for (cx, cy) in centroids]
                raw_trajectory_data.append(frame_block)
                valid_frames_count += 1

                if tracker.n_active < n_markers and frame_idx % 30 == 0:
                    print(f"  [WARN] frame {frame_idx}: "
                          f"only {tracker.n_active}/{n_markers} active")

                if SHOW_PREVIEW:
                    canvas = _draw_quad_view(original, mask, overlay, black)
                    _add_labels(canvas)
                    cv2.imshow(WIN, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        fn = f"screenshot_{frame_idx:05d}.png"
                        cv2.imwrite(fn, canvas)
                        print(f"[INFO] Screenshot saved: {fn}")
                    elif key == ord('t'):
                        _hsv_tuner(loader.cap)

                if frame_idx % 100 == 0 and frame_idx > 0:
                    print(f"  frame {frame_idx:6d} / {total_frames}"
                          f"   active: {tracker.n_active}/{n_markers}")

        finally:
            if SHOW_PREVIEW:
                cv2.destroyAllWindows()

    final_array = np.array(raw_trajectory_data)   # shape: (F, N, 3)
    print("-" * 40)
    print(f"  Frames processed : {valid_frames_count}")
    print(f"  Data shape       : {final_array.shape}")

    if final_array.size > 0:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output_file), trajectories=final_array)
        print(f"  Saved -> {output_file.resolve()}")
    else:
        print("  [WARN] No data collected — nothing saved.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    _validate_config()

    for layout_key, video_file, output_file in TRIAL_BATCHES:
        reference = _layout_to_list(REFERENCE_LAYOUTS[layout_key])
        print(f"\n{'='*60}")
        print(f"  Layout : {layout_key}")
        print(f"  Video  : {Path(video_file).name}")
        print(f"{'='*60}")
        _process_trial(video_file, output_file, reference)

    print("\n[DONE] All trials processed.")


if __name__ == "__main__":
    main()