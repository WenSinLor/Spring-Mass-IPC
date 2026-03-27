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
from processing.tracking_v2 import RobustGridTracker, VideoLoader, _morph_kernel

# ═══════════════════════════════════════════════════════════════
#  CONFIG  —  edit this block only
# ═══════════════════════════════════════════════════════════════

# --- Paths ---
current_script_dir = Path(__file__).parent.resolve()
DATA_DIR    = current_script_dir.parent.parent / "data"
VIDEO_FILE  = DATA_DIR / "camera_data"     / "topology_12_prestress" / "C1430.MP4"
OUTPUT_FILE = DATA_DIR / "experiment_data" / "topology_12_prestress" / "spring_mass_data.npz"

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
    circularity=0.01,
)

# --- Tracker parameters (passed as tracker_params) ---
TRACKER_PARAMS = dict(
    max_dist=350,
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
# REFERENCE_LAYOUTS = {

#     "topology_9": {
#         'M3': (2652.9, 264.6),
#         'M0': (1115.0, 317.0),
#         'M2': (2142.8, 323.4),
#         'M1': (1642.9, 360.5),
#         'M7': (2623.2, 798.1),
#         'M6': (2130.0, 805.0),
#         'M4': (1155.6, 849.5),
#         'M5': (1658.5, 850.6),
#         'M11': (2637.7, 1335.2),
#         'M10': (2140.9, 1361.6),
#         'M9': (1642.8, 1362.1),
#         'M8': (1159.1, 1368.5),
#         'M13': (1640.6, 1837.2),
#         'M14': (2155.0, 1843.0),
#         'M15': (2714.9, 1885.7),
#         'M12': (1103.5, 1906.3),
#     },

# }

# --- Topology 10 Prestress ---
# REFERENCE_LAYOUTS = {

#     "topology_10": {
#         'M3': (2683.9, 272.4),
#         'M0': (1177.9, 328.4),
#         'M2': (2186.3, 371.1),
#         'M1': (1698.6, 394.7),
#         'M7': (2658.0, 812.4),
#         'M6': (2172.7, 860.2),
#         'M4': (1217.4, 870.4),
#         'M5': (1701.3, 885.2),
#         'M11': (2170.7, 1337.3),
#         'M10': (2672.5, 1343.5),
#         'M9': (1690.2, 1354.4),
#         'M8': (1207.2, 1379.4),
#         'M14': (2187.9, 1821.4),
#         'M13': (1673.0, 1832.5),
#         'M15': (2748.0, 1895.3),
#         'M12': (1135.2, 1917.1),
#     },

# }

# --- Topology 10 Prestress Narma ---
# REFERENCE_LAYOUTS = {

#     "topology_10": {
#         'M3': (2682.8, 268.8),
#         'M0': (1176.3, 324.9),
#         'M2': (2185.1, 367.3),
#         'M1': (1697.2, 391.2),
#         'M7': (2656.6, 808.8),
#         'M6': (2171.2, 856.6),
#         'M4': (1216.0, 866.9),
#         'M5': (1699.9, 881.8),
#         'M11': (2169.5, 1333.8),
#         'M10': (2671.2, 1340.0),
#         'M9': (1688.9, 1350.9),
#         'M8': (1205.9, 1375.9),
#         'M14': (2186.7, 1817.9),
#         'M13': (1671.8, 1829.0),
#         'M15': (2746.8, 1891.8),
#         'M12': (1133.9, 1913.6),
#     },

# }

# --- Topology 11 Prestress ---
# REFERENCE_LAYOUTS = {

#     "topology_11": {
#         'M3': (2683.3, 270.2),
#         'M0': (1163.6, 327.5),
#         'M2': (2182.9, 384.6),
#         'M1': (1688.6, 404.7),
#         'M7': (2682.5, 809.9),
#         'M6': (2196.3, 867.7),
#         'M4': (1181.5, 869.1),
#         'M5': (1666.0, 895.6),
#         'M10': (2184.3, 1337.0),
#         'M11': (2682.8, 1339.4),
#         'M9': (1711.5, 1350.0),
#         'M8': (1159.1, 1375.5),
#         'M14': (2188.5, 1820.8),
#         'M13': (1674.1, 1823.9),
#         'M15': (2747.4, 1892.6),
#         'M12': (1134.6, 1915.0),
#     },

# }

# --- Topology 12 Prestress ---
# REFERENCE_LAYOUTS = {

#     "topology_12": {
#         'M3': (2685.6, 270.2),
#         'M0': (1162.6, 326.5),
#         'M2': (2184.0, 335.2),
#         'M1': (1687.8, 399.7),
#         'M7': (2665.7, 810.9),
#         'M4': (1157.0, 867.1),
#         'M6': (2192.6, 881.7),
#         'M5': (1724.5, 887.0),
#         'M11': (2711.1, 1332.7),
#         'M9': (1727.0, 1351.1),
#         'M10': (2194.5, 1353.6),
#         'M8': (1155.7, 1385.1),
#         'M14': (2205.4, 1822.6),
#         'M13': (1690.4, 1823.9),
#         'M15': (2765.9, 1890.8),
#         'M12': (1155.5, 1934.3),
#     },

# }

# --- Topology 13 Prestress ---
REFERENCE_LAYOUTS = {

    "topology_13": {
        'M3': (2686.2, 272.6),
        'M2': (2185.5, 294.6),
        'M0': (1165.9, 328.6),
        'M1': (1684.2, 335.9),
        'M7': (2635.7, 816.7),
        'M4': (1215.9, 868.9),
        'M6': (2154.3, 869.8),
        'M5': (1698.0, 882.2),
        'M10': (2167.1, 1336.4),
        'M11': (2649.9, 1337.9),
        'M9': (1711.0, 1352.8),
        'M8': (1217.5, 1390.8),
        'M15': (2766.9, 1896.3),
        'M14': (2206.1, 1906.1),
        'M13': (1692.6, 1906.5),
        'M12': (1155.3, 1939.5),
    },

}

# --- Topology 14 Prestress ---
# REFERENCE_LAYOUTS = {

#     "topology_14": {
#         'M3': (2686.0, 273.7),
#         'M2': (2184.8, 296.6),
#         'M0': (1164.6, 329.9),
#         'M1': (1683.0, 337.2),
#         'M7': (2633.6, 816.4),
#         'M6': (2162.2, 820.7),
#         'M5': (1700.8, 841.5),
#         'M4': (1217.1, 868.8),
#         'M11': (2642.2, 1338.6),
#         'M10': (2157.7, 1382.9),
#         'M8': (1217.1, 1391.0),
#         'M9': (1700.5, 1391.4),
#         'M15': (2766.1, 1895.5),
#         'M13': (1693.0, 1912.1),
#         'M14': (2206.9, 1915.8),
#         'M12': (1155.2, 1939.3),
#     },

# }

# --- Topology 15 Prestress ---
# REFERENCE_LAYOUTS = {

#     "topology_15": {
#         'M3': (2709.2, 289.3),
#         'M0': (1181.6, 321.8),
#         'M2': (2209.2, 405.0),
#         'M1': (1700.0, 420.2),
#         'M7': (2750.4, 818.7),
#         'M4': (1173.7, 860.8),
#         'M6': (2234.4, 879.9),
#         'M5': (1687.4, 897.4),
#         'M10': (2236.7, 1346.4),
#         'M9': (1684.5, 1349.9),
#         'M11': (2755.4, 1353.1),
#         'M8': (1159.5, 1377.4),
#         'M14': (2201.4, 1822.7),
#         'M13': (1691.5, 1833.1),
#         'M15': (2764.3, 1914.4),
#         'M12': (1149.2, 1932.4),
#     },

# }

# --- Topology 16 Prestress ---
# REFERENCE_LAYOUTS = {

#     "topology_16": {
#         'M3': (2709.1, 288.5),
#         'M0': (1198.5, 320.3),
#         'M2': (2211.2, 386.1),
#         'M1': (1712.2, 401.1),
#         'M7': (2689.1, 831.8),
#         'M4': (1237.1, 861.1),
#         'M6': (2189.9, 868.6),
#         'M5': (1712.4, 880.3),
#         'M10': (2190.0, 1354.2),
#         'M11': (2677.9, 1356.9),
#         'M9': (1704.9, 1359.4),
#         'M8': (1228.1, 1383.1),
#         'M14': (2201.5, 1837.0),
#         'M13': (1691.3, 1844.6),
#         'M15': (2763.4, 1912.9),
#         'M12': (1150.2, 1930.3),
#     },

# }

# Each tuple: (layout_key, video_file, output_npz)
# Add one entry per trial / angle you want to process.
TRIAL_BATCHES = [
    ("topology_13", VIDEO_FILE, OUTPUT_FILE),
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
                frame_block = []
                for m in tracker.markers:
                    cx, cy = m.centroid
                    valid = 1 if m.was_measured else 0
                    frame_block.append([cx, cy, valid])
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