"""
Batch Video Processor — Topology 9
===================================
Scans a directory of paired camera / vibrometer files, tracks markers in
each video using RobustGridTracker, then saves trajectories + sensor
metadata to HDF5 via DataWriter.

Replaces the old MarkerTracker / VideoProcessor interface with the robust
tracking library (Kalman filter + Hungarian assignment).

Edit the CONFIG block, then run:
    python process_topology_9.py
"""

import cv2
import numpy as np
import os
import re
import sys
from pathlib import Path

# ── library imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processing.tracking_v2 import RobustGridTracker, VideoLoader
from data_io.writer import DataWriter
from data_io.loader import SensorLoader

# ═══════════════════════════════════════════════════════════════
#  CONFIG  —  edit this block only
# ═══════════════════════════════════════════════════════════════

# --- Grid ---
GRID_ROWS = 4
GRID_COLS = 4

# --- Paths ---
current_script_dir  = Path(__file__).parent.resolve()
DATA_DIR            = current_script_dir.parent.parent / "data"
CAMERA_DATA_DIR     = DATA_DIR / "camera_data"     / "topology_9_prestress"
SENSOR_DATA_DIR     = DATA_DIR / "vibrometer_data" / "topology_9_prestress"
EXPERIMENT_DATA_DIR = DATA_DIR / "experiment_data" / "topology_9_prestress"

# --- HSV thresholds ---
HSV_PARAMS = dict(
    red_lower_1=(0,   120,  70),
    red_upper_1=(10,  255, 255),
    red_lower_2=(170, 120,  70),
    red_upper_2=(180, 255, 255),
    blur_kernel=5,
    morph_size=7,
)

# --- Detection filters ---
DETECTION_PARAMS = dict(
    min_area=9000,
    max_area=25000,
    circularity=0.01,
)

# --- Tracker parameters ---
TRACKER_PARAMS = dict(
    max_dist=350,
    max_lost=15,
)

# --- Display ---
VISUALIZE = False          # set True to show live preview while processing
VIDEO_FPS = 30000.0 / 1001.0   # fallback FPS if cap.get() returns 0

# --- Reference layout ---
# Keys are "M0".."M{GRID_ROWS*GRID_COLS - 1}", values are (x, y) pixel positions.
#
# HOW TO FILL:
#   1. Open a representative still frame from each amplitude condition.
#   2. Read off the centroid (x, y) of each marker in row-major order
#      (top-left -> top-right, then next row, etc.).
#   3. Paste the coordinates below, replacing the placeholder zeros.
#
# HOW TO FIX A SWAP: rename the key (e.g. "M2" <-> "M3").
#   Coordinates stay as-is. Key order does not matter.
#
# HOW TO ADD MORE CONDITIONS: duplicate a dict block, give it a new key,
#   and map video filenames to that key in FILENAME_TO_LAYOUT below.

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
#         'M10': (2170.7, 1337.3),
#         'M11': (2672.5, 1343.5),
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
# REFERENCE_LAYOUTS = {

#     "topology_13": {
#         'M3': (2686.2, 272.6),
#         'M2': (2185.5, 294.6),
#         'M0': (1165.9, 328.6),
#         'M1': (1684.2, 335.9),
#         'M7': (2635.7, 816.7),
#         'M4': (1215.9, 868.9),
#         'M6': (2154.3, 869.8),
#         'M5': (1698.0, 882.2),
#         'M10': (2167.1, 1336.4),
#         'M11': (2649.9, 1337.9),
#         'M9': (1711.0, 1352.8),
#         'M8': (1217.5, 1390.8),
#         'M15': (2766.9, 1896.3),
#         'M14': (2206.1, 1906.1),
#         'M13': (1692.6, 1906.5),
#         'M12': (1155.3, 1939.5),
#     },

# }

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

# Maps video filename stems (or substrings) to a layout key.
# If a video filename does not match any entry, "default" is used.
# Example: FILENAME_TO_LAYOUT = { "C1420": "amp=2" }
FILENAME_TO_LAYOUT: dict = {}

# ═══════════════════════════════════════════════════════════════


# ── Helpers ───────────────────────────────────────────────────────────────────

def _layout_to_list(layout: dict) -> list:
    """Convert {'M0': (x,y), ...} to an ordered list by marker index."""
    n = GRID_ROWS * GRID_COLS
    return [layout[f"M{i}"] for i in range(n)]


def _validate_config():
    expected = GRID_ROWS * GRID_COLS
    errors = []
    if not REFERENCE_LAYOUTS:
        errors.append("  REFERENCE_LAYOUTS is empty — add at least one layout.")
    for key, layout in REFERENCE_LAYOUTS.items():
        missing = [f"M{i}" for i in range(expected) if f"M{i}" not in layout]
        if missing:
            errors.append(f"  '{key}' — missing keys: {missing}")
        if len(layout) != expected:
            errors.append(f"  '{key}' — has {len(layout)} entries, expected {expected}.")
    for stem, key in FILENAME_TO_LAYOUT.items():
        if key not in REFERENCE_LAYOUTS:
            errors.append(f"  FILENAME_TO_LAYOUT['{stem}'] -> '{key}' not in REFERENCE_LAYOUTS.")
    if errors:
        print("\n[ERROR] Config invalid:")
        for e in errors:
            print(e)
        sys.exit(1)
    print(f"[INFO] Grid {GRID_ROWS}x{GRID_COLS} = {expected} markers.  Config OK.\n")


def _resolve_layout_key(video_filename: str) -> str:
    """
    Return the layout key for a given video filename.

    Resolution order:
      1. First entry in FILENAME_TO_LAYOUT whose substring matches the filename.
      2. 'default' if that key exists in REFERENCE_LAYOUTS.
      3. The only key if REFERENCE_LAYOUTS has exactly one entry.
    """
    for stem, key in FILENAME_TO_LAYOUT.items():
        if stem in video_filename:
            return key
    if "default" in REFERENCE_LAYOUTS:
        return "default"
    if len(REFERENCE_LAYOUTS) == 1:
        return next(iter(REFERENCE_LAYOUTS))
    raise KeyError(
        f"Cannot resolve layout for '{video_filename}'. "
        f"Add a matching entry to FILENAME_TO_LAYOUT or rename one layout to 'default'."
    )


def _extract_amplitude(filename: str) -> str:
    """
    Extract 'amp=X' tag from a filename.
    Example: 'spring-mass-2D-3by3_amp=1_2026-02-04.csv' -> 'amp=1'
    """
    match = re.search(r"(amp=[\d\.]+)", filename)
    return match.group(1) if match else "unknown_amp"


def _read_xml_start_timestamp(xml_path: Path) -> float:
    """
    Read the recording start timestamp from the Sony XML sidecar file.
    Returns 0.0 if the file is missing or unparseable.
    """
    if not xml_path.exists():
        print(f"  [WARN] XML not found: {xml_path.name} — timestamp will be 0.0")
        return 0.0
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        # Sony XML: <CreationDate value="YYYY-MM-DDTHH:MM:SS+XX:XX"/>
        ns   = {"sony": "urn:schemas-sony-com:v1:sonyType"}
        node = root.find(".//CreationDate") or root.find(".//{*}CreationDate")
        if node is not None:
            from datetime import datetime, timezone
            raw = node.get("value", "")
            dt  = datetime.fromisoformat(raw)
            return dt.astimezone(timezone.utc).timestamp()
    except Exception as exc:
        print(f"  [WARN] Could not parse XML timestamp ({exc}) — using 0.0")
    return 0.0


# ── Core processing ───────────────────────────────────────────────────────────

def _process_pair(video_path: Path, sensor_filename: str, output_folder: Path):
    """
    Track markers in one video and save results alongside sensor metadata.
    """
    n_markers   = GRID_ROWS * GRID_COLS
    layout_key  = _resolve_layout_key(video_path.name)
    reference   = _layout_to_list(REFERENCE_LAYOUTS[layout_key])

    print(f"  Layout key  : {layout_key}")
    print(f"  Reference   : {n_markers} markers")

    # ── XML sidecar timestamp ────────────────────────────────────
    xml_name      = f"{video_path.stem.split('-')[0]}M01.XML"
    xml_path      = video_path.parent / xml_name
    xml_start_ts  = _read_xml_start_timestamp(xml_path)

    # ── Tracker ─────────────────────────────────────────────────
    tracker = RobustGridTracker(
        reference_positions=reference,
        hsv_params=HSV_PARAMS,
        detection_params=DETECTION_PARAMS,
        tracker_params=TRACKER_PARAMS,
        show_ids=False,
    )

    raw_trajectories = []   # list of [[cx, cy, 0], …] per frame
    frame_indices    = []   # list of int frame numbers

    with VideoLoader(str(video_path)) as loader:
        fps          = loader.cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
        total_frames = int(loader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  FPS={fps:.4f}   Frames={total_frames}")

        if VISUALIZE:
            WIN = f"Tracking: {video_path.name}  [Q=quit]"
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, 1280, 720)

        try:
            for frame_idx, frame in enumerate(loader.stream_frames()):
                original, mask, overlay, black = tracker.process_frame(frame)

                # Always emit a row; inactive markers fall back to reference.
                centroids   = tracker.get_centroids_ordered()
                frame_block = []
                for m in tracker.markers:
                    cx, cy = m.centroid
                    valid = 1 if getattr(m, "was_measured", True) else 0
                    frame_block.append([cx, cy, valid])
                raw_trajectories.append(frame_block)
                frame_indices.append(frame_idx)

                if tracker.n_active < n_markers and frame_idx % 30 == 0:
                    print(f"  [WARN] frame {frame_idx}: "
                          f"only {tracker.n_active}/{n_markers} active")

                if VISUALIZE:
                    # Simple single-view overlay when running batch
                    cv2.imshow(WIN, cv2.resize(overlay, (1280, 720)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if frame_idx % 200 == 0 and frame_idx > 0:
                    print(f"    frame {frame_idx:6d} / {total_frames}"
                          f"   active: {tracker.n_active}/{n_markers}")

        finally:
            if VISUALIZE:
                cv2.destroyAllWindows()

    # ── Convert and compute timestamps ──────────────────────────
    raw_trajectories = np.array(raw_trajectories)          # (F, N, 3)
    frame_indices    = np.array(frame_indices, dtype=int)
    video_time       = xml_start_ts + (frame_indices / fps)

    print(f"  Tracked frames : {len(frame_indices)}   shape: {raw_trajectories.shape}")

    # ── Save ────────────────────────────────────────────────────
    amp_tag    = _extract_amplitude(sensor_filename)
    final_meta = {
        "fps":              fps,
        "xml_start_time":   xml_start_ts,
        "source_video":     video_path.name,
        "source_sensor":    sensor_filename,
        "amplitude_group":  amp_tag,
        "layout_key":       layout_key,
    }

    data_writer = DataWriter(output_dir=output_folder)
    data_writer.save_to_h5("tracking_data.h5", raw_trajectories, video_time, final_meta)
    print(f"  Saved -> {output_folder / 'tracking_data.h5'}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    _validate_config()

    # ── Scan and pair files ──────────────────────────────────────
    camera_files = sorted(
        f for f in os.listdir(CAMERA_DATA_DIR) if f.endswith(".MP4"))
    sensor_files = sorted(
        (f for f in os.listdir(SENSOR_DATA_DIR) if f.endswith(".csv")),
        reverse=True)

    file_pairs = list(zip(camera_files, sensor_files))
    if not file_pairs:
        print("[ERROR] No file pairs found. Check CAMERA_DATA_DIR and SENSOR_DATA_DIR.")
        sys.exit(1)

    print(f"Found {len(file_pairs)} pair(s) to process in '{CAMERA_DATA_DIR.name}':")
    for v, s in file_pairs:
        print(f"  {v}  <->  {s}")
    print("-" * 50)

    # ── Processing loop ──────────────────────────────────────────
    for video_file_name, sensor_file_name in file_pairs:
        print(f"\nProcessing: {video_file_name}")

        video_path = CAMERA_DATA_DIR / video_file_name

        # Output folder: .../experiment_data/topology_9/amp=1/
        amp_tag       = _extract_amplitude(sensor_file_name)
        output_folder = EXPERIMENT_DATA_DIR / amp_tag
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"  Output folder : {output_folder}")

        _process_pair(video_path, sensor_file_name, output_folder)
        print("  -> Pair complete.")

    print("\n[DONE] All pairs processed successfully.")


if __name__ == "__main__":
    main()