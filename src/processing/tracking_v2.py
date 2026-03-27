"""
processing/tracking.py
======================
Robust marker-tracking library.

Public API
----------
Marker               — dataclass: per-marker state + Kalman filter
detect_candidates    — HSV blob detection with circularity filter
match_to_markers     — Hungarian assignment against Kalman predictions
RobustGridTracker    — stateful tracker: initialise once, call process_frame()
VideoLoader          — context-manager wrapper around cv2.VideoCapture

Typical usage
-------------
    from processing.tracking import RobustGridTracker, VideoLoader

    tracker = RobustGridTracker(
        reference_positions,        # list of (x, y), one per marker
        hsv_params,                 # dict — see RobustGridTracker.__init__
        detection_params,           # dict — see RobustGridTracker.__init__
        tracker_params,             # dict — see RobustGridTracker.__init__
        show_ids=True,
    )

    with VideoLoader(str(video_path)) as loader:
        for frame_idx, frame in enumerate(loader.stream_frames()):
            original, mask, overlay, black = tracker.process_frame(frame)
            centroids = tracker.get_centroids_ordered()
"""

import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Defaults (used when caller omits a key in the params dicts) ──────────────

_DEFAULT_HSV = dict(
    red_lower_1=(0,   120,  70),
    red_upper_1=(10,  255, 255),
    red_lower_2=(170, 120,  70),
    red_upper_2=(180, 255, 255),
    blur_kernel=5,
    morph_size=7,
)

_DEFAULT_DETECTION = dict(
    min_area=9000,
    max_area=25000,
    circularity=0.1,
)

_DEFAULT_TRACKER = dict(
    max_dist=180,
    max_lost=15,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _morph_kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


# ── Detection ────────────────────────────────────────────────────────────────

def detect_candidates(
    frame: np.ndarray,
    hsv_params:       Optional[Dict] = None,
    detection_params: Optional[Dict] = None,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Detect red marker blobs in *frame*.

    Parameters
    ----------
    frame            : BGR image (numpy array)
    hsv_params       : override keys from _DEFAULT_HSV
    detection_params : override keys from _DEFAULT_DETECTION

    Returns
    -------
    candidates : list of dicts — keys: centroid, area, bbox, contour
    mask       : binary uint8 mask (for visualisation)
    """
    hp = {**_DEFAULT_HSV,       **(hsv_params       or {})}
    dp = {**_DEFAULT_DETECTION, **(detection_params or {})}

    bk   = hp["blur_kernel"] if hp["blur_kernel"] % 2 == 1 else hp["blur_kernel"] + 1
    mk   = _morph_kernel(hp["morph_size"])
    blur = cv2.GaussianBlur(frame, (bk, bk), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array(hp["red_lower_1"]), np.array(hp["red_upper_1"])),
        cv2.inRange(hsv, np.array(hp["red_lower_2"]), np.array(hp["red_upper_2"])),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  mk, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mk, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (dp["min_area"] <= area <= dp["max_area"]):
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        perim = cv2.arcLength(cnt, True)
        if perim == 0 or 4 * np.pi * area / (perim * perim) < dp["circularity"]:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        candidates.append({
            "centroid": (cx, cy),
            "area":     area,
            "bbox":     (x, y, w, h),
            "contour":  cnt,
        })
    return candidates, mask


# ── Per-marker Kalman state ───────────────────────────────────────────────────

@dataclass
class Marker:
    marker_id:   int
    centroid:    Tuple[float, float]
    area:        float = 0.0
    bbox:        Tuple[int, int, int, int] = (0, 0, 0, 0)
    contour:     np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1, 2), np.int32))
    history:     deque = field(default_factory=lambda: deque(maxlen=60))
    lost_frames: int = 0
    active:      bool = True
    kalman:      Optional[cv2.KalmanFilter] = field(default=None, repr=False)

    # new fields
    predicted_centroid: Tuple[float, float] = field(init=False)
    was_measured: bool = field(default=True, init=False)

    def __post_init__(self):
        self.history.append(self.centroid)
        self.kalman = self._init_kalman(self.centroid)
        self.predicted_centroid = self.centroid

    def _init_kalman(self, centroid: Tuple[float, float]) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)   # [x, y, vx, vy]
        dt = 1.0
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # more conservative process noise
        kf.processNoiseCov = np.diag([1e-2, 1e-2, 1.0, 1.0]).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        kf.statePost = np.array(
            [centroid[0], centroid[1], 0.0, 0.0], dtype=np.float32
        ).reshape(4, 1)
        return kf

    def predict_once(self) -> Tuple[float, float]:
        p = self.kalman.predict()
        self.predicted_centroid = (float(p[0, 0]), float(p[1, 0]))
        return self.predicted_centroid

    def correct(self, centroid: Tuple[float, float]) -> None:
        self.kalman.correct(
            np.array([[centroid[0]], [centroid[1]]], dtype=np.float32)
        )
        self.centroid = centroid
        self.history.append(centroid)
        self.lost_frames = 0
        self.active = True
        self.was_measured = True

    def mark_missed(self) -> None:
        # use the already-predicted location; DO NOT predict again
        self.centroid = self.predicted_centroid
        self.history.append(self.centroid)
        self.lost_frames += 1
        self.was_measured = False


# ── Hungarian assignment ──────────────────────────────────────────────────────

def match_to_markers(
    candidates: List[Dict],
    markers: List[Marker],
    max_dist: float = 120.0,
) -> Dict[int, Dict]:
    from scipy.optimize import linear_sum_assignment

    N, M = len(markers), len(candidates)
    if M == 0:
        return {}

    cost = np.full((N, M), 1e9, dtype=np.float64)

    for i, marker in enumerate(markers):
        px, py = marker.predicted_centroid

        # velocity-adaptive gate
        vx = float(marker.kalman.statePost[2, 0])
        vy = float(marker.kalman.statePost[3, 0])
        speed = np.hypot(vx, vy)
        gate = max_dist + 0.5 * speed
        gate = min(gate, max_dist * 2.0)

        for j, det in enumerate(candidates):
            dx = px - det["centroid"][0]
            dy = py - det["centroid"][1]
            d = np.hypot(dx, dy)
            if d < gate:
                cost[i, j] = d

    row_ind, col_ind = linear_sum_assignment(cost)

    out = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 1e8:
            out[markers[r].marker_id] = candidates[c]
    return out


# ── Main tracker class ────────────────────────────────────────────────────────

class RobustGridTracker:
    """
    Stateful tracker that maintains a permanent ID for each marker.

    Parameters
    ----------
    reference_positions : list of (x, y) — one per marker, in ID order
    hsv_params          : dict of HSV / blur / morph overrides (optional)
    detection_params    : dict of area / circularity overrides (optional)
    tracker_params      : dict of max_dist / max_lost overrides (optional)
    show_ids            : draw marker ID labels in visualisation frames
    """

    def __init__(
        self,
        reference_positions: List[Tuple[float, float]],
        hsv_params:          Optional[Dict] = None,
        detection_params:    Optional[Dict] = None,
        tracker_params:      Optional[Dict] = None,
        show_ids:            bool = True,
    ):
        self.reference        = list(reference_positions)
        self.n_markers        = len(self.reference)
        self.hsv_params       = {**_DEFAULT_HSV,       **(hsv_params       or {})}
        self.detection_params = {**_DEFAULT_DETECTION, **(detection_params or {})}
        self.tracker_params   = {**_DEFAULT_TRACKER,   **(tracker_params   or {})}
        self.show_ids         = show_ids

        self.markers: List[Marker] = [
            Marker(marker_id=mid, centroid=pos)
            for mid, pos in enumerate(self.reference)
        ]
        self._fps_history = deque(maxlen=30)
        self._prev_time   = time.time()

    # ------------------------------------------------------------------
    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        self._fps_history.append(1.0 / dt if dt > 0 else 0.0)
        fps = float(np.mean(self._fps_history))

        max_lost = self.tracker_params["max_lost"]
        max_dist = self.tracker_params["max_dist"]

        original = frame.copy()
        candidates, mask = detect_candidates(
            frame, self.hsv_params, self.detection_params
        )

        # predict each marker ONCE
        for m in self.markers:
            if m.active:
                m.predict_once()
            else:
                m.predicted_centroid = self.reference[m.marker_id]

        assignment = match_to_markers(candidates, self.markers, max_dist=max_dist)

        for m in self.markers:
            if m.marker_id in assignment:
                det = assignment[m.marker_id]
                m.area = det["area"]
                m.bbox = det["bbox"]
                m.contour = det["contour"]
                m.correct(det["centroid"])
            else:
                if m.active:
                    m.mark_missed()
                    if m.lost_frames > max_lost:
                        m.active = False
                else:
                    # keep inactive marker parked at last estimate or reference
                    m.was_measured = False

        overlay = original.copy()
        black = np.zeros(frame.shape, dtype=np.uint8)

        for m in self.markers:
            if not m.active and m.lost_frames > max_lost:
                color = (0, 0, 255)      # red = lost
            elif m.was_measured:
                color = (0, 255, 0)      # green = measured
            else:
                color = (0, 255, 255)    # yellow = predicted

            cx, cy = int(m.centroid[0]), int(m.centroid[1])
            cv2.circle(overlay, (cx, cy), 15, color, -1)
            cv2.circle(overlay, (cx, cy), 15, (0, 0, 0), 1)

            if m.was_measured and m.contour is not None:
                cv2.drawContours(overlay, [m.contour], -1, color, 1)

            cv2.circle(black, (cx, cy), 15, color, -1)
            cv2.circle(black, (cx, cy), 15, (255, 255, 255), 1)

            if self.show_ids:
                label = str(m.marker_id)
                cv2.putText(overlay, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4)
                cv2.putText(overlay, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(black, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
                cv2.putText(black, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)

        n_active = sum(1 for m in self.markers if m.active)
        hud = f"Tracking: {n_active}/{self.n_markers}   FPS: {fps:.1f}"
        for img in (overlay, black):
            cv2.putText(img, hud, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return original, mask, overlay, black

    # ------------------------------------------------------------------
    def get_centroids_ordered(self) -> List[Tuple[float, float]]:
        """
        Return centroid list in marker-ID order (0, 1, 2, …).
        Inactive markers fall back to their reference position so the
        list always has exactly n_markers entries.
        """
        return [
            m.centroid if m.active else self.reference[m.marker_id]
            for m in self.markers
        ]

    @property
    def n_active(self) -> int:
        return sum(1 for m in self.markers if m.active)


# ── VideoLoader ───────────────────────────────────────────────────────────────

class VideoLoader:
    """
    Thin context-manager wrapper around cv2.VideoCapture.

    Usage
    -----
        with VideoLoader("path/to/video.mp4") as loader:
            for frame_idx, frame in enumerate(loader.stream_frames()):
                ...
    """

    def __init__(self, path: str):
        self.path = path
        self.cap  = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"[VideoLoader] Cannot open: {path}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cap.release()

    def stream_frames(self):
        """Generator: yields BGR frames one at a time."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame