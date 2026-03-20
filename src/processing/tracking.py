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
    """
    Holds the state of a single tracked marker.

    marker_id   : permanent integer ID (never changes)
    centroid    : current (x, y) estimate (Kalman-corrected or predicted)
    lost_frames : frames since last successful detection
    active      : False once lost_frames exceeds max_lost
    """
    marker_id:   int
    centroid:    Tuple[float, float]
    area:        float                    = 0.0
    bbox:        Tuple[int, int, int, int] = (0, 0, 0, 0)
    contour:     np.ndarray               = field(
                     default_factory=lambda: np.zeros((1, 1, 2), np.int32))
    history:     deque                    = field(
                     default_factory=lambda: deque(maxlen=60))
    lost_frames: int                      = 0
    active:      bool                     = True
    kalman:      Optional[cv2.KalmanFilter] = field(default=None, repr=False)

    def __post_init__(self):
        self.history.append(self.centroid)
        self.kalman = self._init_kalman(self.centroid)

    # ------------------------------------------------------------------
    def _init_kalman(self, centroid: Tuple[float, float]) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)   # state: [x, y, vx, vy]
        dt = 1.0
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=np.float32)
        kf.measurementMatrix    = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        kf.processNoiseCov      = np.eye(4, dtype=np.float32) * 1e-1
        kf.processNoiseCov[2,2] = 5.0
        kf.processNoiseCov[3,3] = 5.0
        kf.measurementNoiseCov  = np.eye(2, dtype=np.float32) * 1e-2
        kf.errorCovPost         = np.eye(4, dtype=np.float32)
        kf.statePost = np.array(
            [centroid[0], centroid[1], 0.0, 0.0], dtype=np.float32
        ).reshape(4, 1)
        return kf

    def predict(self) -> Tuple[float, float]:
        """Advance Kalman one step; return predicted (x, y)."""
        p = self.kalman.predict()
        return float(p[0][0]), float(p[1][0])

    def correct(self, centroid: Tuple[float, float]) -> None:
        """Feed a new measurement into the Kalman filter."""
        self.kalman.correct(
            np.array([[centroid[0]], [centroid[1]]], dtype=np.float32))
        self.centroid    = centroid
        self.lost_frames = 0
        self.active      = True
        self.history.append(centroid)

    def update_lost(self) -> None:
        """No detection this frame — advance on prediction only."""
        pred = self.predict()
        self.centroid = pred
        self.history.append(pred)
        self.lost_frames += 1


# ── Hungarian assignment ──────────────────────────────────────────────────────

def match_to_markers(
    candidates:      List[Dict],
    markers:         List[Marker],
    max_dist:        float = _DEFAULT_TRACKER["max_dist"],
) -> Dict[int, Dict]:
    """
    Assign detections to markers using the Hungarian algorithm.

    Cost is the Euclidean distance from each marker's *Kalman-predicted*
    position to each detection centroid, gated at max_dist.

    Returns
    -------
    dict mapping marker_id → candidate dict for each successful assignment
    """
    from scipy.optimize import linear_sum_assignment

    N, M = len(markers), len(candidates)
    if M == 0:
        return {}

    cost = np.full((N, M), max_dist * 10, dtype=np.float64)
    for i, marker in enumerate(markers):
        px, py = marker.predict()
        for j, det in enumerate(candidates):
            d = np.hypot(px - det["centroid"][0], py - det["centroid"][1])
            if d < max_dist:
                cost[i, j] = d

    row_ind, col_ind = linear_sum_assignment(cost)
    return {
        markers[r].marker_id: candidates[c]
        for r, c in zip(row_ind, col_ind)
        if cost[r, c] < max_dist
    }


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
        """
        Run one tracking step.

        Returns
        -------
        original : unmodified copy of the input frame
        mask     : binary HSV mask (grayscale)
        overlay  : frame with tracking annotations drawn on top
        black    : black background with marker dots and IDs
        """
        now = time.time()
        dt  = now - self._prev_time
        self._prev_time = now
        self._fps_history.append(1.0 / dt if dt > 0 else 0.0)
        fps = float(np.mean(self._fps_history))

        max_dist = self.tracker_params["max_dist"]
        max_lost = self.tracker_params["max_lost"]

        original   = frame.copy()
        candidates, mask = detect_candidates(
            frame, self.hsv_params, self.detection_params)
        assignment = match_to_markers(candidates, self.markers, max_dist)

        for m in self.markers:
            if m.marker_id in assignment:
                det = assignment[m.marker_id]
                m.area, m.bbox, m.contour = det["area"], det["bbox"], det["contour"]
                m.correct(det["centroid"])
            else:
                m.update_lost()
                # Soft recovery: anchor back to reference while still lost
                if m.lost_frames > max_lost // 2:
                    m.correct(self.reference[m.marker_id])
                    m.lost_frames = max(0, m.lost_frames - 1)
                if m.lost_frames > max_lost:
                    m.active = False

        # ── Visualisation ────────────────────────────────────────────
        overlay = original.copy()
        black   = np.zeros(frame.shape, dtype=np.uint8)
        GREEN   = (0, 255, 0)

        for m in self.markers:
            if not m.active:
                continue
            cx, cy = int(m.centroid[0]), int(m.centroid[1])
            cv2.circle(overlay, (cx, cy), 15, GREEN, -1)
            cv2.circle(overlay, (cx, cy), 15, (0, 0, 0), 1)
            if m.lost_frames == 0:
                cv2.drawContours(overlay, [m.contour], -1, GREEN, 1)
            cv2.circle(black, (cx, cy), 15, GREEN, -1)
            cv2.circle(black, (cx, cy), 15, (255, 255, 255), 1)
            if self.show_ids:
                label = str(m.marker_id)
                cv2.putText(overlay, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4)
                cv2.putText(overlay, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1)
                cv2.putText(black, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
                cv2.putText(black, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)

        n_active = sum(1 for m in self.markers if m.active)
        hud = f"Tracking: {n_active}/{self.n_markers}   FPS: {fps:.1f}"
        for img in (overlay, black):
            cv2.putText(img, hud, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)

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