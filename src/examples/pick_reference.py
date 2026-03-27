"""
Reference Frame Picker
======================
Scrub through any existing trial video to find a frame where
the arm is visibly at rest. Saves that frame and prints the
19 marker coordinates to paste into REFERENCE_LAYOUT in CONFIG.

Controls:
    LEFT / RIGHT arrow  —  step one frame
    A / D               —  step 10 frames
    SPACE               —  play / pause
    S                   —  save current frame + print coordinates
    Q                   —  quit
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path

# ── Point this at any one of your existing trial videos ───────
SOURCE = "/home/wensin/Documents/spring-mass-ipc/data/camera_data/topology_16_prestress/C1440.MP4"

# ── Same detection params as your main tracker ────────────────
NUM_MARKERS = 16
MIN_AREA    = 9000
MAX_AREA    = 25000
BLUR_KERNEL = 5
MORPH_SIZE  = 7

RED_LOWER_1 = (0,   75,  80)
RED_UPPER_1 = (10,  255, 255)
RED_LOWER_2 = (165, 75,  80)
RED_UPPER_2 = (180, 255, 255)
# ──────────────────────────────────────────────────────────────


def detect(frame):
    mk  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_SIZE, MORPH_SIZE))
    bk  = BLUR_KERNEL if BLUR_KERNEL % 2 == 1 else BLUR_KERNEL + 1
    blur = cv2.GaussianBlur(frame, (bk, bk), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lo1, hi1 = np.array(RED_LOWER_1), np.array(RED_UPPER_1)
    lo2, hi2 = np.array(RED_LOWER_2), np.array(RED_UPPER_2)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1),
                          cv2.inRange(hsv, lo2, hi2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  mk, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mk, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        perim = cv2.arcLength(cnt, True)
        if perim == 0 or 4 * np.pi * area / (perim ** 2) < 0.1:
            continue
        blobs.append((cx, cy, area))
    return blobs, mask


def draw_overlay(frame, blobs):
    vis = frame.copy()
    # Sort top→bottom, left→right so preview labels match what
    # REFERENCE_LAYOUT will contain
    sorted_blobs = sorted(blobs, key=lambda b: (b[1], b[0]))
    for i, (cx, cy, _) in enumerate(sorted_blobs[:NUM_MARKERS]):
        cv2.circle(vis, (int(cx), int(cy)), 8, (0, 255, 0), -1)
        cv2.putText(vis, f"M{i}", (int(cx)+10, int(cy)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    status = (f"{len(blobs)} markers detected"
              if len(blobs) >= NUM_MARKERS
              else f"WARNING: only {len(blobs)}/{NUM_MARKERS} markers detected")
    color = (0, 255, 0) if len(blobs) >= NUM_MARKERS else (0, 0, 255)
    cv2.putText(vis, status, (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return vis


def print_and_save(frame_idx, blobs, frame):
    sorted_blobs = sorted(blobs, key=lambda b: (b[1], b[0]))[:NUM_MARKERS]

    # Save the frame as reference image
    out_img = f"reference_frame_{frame_idx:05d}.png"
    cv2.imwrite(out_img, frame)
    print(f"\n[SAVED] Reference frame → {out_img}")

    # Print as dict — reorder keys freely to fix label mismatches across angles
    print("\n--- Paste into REFERENCE_LAYOUTS[\"coor_X\"] in tracker CONFIG ---")
    print("# To fix a swap: just change the key (e.g. swap 'M7' and 'M8').")
    print("# The coordinates stay as-is — only the key name changes.\n")
    print("{")
    for i, (cx, cy, _) in enumerate(sorted_blobs):
        print(f"    'M{i}': ({cx:.1f}, {cy:.1f}),")
    print("}")
    print("\n-------------------------------------------------------------------\n")


def main():
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {SOURCE}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {total} frames total.  Controls: ←/→ step 1 | A/D step 10 "
          f"| SPACE play/pause | S save+print | Q quit")

    WIN = "Reference Frame Picker  [←/→: step | A/D: ×10 | SPACE: play | S: save | Q: quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 500)

    frame_idx = 0
    playing   = False

    def go_to(idx):
        nonlocal frame_idx
        frame_idx = max(0, min(idx, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return
        blobs, mask = detect(frame)
        overlay = draw_overlay(frame, blobs)
        side    = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                             (overlay.shape[1] // 2, overlay.shape[0]))
        # Show frame index in corner
        cv2.putText(overlay, f"frame {frame_idx}/{total-1}", (18, overlay.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        combined = np.hstack([overlay,
                               cv2.resize(side, (overlay.shape[1]//2,
                                                  overlay.shape[0]))])
        cv2.imshow(WIN, combined)
        return frame, blobs

    result = go_to(0)

    while True:
        key = cv2.waitKey(30 if playing else 0) & 0xFF

        if playing:
            ret, frame = cap.read()
            if not ret:
                playing = False
            else:
                frame_idx += 1
                blobs, mask = detect(frame)
                overlay = draw_overlay(frame, blobs)
                cv2.putText(overlay,
                            f"frame {frame_idx}/{total-1}",
                            (18, overlay.shape[0]-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
                cv2.imshow(WIN, overlay)

        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = not playing
        elif key == 81 or key == ord('a'):   # left arrow or A
            playing = False
            step = 1 if key == 81 else 10
            result = go_to(frame_idx - step)
        elif key == 83 or key == ord('d'):   # right arrow or D
            playing = False
            step = 1 if key == 83 else 10
            result = go_to(frame_idx + step)
        elif key == ord('s'):
            # Re-detect on current frame to be sure
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                blobs, _ = detect(frame)
                if len(blobs) >= NUM_MARKERS:
                    print_and_save(frame_idx, blobs, frame)
                else:
                    print(f"[WARN] Only {len(blobs)} markers — "
                          f"find a cleaner frame first.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()