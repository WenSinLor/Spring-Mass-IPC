import cv2
import numpy as np
from pathlib import Path
import sys

# Ensure the package root is in the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.tracking import MarkerTracker, VideoLoader

def main():
    # --- GRID CONFIGURATION ---
    # Define the grid size of markers you are tracking.
    # For a 4x4 grid, set both to 4. For 3x3, set both to 3.
    GRID_ROWS = 4
    GRID_COLS = 4
    # --------------------------

    # 1. Setup Path
    current_script_dir = Path(__file__).parent.resolve()
    DATA_DIR = current_script_dir.parent.parent / "data"
    VIDEO_FILE = DATA_DIR / "camera_data" / "topology_5" / "C1086.MP4"
    OUTPUT_FILE = DATA_DIR / "experiment_data" / "topology_5" / "spring_mass_data.npz"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 2. Setup Tracker (Using your calibrated values)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Area filter: 10k to 20k pixels
    tracker = MarkerTracker(lower_red1, upper_red1, lower_red2, upper_red2, min_area=10000, max_area=25000)

    # 3. Setup Visualization
    window_name = "2x2 View: Clean | Mask | Tracking | Dots"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # --- DATA STORAGE & CONFIG ---
    raw_trajectory_data = [] 
    valid_frames_count = 0
    expected_markers = GRID_ROWS * GRID_COLS

    print(f"Processing: {VIDEO_FILE}")
    print(f"Expecting a {GRID_ROWS}x{GRID_COLS} grid ({expected_markers} markers).")

    with VideoLoader(str(VIDEO_FILE)) as loader:
        total_frames = int(loader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video Length: {total_frames} frames")
        
        for frame_idx, frame in enumerate(loader.stream_frames()):
            # if frame_idx == 570:
            #     print("Pause")
            #     cv2.waitKey(-1)
            
            clean_frame = frame.copy()
            mask, centroids = tracker.process_frame(clean_frame, frame_idx)
            sorted_centroids = []

            # --- DYNAMIC SORTING LOGIC ---
            if len(centroids) == expected_markers:
                
                # 1. Sort all points by Y coordinate (Top -> Bottom)
                y_sorted = sorted(centroids, key=lambda p: p[1])
                
                # 2. Dynamically slice and sort rows
                for i in range(GRID_ROWS):
                    start_idx = i * GRID_COLS
                    end_idx = start_idx + GRID_COLS
                    
                    # Get a slice of points for the current row
                    row = y_sorted[start_idx:end_idx]
                    
                    # Sort this row by X coordinate (Left -> Right)
                    row_sorted = sorted(row, key=lambda p: p[0])
                    
                    # Add the sorted row to our final list
                    sorted_centroids.extend(row_sorted)

                # --- DATA COLLECTION ---
                frame_block = [[cx, cy, 0] for (cx, cy) in sorted_centroids]
                raw_trajectory_data.append(frame_block)
                valid_frames_count += 1
            
            else:
                # If we don't find the expected number of markers, skip the frame.
                if len(centroids) > 0: # Only print if some markers were found
                    print(f"Frame {frame_idx}: Found {len(centroids)} markers (expected {expected_markers}). Skipping.")

            # --- VISUALIZATION ---
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame_with_dots = frame.copy() # Use a fresh copy
            black_with_dots = np.zeros_like(clean_frame)

            for i, centroid in enumerate(sorted_centroids):
                cv2.circle(frame_with_dots, centroid, 15, (0, 255, 0), -1)
                cv2.circle(black_with_dots, centroid, 15, (0, 255, 0), -1)
                
                text_pos = (centroid[0] - 15, centroid[1] - 20)
                cv2.putText(black_with_dots, str(i), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                           3, (0, 0, 0), 4) # Black outline
                cv2.putText(black_with_dots, str(i), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                           3, (255, 255, 255), 2) # White text

            # Stack for display
            row1 = np.hstack([clean_frame, mask_bgr])
            row2 = np.hstack([frame_with_dots, black_with_dots])
            grid = np.vstack([row1, row2])

            cv2.imshow(window_name, grid)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    # --- CONVERT TO 3D NUMPY ARRAY ---
    final_array = np.array(raw_trajectory_data)

    print("-" * 30)
    print("Processing Complete.")
    print(f"Total Valid Frames: {valid_frames_count}")
    print(f"Final Data Shape: {final_array.shape}")
    
    # We save it with the key 'trajectories'
    if final_array.size > 0:
        np.savez_compressed(OUTPUT_FILE, trajectories=final_array)
        print(f"Saved compressed data to {OUTPUT_FILE}")
    else:
        print("No valid frames were processed. Nothing to save.")

if __name__ == "__main__":
    main()