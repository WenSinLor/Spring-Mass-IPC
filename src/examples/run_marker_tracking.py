import cv2
import numpy as np
from pathlib import Path
import sys

# Ensure the package root is in the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.tracking import MarkerTracker, VideoLoader

def main():
    # 1. Setup Path
    current_script_dir = Path(__file__).parent.resolve()
    DATA_DIR = current_script_dir.parent.parent / "data"
    VIDEO_FILE = DATA_DIR / "camera_data" / "topology_1" / "C1065.MP4"
    OUTPUT_FILE = DATA_DIR / "experiment_data" / "topology_1" / "spring_mass_data.npz"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 2. Setup Tracker (Using your calibrated values)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Area filter: 10k to 20k pixels
    tracker = MarkerTracker(lower_red1, upper_red1, lower_red2, upper_red2, min_area=10000, max_area=20000)

    # 3. Setup Visualization
    window_name = "2x2 View: Clean | Mask | Tracking | Dots"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # --- DATA STORAGE ---
    # We will temporarily store as a list of lists, then reshape at the end
    raw_trajectory_data = [] 
    valid_frames_count = 0
    expected_markers = 9 

    print(f"Processing: {VIDEO_FILE}")

    with VideoLoader(str(VIDEO_FILE)) as loader:
        total_frames = int(loader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video Length: {total_frames} frames")
        
        for frame_idx, frame in enumerate(loader.stream_frames()):
            
            clean_frame = frame.copy()
            mask, centroids = tracker.process_frame(clean_frame)

            # --- SORTING LOGIC ---
            # We only sort effectively if we have the expected number of markers (9)
            if len(centroids) == expected_markers:
                
                # 1. Sort all points by Y coordinate (Top -> Bottom)
                y_sorted = sorted(centroids, key=lambda p: p[1])
                
                # 2. Slice into 3 rows of 3 points each
                # Top row (first 3 points by Y), Middle row (next 3), Bottom row (last 3)
                row_1 = y_sorted[0:3]
                row_2 = y_sorted[3:6]
                row_3 = y_sorted[6:9]
                
                # 3. Sort each row by X coordinate (Left -> Right)
                row_1 = sorted(row_1, key=lambda p: p[0])
                row_2 = sorted(row_2, key=lambda p: p[0])
                row_3 = sorted(row_3, key=lambda p: p[0])
                
                # 4. Combine them back into a single list
                sorted_centroids = row_1 + row_2 + row_3

                # --- DATA COLLECTION ---
                # Create a frame block of shape (9, 3)
                frame_block = []
                for (cx, cy) in sorted_centroids:
                    frame_block.append([cx, cy, 0]) # x, y, z=0
                
                raw_trajectory_data.append(frame_block)
                valid_frames_count += 1
            
            else:
                # If we miss a marker, skip
                sorted_centroids = [] # Empty for visualization handling
                print(f"Frame {frame_idx}: Found {len(centroids)} markers. Skipping.")

            # --- DATA COLLECTION ---
            if len(sorted_centroids) == expected_markers:
                # Create a frame block of shape (9, 3)
                frame_block = []
                for (cx, cy) in sorted_centroids:
                    frame_block.append([cx, cy, 0]) # x, y, z=0
                
                raw_trajectory_data.append(frame_block)
                valid_frames_count += 1
            else:
                # If we miss a marker, we skip saving this frame to keep the array rectangular
                # Alternatively, you could append a block of np.nan
                print(f"Frame {frame_idx}: Found {len(sorted_centroids)} markers. Skipping.")

            # --- VISUALIZATION ---
            # (Only needed for you to watch, does not affect data saving)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame_with_dots = frame
            black_with_dots = np.zeros_like(clean_frame)

            for i, centroid in enumerate(sorted_centroids):
                # 1. Draw the Green Dot
                cv2.circle(frame_with_dots, centroid, 15, (0, 255, 0), -1)
                cv2.circle(black_with_dots, centroid, 15, (0, 255, 0), -1)
                
                # 2. Draw the Index Number (Shifted)
                # We shift x by -10 and y by -20 to put the number above-left of the dot
                text_pos = (centroid[0] - 15, centroid[1] - 20)
                
                # Optional: Draw black outline for better contrast
                cv2.putText(black_with_dots, str(i), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                           3, (0, 0, 0), 4) # Thick black line
                
                # Draw white text on top
                cv2.putText(black_with_dots, str(i), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                           3, (255, 255, 255), 2) # Thin white line

            # Stack for display
            row1 = np.hstack([clean_frame, mask_bgr])
            row2 = np.hstack([frame_with_dots, black_with_dots])
            grid = np.vstack([row1, row2])

            cv2.imshow(window_name, grid)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    # --- CONVERT TO 3D NUMPY ARRAY ---
    # Convert list of lists to (T, 9, 3)
    final_array = np.array(raw_trajectory_data)

    print("-" * 30)
    print("Processing Complete.")
    print(f"Total Valid Frames: {valid_frames_count}")
    print(f"Final Data Shape: {final_array.shape}") # Should be (T, 9, 3)
    
    # Save as compressed NPZ
    
    
    # We save it with the key 'trajectories'
    np.savez_compressed(OUTPUT_FILE, trajectories=final_array)
    print(f"Saved compressed data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()