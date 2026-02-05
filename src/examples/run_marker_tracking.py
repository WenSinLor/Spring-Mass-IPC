import cv2
import numpy as np
from pathlib import Path
from image_processing.marker_tracker import MarkerTracker
from image_processing.video_loader import VideoLoader

def main():
    # 1. Setup Path
    current_script_dir = Path(__file__).parent.resolve()
    data_dir = current_script_dir.parent.parent / "data"
    video_path = data_dir / "camera_data" / "C0989-008.MP4"
    output_file = data_dir / "experiment_data" / "spring_mass_data.npz"

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

    print(f"Processing: {video_path}")

    with VideoLoader(str(video_path)) as loader:
        total_frames = int(loader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video Length: {total_frames} frames")
        
        for frame_idx, frame in enumerate(loader.stream_frames()):
            
            clean_frame = frame.copy()
            mask, centroids = tracker.process_frame(clean_frame)

            # --- SORTING LOGIC ---
            # Sort by Y (rows of 50px height), then by X
            sorted_centroids = sorted(centroids, key=lambda p: (p[1] // 50, p[0]))

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
    np.savez_compressed(output_file, trajectories=final_array)
    print(f"Saved compressed data to {output_file}")

if __name__ == "__main__":
    main()