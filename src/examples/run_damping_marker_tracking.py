import cv2
import numpy as np
from pathlib import Path
import sys

# Ensure the package root is in the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.tracking import MarkerTracker, VideoLoader

def print_contour_areas(mask: np.ndarray, frame_idx: int):
    """
    Finds contours, sorts them by size, and prints the TOP 5 largest areas
    on a NEW LINE for each frame. This allows you to scroll back and see history.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Calculate areas for all contours
        areas = [cv2.contourArea(c) for c in contours]
        
        # Sort descending (largest first)
        areas.sort(reverse=True)
        
        # Keep only top 5 largest (usually you only care about the biggest one)
        top_areas = areas[:10]
        
        # Format for clean printing
        areas_str = ", ".join([f"{a:.0f}" for a in top_areas])
        
        # Print on a NEW LINE (removed the \r and end='')
        print(f"Frame {frame_idx:<4} | Found {len(areas):<3} contours | Top Areas: [{areas_str}]")
    else:
        print(f"Frame {frame_idx:<4} | Found 0 contours")

def main():
    # 1. Setup Path
    current_script_dir = Path(__file__).parent.resolve()
    DATA_DIR = current_script_dir.parent.parent / "data"
    
    # NOTE: Update this to your specific video file. 
    VIDEO_FILE = DATA_DIR / "damping_camera_data" / "damping-test-sample-1.MP4"
    OUTPUT_FILE = DATA_DIR / "experiment_data" / "damping_test" / "sample_1" / "damping_spring_mass_data.npz"

    # --- CRITICAL FIX: CREATE DIRECTORY IF IT DOESN'T EXIST ---
    # This checks the parent folder of the output file and creates it if missing.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 2. Setup Tracker
    # Color Thresholds for Red
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Area filter: Adjust these based on the print output!
    min_marker_area = 8000  
    max_marker_area = 12000 

    tracker = MarkerTracker(lower_red1, upper_red1, lower_red2, upper_red2, 
                            min_area=min_marker_area, max_area=max_marker_area)

    # 3. Setup Visualization
    window_name = "2x2 View: Clean | Mask | Tracking | Dots"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # --- DATA STORAGE ---
    raw_trajectory_data = [] 
    valid_frames_count = 0
    expected_markers = 1 

    print(f"Processing: {VIDEO_FILE}")
    if not VIDEO_FILE.exists():
        print(f"\nError: Video file not found at {VIDEO_FILE}")
        print("Please make sure the path and filename are correct in the script.")
        return

    try:
        # Using the context manager for VideoLoader
        with VideoLoader(str(VIDEO_FILE)) as loader:
            total_frames = int(loader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video Length: {total_frames} frames")
            
            for frame_idx, frame in enumerate(loader.stream_frames()):
                
                clean_frame = frame.copy()
                
                # Process frame
                mask, centroids = tracker.process_frame(clean_frame)

                # Debug print
                print_contour_areas(mask, frame_idx)

                # Sorting & Collection
                sorted_centroids = sorted(centroids, key=lambda p: (p[1] // 50, p[0]))

                if len(sorted_centroids) == expected_markers:
                    # Append data [x, y, z] (z=0 placeholder)
                    frame_block = [[cx, cy, 0] for (cx, cy) in sorted_centroids]
                    raw_trajectory_data.append(frame_block)
                    valid_frames_count += 1

                # Visualization
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                frame_with_dots = frame.copy()
                black_with_dots = np.zeros_like(clean_frame)

                for i, centroid in enumerate(sorted_centroids):
                    cv2.circle(frame_with_dots, centroid, 15, (0, 255, 0), -1)
                    cv2.circle(black_with_dots, centroid, 15, (0, 255, 0), -1)
                    text_pos = (centroid[0] - 15, centroid[1] - 20)
                    cv2.putText(black_with_dots, str(i), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Stack images for display
                # Resize if needed to ensure they stack correctly (assuming same size)
                row1 = np.hstack([clean_frame, mask_bgr])
                row2 = np.hstack([frame_with_dots, black_with_dots])
                grid = np.vstack([row1, row2])

                cv2.imshow(window_name, grid)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"\nAn error occurred during video processing: {e}")
    finally:
        cv2.destroyAllWindows()
        print("\n" + "-" * 30)

    # --- SAVE DATA ---
    if valid_frames_count > 0:
        final_array = np.array(raw_trajectory_data)
        print("Processing Complete.")
        print(f"Total Valid Frames: {valid_frames_count}")
        print(f"Final Data Shape: {final_array.shape}")
        
        # Save to the ensured directory
        np.savez_compressed(OUTPUT_FILE, trajectories=final_array)
        print(f"Saved compressed data to {OUTPUT_FILE}")
    else:
        print("Processing Complete.")
        print("No valid frames with the expected number of markers were found.")
        print("Recommendation: Check 'All contour areas found' printout and adjust area filters.")

if __name__ == "__main__":
    main()