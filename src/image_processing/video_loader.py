import cv2

class VideoLoader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

    def stream_frames(self):
        """Generator that yields frames one by one, ensuring it stops at EOF."""
        if not self.cap.isOpened():
            return

        # robustly get total frames to prevent infinite looping
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while self.cap.isOpened():
            # Safety Check 1: logical limit
            if current_frame >= total_frames:
                break
                
            ret, frame = self.cap.read()
            
            # Safety Check 2: OpenCV signal
            if not ret:
                break
            
            yield frame
            current_frame += 1