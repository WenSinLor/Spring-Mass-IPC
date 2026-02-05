import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class TrajectoryAnalyzer:
    """
    Manages loading and post-processing of tracking data.
    """
    def __init__(self, data_path: str, fps: float = 60.0):
        self.data_path = Path(data_path)
        self.fps = fps
        self.trajectories = None  # Shape: (T, N, 3)
        self.timestamps = None
        self._load_data()

    def _load_data(self):
        """Internal method to load data on initialization."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        
        with np.load(self.data_path) as data:
            # We look for 'trajectories' key as established in previous steps
            if 'trajectories' not in data:
                raise KeyError("The .npz file does not contain a 'trajectories' array.")
            self.trajectories = data['trajectories']
            
    def get_displacement(self, node_idx: int, axis_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates displacement relative to the first frame (t=0).
        
        Args:
            node_idx (int): Index of the marker (0-8)
            axis_idx (int): 0=X, 1=Y, 2=Z
            
        Returns:
            (time_axis, displacement_values)
        """
        # 1. Validation
        num_frames, num_nodes, _ = self.trajectories.shape
        if node_idx >= num_nodes:
            raise ValueError(f"Node index {node_idx} out of range (Max: {num_nodes-1})")

        # 2. Extract Data
        raw_coords = self.trajectories[:, node_idx, axis_idx]
        
        # 3. Calculate Relative Displacement
        # Displacement = Current_Pos - Initial_Pos
        initial_pos = raw_coords[0]
        displacement = raw_coords - initial_pos
        
        # 4. Generate Time Axis
        time_axis = np.arange(num_frames) / self.fps
        
        return time_axis, displacement

    @property
    def node_count(self) -> int:
        return self.trajectories.shape[1] if self.trajectories is not None else 0
