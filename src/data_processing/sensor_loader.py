import pandas as pd
import numpy as np
from pathlib import Path

class SensorLoader:
    """
    Handles loading and parsing of vibrometer/sensor CSV data.
    """
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.df = None
        self._load_data()

    def _load_data(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Sensor file not found: {self.csv_path}")
        
        try:
            # FIX: Explicitly set separator to comma and handle spacing
            # quoting=3 (QUOTE_NONE) prevents pandas from getting confused by the quotes
            # we will clean the quotes manually later
            self.df = pd.read_csv(
                self.csv_path, 
                header=7,         # Line 8 contains the header
                sep=',',          # Force comma separator
                quotechar='"',    # Standard quote char
                skipinitialspace=True # Helps with " 12:55..." spaces
            )
        except Exception as e:
            raise IOError(f"Failed to parse CSV: {e}")

        # CLEANUP: The columns might still have quotes in their names
        # Example: ' " Epoch time (ms)"' -> 'Epoch time (ms)'
        self.df.columns = [c.replace('"', '').strip() for c in self.df.columns]
        
        # CLEANUP: If the data inside rows has quotes (e.g. " 1.952"), remove them
        # We apply a lambda to string columns only
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].astype(str).str.replace('"', '').str.strip()

        # Verify required columns exist
        if 'Epoch time (ms)' not in self.df.columns:
            # Fallback debug
            raise ValueError(f"Could not find 'Epoch time (ms)' in columns: {self.df.columns.tolist()}")

    def get_data(self):
        """
        Returns processed time (relative seconds) and distance data.
        """
        # 1. Extract Time
        # Ensure it's numeric (in case quotes made it a string)
        epochs = pd.to_numeric(self.df['Epoch time (ms)'])
        epochs = epochs.values
        
        # Convert to relative seconds (starting at 0)
        time_rel = (epochs - epochs[0]) / 1000.0
        
        # 2. Extract Data (Assuming last column is the measurement)
        # We look for 'Distance1' or similar
        dist_col = [c for c in self.df.columns if 'Distance' in c]
        
        if dist_col:
            target_col = dist_col[0]
        else:
            # Fallback to last column
            target_col = self.df.columns[-1]
            
        data = pd.to_numeric(self.df[target_col]).values
            
        return time_rel, data