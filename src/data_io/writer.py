import h5py
from pathlib import Path

class DataWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_to_h5(self, filename, trajectories, time_array, metadata=None):
        filepath = self.output_dir / filename
        print(f"Saving to {filepath.name}...")
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset('trajectories', data=trajectories, compression="gzip")
            hf.create_dataset('time', data=time_array, compression="gzip")
            if metadata:
                for k, v in metadata.items(): hf.attrs[k] = v
        print("Save Successful.")
