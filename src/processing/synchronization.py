import numpy as np
from scipy import signal

class TimeSynchronizer:
    @staticmethod
    def calculate_offset(ref_t, ref_y, target_t, target_y):
        print("Synchronizing Signals...")
        t_start, t_end = max(ref_t.min(), target_t.min()), min(ref_t.max(), target_t.max())
        if t_start >= t_end:
            t_start, t_end = min(ref_t.min(), target_t.min()), max(ref_t.max(), target_t.max())
        
        dt = 0.001
        t_common = np.arange(t_start, t_end, dt)
        
        ref_interp = np.interp(t_common, ref_t, ref_y) - np.mean(ref_y)
        tgt_interp = np.interp(t_common, target_t, target_y) - np.mean(target_y)
        
        lags = signal.correlation_lags(len(ref_interp), len(tgt_interp), mode='full')
        corr = signal.correlate(ref_interp, tgt_interp, mode='full')
        
        offset = lags[np.argmax(corr)] * dt
        print(f"  -> Calculated Offset: {offset:.4f} seconds")
        return offset
