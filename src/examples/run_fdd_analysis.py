import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter1d


# ============================================================
# 1. LOAD YOUR experiment.h5 FORMAT
# ============================================================

def load_experiment_h5(h5_path: Path):
    """
    Load your experiment.h5 structure:

      /time_series/time
      /time_series/nodes/positions   shape (T, N, 3)

    where the last dimension is [x, y, valid_flag].
    We only use x,y for FDD.

    Returns
    -------
    t : ndarray, shape (T,)
    pos_xy : ndarray, shape (T, N, 2)
    """
    with h5py.File(h5_path, "r") as f:
        t = f["time_series/time"][:]
        pos = f["time_series/nodes/positions"][:]

    if pos.ndim != 3:
        raise ValueError(f"Expected positions shape (T,N,3), got {pos.shape}")

    if pos.shape[2] < 2:
        raise ValueError(f"Expected at least x,y coordinates, got {pos.shape}")

    pos_xy = pos[:, :, :2].astype(np.float64)
    return t, pos_xy


# ============================================================
# 2. CONVERT TO MULTI-CHANNEL SIGNAL MATRIX
# ============================================================

def build_channel_matrix(pos_xy, subtract_first_frame=True):
    """
    Convert (T, N, 2) -> (T, 2N)
    channel order:
        [x0, y0, x1, y1, ..., x15, y15]

    Parameters
    ----------
    pos_xy : ndarray, shape (T, N, 2)
    subtract_first_frame : bool
        If True, convert to displacement relative to first frame.

    Returns
    -------
    Y : ndarray, shape (T, 2N)
    """
    X = pos_xy.copy()

    if subtract_first_frame:
        X = X - X[0:1, :, :]

    T, N, _ = X.shape
    Y = X.reshape(T, 2 * N)

    # Remove mean from each channel
    Y = Y - np.mean(Y, axis=0, keepdims=True)
    return Y


# ============================================================
# 3. COMPUTE CROSS-PSD MATRIX
# ============================================================

def compute_cpsd_matrix(Y, fs, nperseg=256, noverlap=None, window="hann"):
    """
    Compute cross spectral density matrix Syy(f).

    Parameters
    ----------
    Y : ndarray, shape (T, M)
    fs : float
    nperseg : int
    noverlap : int or None

    Returns
    -------
    f : ndarray, shape (F,)
    Syy : ndarray, shape (F, M, M)
    """
    if noverlap is None:
        noverlap = nperseg // 2

    T, M = Y.shape

    # frequency grid from one auto-spectrum
    f, _ = signal.csd(
        Y[:, 0], Y[:, 0],
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        return_onesided=True
    )

    F = len(f)
    Syy = np.zeros((F, M, M), dtype=np.complex128)

    for i in range(M):
        for j in range(i, M):
            _, Pij = signal.csd(
                Y[:, i], Y[:, j],
                fs=fs,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
                return_onesided=True
            )
            Syy[:, i, j] = Pij
            if i != j:
                Syy[:, j, i] = np.conjugate(Pij)

    return f, Syy


# ============================================================
# 4. FDD = SVD OF CROSS-PSD AT EACH FREQUENCY
# ============================================================

def run_fdd(Syy):
    """
    Perform SVD at each frequency bin.

    Returns
    -------
    U_all : ndarray, shape (F, M, M)
    S_all : ndarray, shape (F, M)
    """
    F, M, _ = Syy.shape
    U_all = np.zeros((F, M, M), dtype=np.complex128)
    S_all = np.zeros((F, M), dtype=np.float64)

    for k in range(F):
        U, s, _ = np.linalg.svd(Syy[k], full_matrices=True)
        U_all[k] = U
        S_all[k] = s

    return U_all, S_all


# ============================================================
# 5. PLOTTING SINGULAR VALUES
# ============================================================

def plot_fdd_singular_values(f, S_all, title="FDD Singular Values", n_show=3):
    plt.figure(figsize=(9, 4))
    for i in range(min(n_show, S_all.shape[1])):
        plt.semilogy(f, S_all[:, i], label=f"SV{i+1}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Singular value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. ROBUST PEAK PICKING
# ============================================================

def find_fdd_peaks(f, S_all, min_freq=1.0, max_freq=15.0,
                   prominence_ratio=0.02, smooth_sigma=1.5):
    """
    Peak picking on the first singular value curve.

    Returns
    -------
    peaks_idx : ndarray
    f_sub : ndarray
    s1_sub : ndarray
    s1_smooth : ndarray
    """
    s1 = S_all[:, 0]

    mask = f >= min_freq
    if max_freq is not None:
        mask &= (f <= max_freq)

    idx = np.where(mask)[0]
    f_sub = f[idx]
    s1_sub = s1[idx]

    # smooth broad humps
    s1_smooth = gaussian_filter1d(s1_sub, sigma=smooth_sigma)

    prom = prominence_ratio * (np.max(s1_smooth) - np.min(s1_smooth))

    peaks_rel, _ = signal.find_peaks(s1_smooth, prominence=prom)
    peaks_idx = idx[peaks_rel]

    return peaks_idx, f_sub, s1_sub, s1_smooth


def plot_detected_peaks(f, S_all, peaks_idx, f_sub=None, s1_smooth=None, title="FDD Peak Detection"):
    plt.figure(figsize=(9, 4))
    plt.semilogy(f, S_all[:, 0], label="SV1 raw")

    if f_sub is not None and s1_smooth is not None:
        plt.semilogy(f_sub, s1_smooth, label="SV1 smooth")

    if len(peaks_idx) > 0:
        plt.scatter(f[peaks_idx], S_all[peaks_idx, 0], color="red", zorder=5, label="Detected peaks")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("SV1")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. EXTRACT MODE SHAPE AT A GIVEN FREQUENCY INDEX
# ============================================================

def extract_mode_shape(U_all, freq_idx, n_nodes, normalize=True):
    """
    Use the first singular vector at a chosen frequency index
    as the estimated mode shape.

    Parameters
    ----------
    U_all : ndarray, shape (F, M, M)
    freq_idx : int
    n_nodes : int

    Returns
    -------
    phi : ndarray, shape (N, 2), complex
    """
    u1 = U_all[freq_idx, :, 0]   # first singular vector
    phi = u1.reshape(n_nodes, 2)

    if normalize:
        mag = np.max(np.linalg.norm(phi, axis=1))
        if mag > 0:
            phi = phi / mag

    return phi


# ============================================================
# 8. PLOT MODE SHAPE ON 4x4 GRID
# ============================================================

def default_grid_coords_4x4():
    """
    Node coordinates only for visualization.
    Assumes row-major ordering:
        0  1  2  3
        4  5  6  7
        8  9 10 11
       12 13 14 15
    """
    coords = []
    for r in range(4):
        for c in range(4):
            coords.append([c, -r])
    return np.array(coords, dtype=float)


def plot_mode_shape(phi, coords=None, title="Mode Shape", scale=0.8, use_real_part=True):
    """
    Plot estimated mode shape.

    phi shape = (N,2), complex
    """
    N = phi.shape[0]

    if coords is None:
        if N != 16:
            raise ValueError("coords must be provided if N != 16")
        coords = default_grid_coords_4x4()

    disp = np.real(phi) if use_real_part else np.abs(phi)
    disp = disp * scale

    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=50, label="original")
    plt.scatter(coords[:, 0] + disp[:, 0], coords[:, 1] + disp[:, 1], s=50, label="deformed")

    for i in range(N):
        plt.plot(
            [coords[i, 0], coords[i, 0] + disp[i, 0]],
            [coords[i, 1], coords[i, 1] + disp[i, 1]],
            "k-", alpha=0.5
        )
        plt.text(coords[i, 0], coords[i, 1], str(i), fontsize=8)

    plt.axis("equal")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 9. MAC FOR COMPARING TWO MODE SHAPES
# ============================================================

def mac(phi_a, phi_b):
    a = np.asarray(phi_a).reshape(-1)
    b = np.asarray(phi_b).reshape(-1)

    num = np.abs(np.vdot(a, b)) ** 2
    den = np.vdot(a, a).real * np.vdot(b, b).real

    if den == 0:
        return 0.0
    return float(num / den)


# ============================================================
# 10. MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_experiment_h5(
    h5_path: Path,
    nperseg=256,
    min_freq=1.0,
    max_freq=15.0,
    prominence_ratio=0.02,
    smooth_sigma=1.5,
    manual_freqs=(6.0, 8.5),
):
    """
    Full FDD workflow on one experiment.h5
    """
    # --- Load ---
    t, pos_xy = load_experiment_h5(h5_path)
    Y = build_channel_matrix(pos_xy, subtract_first_frame=True)

    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    n_nodes = pos_xy.shape[1]

    print("=" * 60)
    print(f"FILE: {h5_path}")
    print(f"time shape       : {t.shape}")
    print(f"positions shape  : {pos_xy.shape}")
    print(f"channel matrix   : {Y.shape}")
    print(f"fs               : {fs:.6f} Hz")
    print("=" * 60)

    # --- FDD ---
    f, Syy = compute_cpsd_matrix(Y, fs=fs, nperseg=nperseg)
    U_all, S_all = run_fdd(Syy)

    # --- Plot singular values ---
    plot_fdd_singular_values(
        f, S_all,
        title=f"FDD Singular Values: {h5_path.name}",
        n_show=3
    )

    # --- Robust peak detection ---
    peaks_idx, f_sub, s1_sub, s1_smooth = find_fdd_peaks(
        f, S_all,
        min_freq=min_freq,
        max_freq=max_freq,
        prominence_ratio=prominence_ratio,
        smooth_sigma=smooth_sigma
    )

    plot_detected_peaks(
        f, S_all, peaks_idx,
        f_sub=f_sub,
        s1_smooth=s1_smooth,
        title=f"Detected Peaks: {h5_path.name}"
    )

    print("\nDetected peaks:")
    if len(peaks_idx) == 0:
        print("  None detected with current settings.")
    else:
        for i, p in enumerate(peaks_idx):
            print(f"  Peak {i+1}: f = {f[p]:.4f} Hz, SV1 = {S_all[p,0]:.4e}")

    # --- Extract manually chosen mode shapes ---
    manual_modes = []
    print("\nManual frequency mode shapes:")
    for target_freq in manual_freqs:
        idx = np.argmin(np.abs(f - target_freq))
        actual_freq = f[idx]
        phi = extract_mode_shape(U_all, idx, n_nodes=n_nodes, normalize=True)
        manual_modes.append((actual_freq, phi))

        print(f"  Requested {target_freq:.3f} Hz -> using {actual_freq:.4f} Hz")

        plot_mode_shape(
            phi,
            title=f"Manual Mode Shape at {actual_freq:.3f} Hz",
            scale=0.8,
            use_real_part=True
        )

    # --- Extract automatically detected peak mode shapes ---
    auto_modes = []
    if len(peaks_idx) > 0:
        print("\nAuto peak mode shapes:")
        for i, p in enumerate(peaks_idx):
            phi = extract_mode_shape(U_all, p, n_nodes=n_nodes, normalize=True)
            auto_modes.append((f[p], phi))

            print(f"  Auto peak {i+1}: {f[p]:.4f} Hz")

            plot_mode_shape(
                phi,
                title=f"Auto Peak Mode {i+1} at {f[p]:.3f} Hz",
                scale=0.8,
                use_real_part=True
            )

    return {
        "t": t,
        "pos_xy": pos_xy,
        "Y": Y,
        "fs": fs,
        "f": f,
        "Syy": Syy,
        "U_all": U_all,
        "S_all": S_all,
        "peaks_idx": peaks_idx,
        "manual_modes": manual_modes,
        "auto_modes": auto_modes,
    }


# ============================================================
# 11. EXAMPLE RUN
# ============================================================

if __name__ == "__main__":
    current_script_dir = Path(__file__).parent.resolve()
    h5_path = current_script_dir.parent.parent / "data" / "experiment_data" / "topology_14_prestress" / "amp=2.5" / "sample_0" / "experiment.h5"

    results = analyze_experiment_h5(
        h5_path=h5_path,
        nperseg=256,
        min_freq=1.0,
        max_freq=15.0,
        prominence_ratio=0.02,
        smooth_sigma=1.5,
        manual_freqs=(6.0, 8.5),   # you can change these
    )