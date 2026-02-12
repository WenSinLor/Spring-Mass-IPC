import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# --- 1. SETUP PATHS ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.trajectory import TrajectoryAnalyzer


# --- 2. PHYSICS MODEL ---
def underdamped_harmonic_oscillator(t, A, zeta, phi, B, omega_n):
    """
    Standard underdamped model:
        y(t) = A * exp(-zeta*omega_n*t) * cos(omega_d*t + phi) + B
        omega_d = omega_n * sqrt(1 - zeta^2)
    Assumes 0 <= zeta < 1 (enforced by bounds in curve_fit).
    """
    omega_d = omega_n * np.sqrt(1.0 - zeta**2)
    decay = np.exp(-zeta * omega_n * t)
    oscillation = np.cos(omega_d * t + phi)
    return A * decay * oscillation + B


def main():
    # ==========================================
    #       PHYSICS CONFIGURATION
    # ==========================================
    MASS_KG = 0.1012    # Mass in kg (101.2g)
    K_NM    = 222.15    # Spring Constant in N/m
    FPS     = 240       # Video Framerate (used only for TrajectoryAnalyzer)
    # ==========================================

    # --- 3. LOAD DATA ---
    current_script_dir = Path(__file__).parent.resolve()
    data_dir = current_script_dir.parent.parent / "data"
    data_file = data_dir / "experiment_data" / "damping_test" / "sample_3" / "damping_spring_mass_data.npz"

    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        return

    try:
        video_analyzer = TrajectoryAnalyzer(str(data_file), fps=FPS)
        time_axis, displacement_y = video_analyzer.get_displacement(node_idx=0, axis_idx=1)
        displacement_y = -displacement_y  # Invert so 'up' is positive
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Basic sanity
    time_axis = np.asarray(time_axis).astype(float)
    displacement_y = np.asarray(displacement_y).astype(float)

    if len(time_axis) < 5:
        print("Error: Not enough data points.")
        return

    # --- 4. INTERACTIVE SELECTION ---
    print("Displaying raw data...")
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, displacement_y, color="gray", label="Raw Data")
    plt.title("Select the FREE DECAY region")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (pixels)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show(block=False)
    plt.pause(0.5)

    print("\n--- INPUT TIME WINDOW ---")
    try:
        t_start = float(input("Enter START time (seconds): "))
        t_end = float(input("Enter END time (seconds): "))
    except ValueError:
        print("Invalid input.")
        return

    if t_end <= t_start:
        print("Error: END time must be greater than START time.")
        return

    # --- 5. CALCULATE THEORETICAL Wn ---
    omega_n_theory = np.sqrt(K_NM / MASS_KG)
    print(f"\n[Config] Mass={MASS_KG} kg | k={K_NM} N/m | Theoretical ωn={omega_n_theory:.6f} rad/s")

    # --- 6. DATA PREPARATION ---
    mask = (time_axis >= t_start) & (time_axis <= t_end)
    t_fit = time_axis[mask]
    y_fit = displacement_y[mask]

    if len(t_fit) < 10:
        print("Error: Too few samples in selection window.")
        return

    # Normalize time so fit starts at 0
    t_fit_norm = t_fit - t_fit[0]

    # Robust dt/fs from data (do NOT assume FPS spacing)
    dt = float(np.median(np.diff(t_fit_norm)))
    if not np.isfinite(dt) or dt <= 0:
        print("Error: Invalid time axis spacing.")
        return
    fs = 1.0 / dt

    # --- 7. SMART GUESSES ---
    B_guess = float(np.mean(y_fit))
    A_guess = float((np.max(y_fit) - np.min(y_fit)) / 2.0)

    # FFT-based frequency guess (rough initial guess)
    # Use dt from data (not FPS)
    freqs = np.fft.fftfreq(len(y_fit), dt)
    fft_vals = np.abs(np.fft.fft(y_fit - B_guess))
    idx_peak = int(np.argmax(fft_vals[1:]) + 1)  # skip DC
    freq_hz_guess = float(np.abs(freqs[idx_peak]))
    omega_n_guess = float(2 * np.pi * max(freq_hz_guess, 1e-6))

    # --- 8. FIT #1: FIXED FREQUENCY (THEORY ωn) ---
    def fixed_wrapper(t, A, zeta, phi, B):
        return underdamped_harmonic_oscillator(t, A, zeta, phi, B, omega_n_theory)

    p0_fixed = [A_guess, 0.05, 0.0, B_guess]
    bounds_fixed = ([-np.inf, 0.0, -np.pi, -np.inf], [np.inf, 0.999, np.pi, np.inf])

    y_model_fixed = np.zeros_like(y_fit, dtype=float)
    zeta_fixed = 0.0
    popt1 = None
    try:
        popt1, _ = curve_fit(
            fixed_wrapper,
            t_fit_norm,
            y_fit,
            p0=p0_fixed,
            bounds=bounds_fixed,
            maxfev=20000,
        )
        y_model_fixed = fixed_wrapper(t_fit_norm, *popt1)
        zeta_fixed = float(popt1[1])
    except Exception as e:
        print(f"[Fit #1] Failed: {e}")

    # --- 9. FIT #2: FREE FREQUENCY (EXPERIMENT ωn) ---
    p0_free = [A_guess, 0.05, 0.0, B_guess, omega_n_guess]
    bounds_free = ([-np.inf, 0.0, -np.pi, -np.inf, 0.1], [np.inf, 0.999, np.pi, np.inf, np.inf])

    y_model_free = np.zeros_like(y_fit, dtype=float)
    zeta_free = 0.0
    omega_free = 0.0
    B_opt_free = B_guess
    popt2 = None

    try:
        popt2, _ = curve_fit(
            underdamped_harmonic_oscillator,
            t_fit_norm,
            y_fit,
            p0=p0_free,
            bounds=bounds_free,
            maxfev=20000,
        )
        y_model_free = underdamped_harmonic_oscillator(t_fit_norm, *popt2)
        zeta_free = float(popt2[1])
        omega_free = float(popt2[4])
        B_opt_free = float(popt2[3])
    except Exception as e:
        print(f"[Fit #2] Failed: {e}")

    # --- 10. METHOD #3: LOGARITHMIC DECREMENT (ROBUST PEAK METHOD) ---
    peaks = np.array([], dtype=int)
    zeta_log_dec = 0.0
    delta = np.nan

    # Only proceed if we have a valid free-fit frequency
    if omega_free > 0 and (0.0 <= zeta_free < 1.0):
        omega_d_est = omega_free * np.sqrt(1.0 - zeta_free**2)  # damped frequency for peak spacing
        if omega_d_est > 0:
            period_samples = int((2.0 * np.pi / omega_d_est) * fs)
            min_dist = max(1, int(0.8 * period_samples))

            # Scaled prominence instead of hard-coded pixels
            prom = 0.05 * np.ptp(y_fit)  # 5% of peak-to-peak
            prom = float(max(prom, 1e-6))

            # Find peaks of y_fit (if you want troughs too, you can also run on -y_fit)
            peaks, props = find_peaks(y_fit, distance=min_dist, prominence=prom)

            if len(peaks) >= 3:
                peak_amps = np.abs(y_fit[peaks] - B_opt_free)

                # Filter tiny peaks before log
                floor = 0.02 * float(np.max(peak_amps))  # keep >2% of max peak amp
                keep = peak_amps > max(floor, 1e-12)
                peaks = peaks[keep]
                peak_amps = peak_amps[keep]

                if len(peak_amps) >= 3:
                    # Regression-based log decrement:
                    # ln(A_k) = ln(A0) - k*delta  => slope = -delta
                    k = np.arange(len(peak_amps), dtype=float)
                    lnA = np.log(peak_amps)

                    slope, intercept = np.polyfit(k, lnA, 1)
                    delta = float(-slope)

                    if delta > 0:
                        zeta_log_dec = float(delta / np.sqrt((2.0 * np.pi) ** 2 + delta**2))

                    print(
                        f"[Log Dec] peaks={len(peak_amps)} | prom={prom:.3g} | "
                        f"ωd_est={omega_d_est:.6f} rad/s | delta={delta:.8f}"
                    )
                else:
                    print("[Log Dec] Not enough valid peaks after filtering.")
            else:
                print("[Log Dec] Not enough peaks found. Try a longer clean free-decay window or adjust prom.")
        else:
            print("[Log Dec] Skipped: ωd_est invalid.")
    else:
        print("[Log Dec] Skipped: ω_free invalid or ζ_free out of range.")

    # --- 11. ERROR ANALYSIS ---
    # c = 2 * zeta * m * wn
    c_fixed = 2.0 * zeta_fixed * MASS_KG * omega_n_theory
    c_free = 2.0 * zeta_free * MASS_KG * omega_free
    c_log = 2.0 * zeta_log_dec * MASS_KG * omega_free if omega_free > 0 else 0.0

    err_omega = ((omega_free - omega_n_theory) / omega_n_theory) * 100.0 if omega_free > 0 else np.nan
    err_zeta = ((zeta_free - zeta_fixed) / zeta_fixed) * 100.0 if zeta_fixed > 0 else np.nan
    err_c = ((c_free - c_fixed) / c_fixed) * 100.0 if c_fixed > 0 else np.nan

    # --- 12. PRINT REPORT ---
    print("\n" + "=" * 85)
    print(f"{'DAMPING ANALYSIS REPORT':^85}")
    print("=" * 85)
    print(f"{'PARAMETER':<25} | {'THEORY (Fixed)':<15} | {'EXP (Curve Fit)':<15} | {'EXP (Log Dec)':<15}")
    print("-" * 85)
    print(f"{'Natural Freq (rad/s)':<25} | {omega_n_theory:<15.6f} | {omega_free:<15.6f} | {'--':<15}")
    print(f"{'Damping Ratio (Zeta)':<25} | {zeta_fixed:<15.6f} | {zeta_free:<15.6f} | {zeta_log_dec:<15.6f}")
    print(f"{'Damping Coeff (Ns/m)':<25} | {c_fixed:<15.6f} | {c_free:<15.6f} | {c_log:<15.6f}")
    print("-" * 85)
    print(f"{'Err ω (free vs theory) %':<25} | {'--':<15} | {err_omega:<15.6f} | {'--':<15}")
    print(f"{'Err ζ (free vs fixed) %':<25} | {'--':<15} | {err_zeta:<15.6f} | {'--':<15}")
    print(f"{'Err c (free vs fixed) %':<25} | {'--':<15} | {err_c:<15.6f} | {'--':<15}")
    print("=" * 85)

    # --- 13. PLOTTING ---
    plt.figure(figsize=(12, 7))

    # Use absolute time for plotting consistency
    t_plot = t_fit_norm + t_fit[0]

    plt.step(t_plot, y_fit, color="lightgray", where="mid", linewidth=3, label="Experimental Data")

    plt.plot(
        t_plot,
        y_model_fixed,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Theory Fit (fixed ωn) (ζ={zeta_fixed:.4f})",
    )

    plt.plot(
        t_plot,
        y_model_free,
        color="red",
        linestyle="-",
        linewidth=2,
        alpha=0.9,
        label=f"Exp Curve Fit (free ωn) (ζ={zeta_free:.4f})",
    )

    if len(peaks) > 0:
        plt.plot(
            t_plot[peaks],
            y_fit[peaks],
            "x",
            color="green",
            markersize=10,
            markeredgewidth=3,
            label=f"Log Dec Peaks (ζ={zeta_log_dec:.4f})",
        )

    plt.title(
        "Damping Methods Comparison\n"
        f"Curve Fit ζ={zeta_free:.4f} | Log Dec ζ={zeta_log_dec:.4f} | "
        f"ω_free={omega_free:.3f} rad/s | ω_theory={omega_n_theory:.3f} rad/s"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (pixels)")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
