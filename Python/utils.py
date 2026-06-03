import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from farms_core import pylog
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

def find_peak(drive, outputs, times):

    # Find peaks of top oscillator after transient
    if np.isscalar(drive):
        reference_signal = outputs[:, 0]
        # Ignore first 3 seconds
        start_idx = np.searchsorted(times, 3.0)
        # Find peaks
        peaks, _ = find_peaks(
            reference_signal[start_idx:],
            distance=80,       # avoid nearby peaks
            prominence=0.05,
        )
        # Convert back to global indices
        peaks = peaks + start_idx
        # Keep first two peaks
        transition_times = times[peaks[:1]]
    else:
        reference_signal = outputs[:, 0]
        peaks, _ = find_peaks(
            reference_signal,
            distance=80,
            prominence=0.05,
        )
        peak_times = times[peaks]
        peak1_idx = np.where(peak_times > 5.0)[0][0]
        peak2_idx = np.where(peak_times > 10.0)[0][0]
        transition_times = [
            peak_times[peak1_idx],
            peak_times[peak2_idx],
        ]

    return transition_times

def find_travelling_wave_peaks(outputs, times, oscs, after_time=10.0):
    """
    For each oscillator in oscs, find the first peak after `after_time`
    that comes strictly after the previous oscillator's peak.
    Returns list of (osc_index, time_of_peak) pairs.
    """
    wave_peaks = []
    search_after = after_time

    for osc in oscs:
        signal = outputs[:, osc]
        start_idx = np.searchsorted(times, search_after)
        if start_idx >= len(times) - 1:
            break
        peaks, _ = find_peaks(signal[start_idx:], distance=20, prominence=0.01)
        if len(peaks) == 0:
            break
        peak_global = peaks[0] + start_idx
        t_peak = times[peak_global]
        wave_peaks.append((osc, t_peak))
        search_after = t_peak  # next osc must peak after this one

    return wave_peaks


def compute_instantaneous_frequency(times, phases):
    """Compute instantaneous frequency from oscillator phase traces.

    Parameters
    ----------
    times : np.ndarray, shape (n_steps,)
        Sample times.
    phases : np.ndarray, shape (n_steps,) or (n_steps, n_channels)
        Oscillator phases in radians.

    Returns
    -------
    freq_hz : np.ndarray
        Instantaneous frequency in Hz.
    """
    times = np.asarray(times)
    phases = np.asarray(phases)

    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a 1D array with at least two samples")

    if phases.ndim == 1:
        unwrapped = np.unwrap(phases)
        return np.gradient(unwrapped, times) / (2.0 * np.pi)

    if phases.ndim == 2:
        unwrapped = np.unwrap(phases, axis=0)
        return np.gradient(unwrapped, times, axis=0) / (2.0 * np.pi)

    raise ValueError(f"phases must be 1D or 2D, got {phases.shape}")

def plot_stacked_group(ax, times, outputs, oscs, base_color, label, group_boxes=None, 
                       scale_bar=None, transition_times=None, wave_peaks=None, drive=None, body = True):
    n = len(oscs)

    # Dynamically compute offset based on actual signal amplitude
    all_signals = [outputs[:, osc] for osc in oscs]
    max_amp = max(np.max(np.abs(s)) for s in all_signals)
    offset = max_amp * 2.5

    # Smooth alpha gradient for any number of oscillators
    alphas = np.linspace(1.0, 0.45, n)
    rgb = mcolors.to_rgb(base_color)

    duration = times[-1] - times[0]
    # Scale margins proportionally to duration
    label_margin = duration * 0.015
    box_margin = duration * 0.055
    box_width = duration * 0.025

    # Plot stacked traces
    for idx, osc in enumerate(oscs):
        color = (*rgb, alphas[idx])
        y = outputs[:, osc] + (n - 1 - idx) * offset
        ax.plot(times, y, color=color, linewidth=1.8)
        # oscillator labels
        ax.text(times[0] - label_margin, y[0], rf"$x_{{{osc}}}$", fontsize=10, va='center', ha='right', color='black')

    if body:
        ax.set_ylabel("x Body", fontsize=12)
    else:
        ax.set_ylabel("x Limb", fontsize=12)
    ax.set_title(label, fontsize=12, loc='left')
    ax.set_ylim(-0.5, n * offset + 0.5)
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Trunk / Tail boxes
    if group_boxes is not None:
        for (osc_subset, text, color_box) in group_boxes:
            # indices of oscillators inside current subplot
            subset_ids = [oscs.index(o) for o in osc_subset if o in oscs]
            if len(subset_ids) == 0:
                continue
            # convert stacked indices into y coordinates
            y_positions = [(n - 1 - idx) * offset for idx in subset_ids]
            y_min = min(y_positions) - 0.55
            y_max = max(y_positions) + 0.75
            # place rectangle just outside plot
            x_left = times[0] - box_margin - box_width
            rect = patches.Rectangle((x_left, y_min), box_width, y_max - y_min, linewidth=2, edgecolor=color_box,
                                        facecolor='none', clip_on=False, zorder=20)
            ax.add_patch(rect)
            # centered label
            ax.text(x_left + box_width/2, (y_min + y_max)/2, text, rotation=90, fontsize=12, ha='center', va='center',
                    color=color_box, fontweight='bold', zorder=30)
    
    # Vertical red transition lines
    if transition_times is not None:
        for t in np.atleast_1d(transition_times):
            ax.axvline(t, color='red', linewidth=2, alpha=0.7)

    # Diagonal line: travelling wave
    if wave_peaks is not None and not np.isscalar(drive) and drive is not None and len(wave_peaks) >= 2:
        # wave_peaks: list of (osc, t_peak) for oscillators in this subplot
        # filter to only those oscs present here
        osc_set = set(oscs)
        local_peaks = [(osc, t) for (osc, t) in wave_peaks if osc in osc_set]
        if len(local_peaks) >= 2:
            xs = [t for (_, t) in local_peaks]
            ys = []
            for (osc, t) in local_peaks:
                idx = oscs.index(osc)
                # y center of that oscillator's trace
                t_idx = np.argmin(np.abs(times - t))
                y_signal = outputs[:, osc][t_idx]
                y_center = y_signal + (n - 1 - idx) * offset
                ys.append(y_center)
            ax.plot(xs, ys, color='black', linewidth=2,
                    linestyle='--', marker='o', markersize=5,
                    alpha=0.85, zorder=10, label='Travelling wave')

    # Scale bar (π/3)
    if scale_bar is not None:
        scale_val, scale_label = scale_bar
        x_bar = times[-1] - 0.6
        y0 = n * offset - 0.8
        y1 = y0 + scale_val
        ax.plot([x_bar, x_bar], [y0, y1], color='black', linewidth=3, solid_capstyle='butt')
        ax.text(x_bar + 0.08, (y0 + y1) / 2, scale_label, fontsize=12, va='center')

LINKS_MASSES = np.array([
    0.328768, 0.274101, 0.107688, 0.107688, 0.107688,  # link_body_00 - link_body_04
    0.0433459, 0.107688, 0.107688,                      # link_body_06 - link_body_08
    0.18959,                                             # link_body_10
    0.0194482, 0.164364, 0.0194482, 0.164364,           # link_leg_0_L - link_leg_0_R
    0.0194482, 0.164364, 0.0194482, 0.164364,           # link_leg_1_L - link_leg_1_R
    0.321614, 0.164651,                                  # link_body_05 link_body_09
])
TOTAL_MASS = np.sum(LINKS_MASSES)
N_LINKS = len(LINKS_MASSES)  # 19

def compute_fws(links_positions, times):
    """
    Compute average forward speed from CoM trajectory.
    Forward axis is Y (index 1).

    Parameters
    ----------
    links_positions : np.ndarray, shape (n_steps, n_links, 3)
    times           : np.ndarray, shape (n_steps,)

    Returns
    -------
    fwd_speed : float, [m/s]
    """
    com = np.sum(
        links_positions[:, :N_LINKS, :] * LINKS_MASSES[np.newaxis, :, np.newaxis],
        axis=1
    ) / TOTAL_MASS  # (n_steps, 3)

    dur = times[-1] - times[0]
    fwd_speed = np.abs((com[-1, 1] - com[0, 1]) / dur)
    return fwd_speed



def compute_cot(links_positions, joints_torques, joints_velocities, times):
    """
    Compute Cost of Transport using positive mechanical power only.

    CoT = E / (m * g * d_fwd)  where d_fwd is total CoM forward distance.

    Parameters
    ----------
    links_positions    : np.ndarray, shape (n_steps, n_links, 3)
    joints_torques     : np.ndarray, shape (n_steps, n_joints)
    joints_velocities  : np.ndarray, shape (n_steps, n_joints)
    times              : np.ndarray, shape (n_steps,)

    Returns
    -------
    energy : float, [J]
    cot    : float, dimensionless [J / (N·m)]
    """
    dt = times[1] - times[0]

    # Only positive power (no regenerative storage)
    power_positive = np.maximum(joints_torques * joints_velocities, 0)
    energy = dt * np.sum(power_positive)

    # CoM trajectory
    com = np.sum(
        links_positions[:, :N_LINKS, :] * LINKS_MASSES[np.newaxis, :, np.newaxis],
        axis=1
    ) / TOTAL_MASS  # (n_steps, 3)

    # Integrate total forward distance (Y axis)
    d_fwd = np.sum(np.abs(np.diff(com[:, 1])))

    if d_fwd < 1e-6:
        return energy, np.nan

    cot = energy / (TOTAL_MASS * 9.81 * d_fwd)
    return energy, cot

def compute_lat_deviation(links_positions, times):
    """
    Compute Lateral Path Deviation as the RMS perpendicular distance from the
    ideal straight-line trajectory connecting start and end of steady-state motion.

    A perfectly straight gait yields D_lat = 0; larger values indicate lateral
    wandering and reduced locomotor stability.

    Parameters
    ----------
    links_positions : np.ndarray, shape (n_steps, n_links, 3)
    times           : np.ndarray, shape (n_steps,)

    Returns
    -------
    lateral_path_deviation_rms : float, [m]
    """
    # Steady-state selection: discard the first 20% of samples (transient) 
    N = len(times)
    start_idx = N // 5

    # Centre-of-mass (CoM) trajectory via mass-weighted link positions
    com = np.sum(
        links_positions[:, :N_LINKS, :] * LINKS_MASSES[np.newaxis, :, np.newaxis],
        axis=1
    ) / TOTAL_MASS  # (n_steps, 3)

    # Steady-state 2D horizontal position: X = lateral (index 0), Y = forward (index 1)
    x = com[start_idx:, 0]
    y = com[start_idx:, 1]

    # Ideal straight-line path: line through P_start and P_end
    P_start = np.array([x[0],  y[0]])
    P_end   = np.array([x[-1], y[-1]])

    direction   = P_end - P_start          # vector along the ideal path
    path_length = np.linalg.norm(direction) # ||P_end - P_start||

    if path_length < 1e-9:
        # Robot did not translate; deviation is undefined -> return 0
        return 0.0

    # Perpendicular (point-to-line) distance for every sample 
    dx, dy = direction
    d = np.abs((x - P_start[0]) * dy - (y - P_start[1]) * dx) / path_length

    # RMS deviation: D_lat = sqrt( mean(d_i^2) ) 
    # higher weight to large deviations
    lateral_path_deviation_rms = np.sqrt(np.mean(d ** 2))

    return lateral_path_deviation_rms


def compute_phase_locking_error(phi_axial, phi_limb, psi_desired):
    """
    Compute mean-absolute and RMS phase-locking error between limb and axial
    CPG oscillators.

    The metric quantifies how well the limb oscillators maintain the desired
    phase relationship psi_desired with respect to the axial oscillators.
    Perfect coordination yields error = 0.

    Parameters
    ----------
    phi_axial   : np.ndarray, shape (N,)  – axial oscillator phase [rad]
    phi_limb    : np.ndarray, shape (N,)  – limb oscillator phase [rad]
    psi_desired : float                   – desired phase lag [rad]

    Returns
    -------
    phase_locking_error_mean : float, [rad]
    phase_locking_error_rms  : float, [rad]
    """
    # Steady-state selection: discard the first 20% of samples (transient) 
    N = len(phi_axial)
    start_idx = N // 5

    phi_axial_ss = phi_axial[start_idx:]
    phi_limb_ss  = phi_limb[start_idx:]

    # 1. Instantaneous phase difference 
    delta_phi = phi_limb_ss - phi_axial_ss

    # 2. Wrap delta_phi to [-pi, pi] 
    delta_phi_wrapped = (delta_phi + np.pi) % (2.0 * np.pi) - np.pi

    # 3. Residual error after removing the desired phase offset 
    error_raw = delta_phi_wrapped - psi_desired
    error_phi = (error_raw + np.pi) % (2.0 * np.pi) - np.pi

    # 4. Summary statistics 
    phase_locking_error_mean = np.mean(np.abs(error_phi))

    # RMS Error: penalising large transient phase slips more than MAE
    phase_locking_error_rms  = np.sqrt(np.mean(error_phi ** 2))

    return phase_locking_error_mean, phase_locking_error_rms


def compute_heading_variance(links_positions, times):
    """
    Compute the circular variance of the robot's heading (yaw angle) during
    steady-state locomotion.

    Yaw is estimated from the direction of the CoM velocity vector in the
    horizontal plane: yaw(t) = arctan2(v_x(t), v_y(t)), where Y is the
    forward axis and X is the lateral axis (consistent with compute_fws).

    Standard variance cannot be applied to angles because yaw wraps at ±pi.
    Circular statistics (Mardia & Jupp, 2000) are used instead.

    Parameters
    ----------
    links_positions : np.ndarray, shape (n_steps, n_links, 3)
    times           : np.ndarray, shape (n_steps,)

    Returns
    -------
    heading_circular_variance : float, dimensionless in [0, 1]
        0 = perfectly constant heading; values near 1 = highly dispersed heading.
    """
    # --- Centre-of-mass trajectory (same formula as compute_fws) -------------
    com = np.sum(
        links_positions[:, :N_LINKS, :] * LINKS_MASSES[np.newaxis, :, np.newaxis],
        axis=1
    ) / TOTAL_MASS  # (n_steps, 3)

    # --- Instantaneous yaw from CoM velocity direction -----------------------
    # Finite-difference velocity in the horizontal plane (X = lateral, Y = forward).
    # np.gradient uses central differences, giving one value per sample.
    v_x = np.gradient(com[:, 0], times)
    v_y = np.gradient(com[:, 1], times)

    # yaw(t) = direction of travel in the XY plane; arctan2 maps to (-pi, pi]
    yaw = np.arctan2(v_x, v_y)

    # --- Steady-state selection: discard the first 20% of samples (transient) ---
    N = len(times)
    start_idx = N // 5

    yaw_ss = yaw[start_idx:]

    # 1. Mean cosine and sine components
    # Projecting angles onto the unit circle before averaging avoids the
    # wrap-around problem of direct angle arithmetic.
    C = np.mean(np.cos(yaw_ss))
    S = np.mean(np.sin(yaw_ss))

    # 2. Mean resultant length
    R = np.sqrt(C ** 2 + S ** 2)

    # 3. Circular variance: V = 1 - R  in [0, 1]
    heading_circular_variance = 1.0 - R

    return heading_circular_variance

def plot_spine_phase_lag(times, joint_angles, joint_velocities, plot = True, steady_state_time = 2.0, smooth_window = 50):
    n_lags = 7
    spine_phase_lags = np.zeros((joint_angles.shape[0], n_lags))
    mean_lag = np.zeros(n_lags)
    
    # Isolate the index where the robot has reached a steady-state gait
    steady_idx = np.argmax(times >= steady_state_time)
    
    for i in range(n_lags):
        phi_current = np.arctan2(joint_velocities[:, i],  joint_angles[:, i])
        phi_next = np.arctan2(joint_velocities[:, i + 1], joint_angles[:, i + 1])
        
        # Wrap the difference strictly into [-pi, pi]
        diff = (phi_next - phi_current + np.pi) % (2 * np.pi) - np.pi
        spine_phase_lags[:, i] = diff
        
        # Calculate the mathematical Circular Mean on the steady-state portion only
        steady_diff = diff[steady_idx:]
        mean_lag[i] = np.arctan2(np.mean(np.sin(steady_diff)), np.mean(np.cos(steady_diff)))

    if plot:
        fig, axes =plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        
        for i in range(n_lags):
            # Apply a rolling average filter purely for visualization
            smoothed_lag = uniform_filter1d(spine_phase_lags[:, i], size=smooth_window)
            axes[0].plot(times, smoothed_lag, label=f'Lag {i} to {i+1}', alpha=0.85)
        for i in range(n_lags + 1):
            axes[1].plot(times, joint_angles[:, i], label=f'Spine {i}', alpha=0.85)
            
        axes[1].set_xlabel('Time [s]')
        axes[0].set_ylabel('Phase Lag [rad]')
        axes[0].set_title('Spine Joint Phase Lags (Smoothed Time Series)')
        
        axes[1].set_ylabel('Angle [rad]')
        axes[1].set_title(f'Spine Joints')
        axes[1].grid(True, alpha=0.3)
        
        # Format the Y-axis to cleanly display Pi intervals
        axes[0].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
                            [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        axes[0].set_ylim(-np.pi - 0.2, np.pi + 0.2)
        
        # Move legend outside the plot so it doesn't cover the data lines
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
    return mean_lag