import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from scipy.signal import find_peaks

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

def plot_stacked_group(ax, times, outputs, oscs, base_color, label, group_boxes=None, 
                       scale_bar=None, transition_times=None, wave_peaks=None, drive=None):
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
        ax.text(times[0] - label_margin, y[0], rf"$x_{osc}$", fontsize=10, va='center', ha='right', color='black')

    ax.set_ylabel("x Body", fontsize=12)
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