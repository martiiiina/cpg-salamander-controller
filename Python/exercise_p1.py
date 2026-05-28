"""[Project1] Exercise 1: Implement & run network without MuJoCo"""

import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from farms_core import pylog
from salamandra_simulation.data import SalamandraState
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork


@dataclass
class DataState:
    state: SalamandraState


def run_network(duration, update=False, drive=0, timestep=1e-2):
    """ Run network without MuJoCo and plot results
    Parameters
    ----------
    duration: <float>
        Duration in [s] for which the network should be run
    update: <bool>
        True: use the prescribed drive parameter, False: update the drive during the simulation
    drive: <float/array>
        Central drive to the oscillators
    """
    # Simulation setup
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    
    # Amplitude gradient from head to tail
    n_body_joints = 8
    body_gradient = np.linspace(0.5, 1.0, n_body_joints)
    limb_gradient = np.ones(8)
    full_gradient = np.concatenate([body_gradient, limb_gradient])

    sim_parameters = SimulationParameters(
        drive=drive[0] if not np.isscalar(drive) else drive,
        amplitude_gradient=None,
        phase_lag_body=None,
        # Feel free to include parameters
    )
    pylog.warning(
        'Modify the scalar drive to be a vector of length n_iterations. By doing so the drive will be modified to be drive[i] at each time step i.')
    state = SalamandraState.salamandra_robot(n_iterations, n_oscillators=32)
    network = SalamandraNetwork(
        sim_parameters,
        n_iterations,
        DataState(
            state=state))
        
    # Logs
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases(iteration=0))
    ])
    phases_log[0, :] = network.state.phases(iteration=0)
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)
    freqs_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
    freqs_log[0, :] = network.robot_parameters.freqs

    # comment below pass to run file
    #pylog.warning('Remove the pass to run your code!!')
    #pass

    pylog.warning(
        'Implement plots here, try to plot the various logged data to check the implementation')
    # Run network ODE and log data
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        if update:
            network.robot_parameters.update(
                SimulationParameters(drive=drive[i+1]
                )
            )
        network.step(i, time0, timestep)
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
        freqs_log[i+1, :] = network.robot_parameters.freqs
    toc = time.time()

    # Network performance
    pylog.info('Time to run simulation for {} steps: {} [s]'.format(
        n_iterations,
        toc - tic
    ))

    # Implement plots of network results
    #pylog.warning('Implement plots')

    # Compute muscle outputs x_i = r_i * (1 + cos(phi_i))
    outputs = amplitudes_log * (1 + np.cos(phases_log))

    def plot_stacked_group(
        ax,
        times,
        outputs,
        oscs,
        base_color,
        label,
        group_boxes=None,
        scale_bar=None,
        transition_times=None,
    ):

        import matplotlib.colors as mcolors
        import matplotlib.patches as patches

        n = len(oscs)
        offset = 1.2

        rgb = mcolors.to_rgb(base_color)

        # Smooth alpha gradient for any number of oscillators
        alphas = np.linspace(1.0, 0.45, n)

        # ---------------------------------------------------
        # Plot stacked traces
        # ---------------------------------------------------
        for idx, osc in enumerate(oscs):

            color = (*rgb, alphas[idx])

            y = outputs[:, osc] + (n - 1 - idx) * offset

            ax.plot(
                times,
                y,
                color=color,
                linewidth=1.8,
            )

            # oscillator labels
            ax.text(
                times[0] - 0.03,
                y[0],
                rf"$x_{osc}$",
                fontsize=10,
                va='center',
                ha='right',
                color='black',
            )

        # ---------------------------------------------------
        # Cosmetics
        # ---------------------------------------------------
        ax.set_ylabel("x Body", fontsize=12)
        ax.set_title(label, fontsize=12, loc='left')
        ax.set_ylim(-0.5, n * offset + 0.5)
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Trunk / Tail boxes
        # ---------------------------------------------------
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
                x_left = times[0] - 0.95
                width = 0.42
                rect = patches.Rectangle(
                    (x_left, y_min),
                    width,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor=color_box,
                    facecolor='none',
                    clip_on=False,
                    zorder=20,
                )
                ax.add_patch(rect)
                # centered label
                ax.text(
                    x_left + width/2,
                    (y_min + y_max)/2,
                    text,
                    rotation=90,
                    fontsize=12,
                    ha='center',
                    va='center',
                    color=color_box,
                    fontweight='bold',
                    zorder=30,
                )
        # ---------------------------------------------------
        # Vertical red transition lines
        # ---------------------------------------------------
        if transition_times is not None:
            for t in transition_times:
                ax.axvline(
                    t,
                    color='red',
                    linewidth=2,
                    alpha=0.7,
                )
        # ---------------------------------------------------
        # Scale bar (π/3)
        # ---------------------------------------------------
        if scale_bar is not None:
            scale_val, scale_label = scale_bar
            x_bar = times[-1] - 0.6
            y0 = n * offset - 0.8
            y1 = y0 + scale_val
            ax.plot(
                [x_bar, x_bar],
                [y0, y1],
                color='black',
                linewidth=3,
                solid_capstyle='butt',
            )
            ax.text(
                x_bar + 0.08,
                (y0 + y1) / 2,
                scale_label,
                fontsize=12,
                va='center',
            )

    from scipy.signal import find_peaks
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
    # =========================================================
    # BODY + LIMBS : 2x2 GRID
    # =========================================================
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
        sharex=True,
    )

    # ---------------------------------------------------------
    # TOP LEFT : LEFT BODY
    # ---------------------------------------------------------

    plot_stacked_group(
        axes[0, 0],
        times,
        outputs,
        [0,2,4,6,8,10,12,14],
        '#1f6fbf',
        'Left body',
        group_boxes=[
            ([0,2,4,6], 'Trunk', '#1f6fbf'),
            ([8,10,12,14], 'Tail', '#1a8c4e'),
        ],
        transition_times=transition_times,
        scale_bar=(np.pi/3, r'$\pi/3$'),
    )

    # ---------------------------------------------------------
    # TOP RIGHT : RIGHT BODY
    # ---------------------------------------------------------

    plot_stacked_group(
        axes[0, 1],
        times,
        outputs,
        [1,3,5,7,9,11,13,15],
        '#4a90e2',
        'Right body',
        group_boxes=[
            ([1,3,5,7], 'Trunk', '#1f6fbf'),
            ([9,11,13,15], 'Tail', '#1a8c4e'),
        ],
        transition_times=transition_times,
        scale_bar=(np.pi/3, r'$\pi/3$'),
    )

    # ---------------------------------------------------------
    # BOTTOM LEFT : LEFT LIMBS
    # ---------------------------------------------------------

    plot_stacked_group(
        axes[1, 0],
        times,
        outputs,
        [16,17,18,19,24,25,26,27],
        '#d4620a',
        'Left limbs',
        group_boxes=[
            ([16,17,18,19], 'Fore', '#d4620a'),
            ([24,25,26,27], 'Hind', '#8e44ad'),
        ],
        transition_times=transition_times,
        scale_bar=(np.pi/3, r'$\pi/3$'),
    )

    # ---------------------------------------------------------
    # BOTTOM RIGHT : RIGHT LIMBS
    # ---------------------------------------------------------

    plot_stacked_group(
        axes[1, 1],
        times,
        outputs,
        [20,21,22,23,28,29,30,31],
        '#c0392b',
        'Right limbs',
        group_boxes=[
            ([20,21,22,23], 'Fore', '#c0392b'),
            ([28,29,30,31], 'Hind', '#6c3483'),
        ],
        transition_times=transition_times,
        scale_bar=(np.pi/3, r'$\pi/3$'),
    )

    # ---------------------------------------------------------
    # AXIS LABELS
    # ---------------------------------------------------------

    axes[1,0].set_xlabel('Time [s]', fontsize=12)
    axes[1,1].set_xlabel('Time [s]', fontsize=12)

    # ---------------------------------------------------------
    # GLOBAL TITLE
    # ---------------------------------------------------------

    fig.suptitle(
        'Salamandra CPG muscle outputs',
        fontsize=18,
        fontweight='bold',
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # --- Figure 4: Intra-limb phase diagnostics ---
    def wpd(a, b): return np.angle(np.exp(1j*(a - b)))

    theta = phases_log
    fig4, axes4 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes4[0].plot(times, wpd(theta[:,18], theta[:,16]), color='purple', label=r'$\theta_{18}-\theta_{16}$ (knee vs girdle)')
    axes4[0].axhline(-np.pi/2, color='red', linestyle='--', alpha=0.7, label=r'Target $-\pi/2$')
    axes4[0].set_ylim([-np.pi, np.pi]); axes4[0].set_ylabel('Phase [rad]')
    axes4[0].set_title('Intra-limb: girdle vs knee'); axes4[0].legend(fontsize=8); axes4[0].grid(alpha=0.2)

    axes4[1].plot(times, np.abs(wpd(theta[:,17], theta[:,16])), color='orange', label=r'$|\theta_{17}-\theta_{16}|$ girdle antagonists')
    axes4[1].plot(times, np.abs(wpd(theta[:,19], theta[:,18])), color='crimson', linestyle=':', label=r'$|\theta_{19}-\theta_{18}|$ knee antagonists')
    axes4[1].axhline(np.pi, color='green', linestyle='--', alpha=0.7, label=r'Target $\pi$')
    axes4[1].set_ylim([0, np.pi+0.3]); axes4[1].set_ylabel('Phase [rad]'); axes4[1].set_xlabel('Time [s]')
    axes4[1].set_title('Antagonist anti-phase'); axes4[1].legend(fontsize=8); axes4[1].grid(alpha=0.2)
    plt.tight_layout()

    # --- Figure 5: Inter-limb trot diagnostics ---
    fig5, axes5 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes5[0].plot(times, wpd(theta[:,28], theta[:,16]), color='blue',  label=r'$\theta_{28}-\theta_{16}$ FL vs RH (target 0)')
    axes5[0].plot(times, wpd(theta[:,24], theta[:,20]), color='green', label=r'$\theta_{24}-\theta_{20}$ RF vs LH (target 0)')
    axes5[0].axhline(0, color='gray', linestyle='--', alpha=0.7)
    axes5[0].set_ylim([-np.pi, np.pi]); axes5[0].set_ylabel('Phase [rad]')
    axes5[0].set_title('Diagonal pairs — should be in-phase'); axes5[0].legend(fontsize=8); axes5[0].grid(alpha=0.2)

    axes5[1].plot(times, wpd(theta[:,20], theta[:,16]), color='red',    label=r'$\theta_{20}-\theta_{16}$ FL vs RF (target $\pi$)')
    axes5[1].plot(times, wpd(theta[:,24], theta[:,16]), color='orange',  label=r'$\theta_{24}-\theta_{16}$ FL vs LH (target $\pi$)')
    axes5[1].axhline( np.pi, color='gray', linestyle='--', alpha=0.7)
    axes5[1].axhline(-np.pi, color='gray', linestyle='--', alpha=0.7)
    axes5[1].set_ylim([-np.pi, np.pi]); axes5[1].set_ylabel('Phase [rad]'); axes5[1].set_xlabel('Time [s]')
    axes5[1].set_title('Same-side pairs — should be anti-phase'); axes5[1].legend(fontsize=8); axes5[1].grid(alpha=0.2)
    plt.tight_layout()
    return


def exercise_1a_networks(plot, timestep=1e-2):
    """[Project 1] Exercise 1: """

    run_network(duration=10, drive=2)       # NOTE: FIXED DRIVE FOR PART A

    # Show plots
    if True:
        if plot:
            plt.show()
        else:
            save_figures()
        return


def exercise_1b_networks(plot, timestep=1e-2):
    """Exercise 1b: Drive ramp"""

    duration = 20
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    drive = np.linspace(0, 6, n_iterations)

    run_network(
        duration=duration,
        drive=drive,
        update=True,
        timestep=timestep,
    )

    # Show plots
    if True:
        if plot:
            plt.show()
        else:
            save_figures()
        return


if __name__ == '__main__':
    exercise_1a_networks(plot=not save_plots())
    exercise_1b_networks(plot=not save_plots())

