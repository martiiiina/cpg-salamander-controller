"""[Project1] Exercise 1: Implement & run network without MuJoCo"""

import time
from dataclasses import dataclass

import numpy as np
import os
import matplotlib.pyplot as plt
from utils import (
    plot_stacked_group,
    find_peak,
    find_travelling_wave_peaks,
    compute_instantaneous_frequency,
)
from farms_core import pylog
from salamandra_simulation.data import SalamandraState
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork


@dataclass
class DataState:
    state: SalamandraState


def run_network(duration, update=False, drive=0, timestep=1e-2, decouple=False, amplitude_gradient=None, phase_lag_body=None):
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
    
    if decouple:
        sim_parameters = SimulationParameters(
            drive=drive[0] if not np.isscalar(drive) else drive,
            amplitude_gradient=amplitude_gradient,
            phase_lag_body=phase_lag_body,
            w_limb_body=0,
            limb_body_phase_offset=0,
            body_gain=1,
            limb_gain=1
        )
    else:
        sim_parameters = SimulationParameters(
            drive=drive[0] if not np.isscalar(drive) else drive,
            amplitude_gradient=amplitude_gradient,
            phase_lag_body=phase_lag_body,
            w_limb_body=150,
            limb_body_phase_offset=0,
            body_gain=1,
            limb_gain=1
        )
    #pylog.warning(
    #    'Modify the scalar drive to be a vector of length n_iterations. By doing so the drive will be modified to be drive[i] at each time step i.')
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
    # comment below pass to run file
    #pylog.warning('Remove the pass to run your code!!')
    #pass

    #pylog.warning(
    #    'Implement plots here, try to plot the various logged data to check the implementation')
    # Run network ODE and log data
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        if update:
            sim_parameters.drive = drive[i+1]
            network.robot_parameters.update(sim_parameters)
        network.step(i, time0, timestep)
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
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


    ##########################
    # VISUALIZATION PLOTS 
    ##########################

    transition_times = find_peak(drive, outputs, times)
    # Find travelling wave peaks for body oscillators after second transition
    body_oscs_left  = [0,2,4,6,8,10,12,14]
    body_oscs_right = [1,3,5,7,9,11,13,15]

    if not np.isscalar(drive) and len(transition_times) > 1:
        wave_peaks_left  = find_travelling_wave_peaks(outputs, times, body_oscs_left,  after_time=transition_times[1])
        wave_peaks_right = find_travelling_wave_peaks(outputs, times, body_oscs_right, after_time=transition_times[1])
    else:
        wave_peaks_left  = None
        wave_peaks_right = None

    # Figure 1: body+limbs outputs + frequencies + drive - 3x2 GRID
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), sharex=True)

    drive_markers = [1.0, 3.0, 5.0]
    if np.isscalar(drive):
        marker_times = []
    else:
        drive_series = np.asarray(drive, dtype=float)
        marker_times = [
            times[np.argmin(np.abs(drive_series - marker))]
            for marker in drive_markers
            if drive_series.min() <= marker <= drive_series.max()
        ]

    def add_drive_markers(*plot_axes):
        for ax in plot_axes:
            for marker_time in marker_times:
                ax.axvline(marker_time, color='0.6', linestyle='--', linewidth=1.0, alpha=0.8, zorder=0)

    plot_stacked_group(axes[0, 0], times, outputs, body_oscs_left,'#1f6fbf', 'Left body',
                        group_boxes=[([0,2,4,6], 'Trunk', '#1f6fbf'),([8,10,12,14], 'Tail', '#1a8c4e'),],
                        transition_times=transition_times[0], scale_bar=(np.pi/3, r'$\pi/3$'), wave_peaks=wave_peaks_left, drive=drive, body=True)
    plot_stacked_group(axes[0, 1], times, outputs, body_oscs_right, '#4a90e2', 'Right body',
                        group_boxes=[([1,3,5,7], 'Trunk', '#1f6fbf'),([9,11,13,15], 'Tail', '#1a8c4e'),],
                        transition_times=transition_times[0], scale_bar=(np.pi/3, r'$\pi/3$'), wave_peaks=wave_peaks_right, drive=drive)
    plot_stacked_group(axes[1, 0], times, outputs, [16,17,18,19,24,25,26,27], '#d4620a', 'Left limbs',
                        group_boxes=[([16,17,18,19], 'Fore', '#d4620a'),([24,25,26,27], 'Hind', '#8e44ad'),],
                        transition_times=transition_times[0],scale_bar=(np.pi/3, r'$\pi/3$'), drive=drive, body=False)
    plot_stacked_group(axes[1, 1], times, outputs, [20,21,22,23,28,29,30,31], '#c0392b', 'Right limbs',
                        group_boxes=[([20,21,22,23], 'Fore', '#c0392b'),([28,29,30,31], 'Hind', '#6c3483'),],
                        transition_times=transition_times[0], scale_bar=(np.pi/3, r'$\pi/3$'), drive=drive, body=False)

    freqs_log = compute_instantaneous_frequency(times, phases_log)
    n_freqs = freqs_log.shape[1]
    freq_colors = plt.cm.viridis(np.linspace(0.05, 0.95, n_freqs))
    freq_alphas = np.linspace(0.9, 0.35, n_freqs)
    for idx in range(n_freqs):
        axes[2, 0].plot(
            times,
            freqs_log[:, idx],
            color=(*freq_colors[idx][:3], freq_alphas[idx]),
            linewidth=1.1,
        )
    axes[2, 0].set_title('Frequencies', fontsize=12, loc='left')
    axes[2, 0].set_ylabel('Frequency [Hz]', fontsize=12)
    axes[2, 0].spines['right'].set_visible(False)
    axes[2, 0].spines['top'].set_visible(False)

    if np.isscalar(drive):
        drive_series = np.full_like(times, float(drive))
    else:
        drive_series = np.asarray(drive, dtype=float)
        if drive_series.shape[0] != len(times):
            drive_series = np.interp(
                times,
                np.linspace(times[0], times[-1], drive_series.shape[0]),
                drive_series,
            )
    axes[2, 1].plot(times, drive_series, color='#333333', linewidth=1.8)
    axes[2, 1].set_title('Drive', fontsize=12, loc='left')
    axes[2, 1].set_ylabel('Drive', fontsize=12)
    axes[2, 1].spines['right'].set_visible(False)
    axes[2, 1].spines['top'].set_visible(False)

    add_drive_markers(*axes.flat)

    axes[2, 0].set_xlabel('Time [s]', fontsize=12)
    axes[2, 1].set_xlabel('Time [s]', fontsize=12)
    fig.suptitle('Salamandra CPG outputs, frequencies, and drive', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if np.isscalar(drive):
        out_dir = os.path.join("logs", "ex1a")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, 'cpg_output_fix.png')
        fig.savefig(fname, bbox_inches='tight')
    else:
        out_dir = os.path.join("logs", "ex1b")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, 'cpg_output_ramp.png')
        fig.savefig(fname, bbox_inches='tight')

    def wpd(a, b):
        """Wrapped phase difference in [-π, π]"""
        return np.angle(np.exp(1j * (a - b)))

    def wpd_pos(a, b):
        """Wrapped phase difference in [0, 2π] — use when target > 0"""
        return np.mod(a - b, 2 * np.pi)    
    
    theta = phases_log

    # Figure 2 - Intra-limb and antagonist diagnostics
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes2[0].plot(times, wpd(theta[:,18], theta[:,16]), color='purple', label=r'$\theta_{18}-\theta_{16}$ (knee vs girdle)')
    axes2[0].axhline(-np.pi/2, color='red', linestyle='--', alpha=0.7, label=r'Target $-\pi/2$')
    axes2[0].set_ylim([-np.pi, np.pi]); axes2[0].set_ylabel('Phase [rad]')
    axes2[0].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axes2[0].set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    axes2[0].set_title('Intra-limb: girdle vs knee'); axes2[0].legend(fontsize=8); axes2[0].grid(alpha=0.2)

    axes2[1].plot(times, np.abs(wpd_pos(theta[:,17], theta[:,16])), color='orange', label=r'$|\theta_{17}-\theta_{16}|$ girdle antagonists')
    axes2[1].plot(times, np.abs(wpd_pos(theta[:,19], theta[:,18])), color='crimson', linestyle=':', label=r'$|\theta_{19}-\theta_{18}|$ knee antagonists')
    axes2[1].axhline(np.pi, color='green', linestyle='--', alpha=0.7, label=r'Target $\pi$')
    axes2[1].set_ylim([0, 2*np.pi]); axes2[1].set_ylabel('Phase [rad]'); axes2[1].set_xlabel('Time [s]')
    axes2[1].set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    axes2[1].set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    axes2[1].set_title('Antagonist anti-phase'); axes2[1].legend(fontsize=8); axes2[1].grid(alpha=0.2)
    plt.tight_layout()

    if np.isscalar(drive):
        out_dir = os.path.join("logs", "ex1a")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, 'intralimb_fix.png')
        fig2.savefig(fname, bbox_inches='tight')
    else:
        out_dir = os.path.join("logs", "ex1b")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, 'intralimb_ramp.png')
        fig2.savefig(fname, bbox_inches='tight')

    # Figure 3: Inter-limb trot diagnostics 
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes3[0].plot(times, wpd(theta[:,28], theta[:,16]), color='blue',  label=r'$\theta_{28}-\theta_{16}$ FL vs RH (target 0)')
    axes3[0].plot(times, wpd(theta[:,24], theta[:,20]), color='green', label=r'$\theta_{24}-\theta_{20}$ RF vs LH (target 0)')
    axes3[0].axhline(0, color='gray', linestyle='--', alpha=0.7)
    axes3[0].set_ylim([-np.pi, np.pi]); axes3[0].set_ylabel('Phase [rad]')
    axes3[0].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axes3[0].set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    axes3[0].set_title('Diagonal pairs — should be in-phase'); axes3[0].legend(fontsize=8); axes3[0].grid(alpha=0.2)

    axes3[1].plot(times, wpd_pos(theta[:,20], theta[:,16]), color='red',    label=r'$\theta_{20}-\theta_{16}$ FL vs RF (target $\pi$)')
    axes3[1].plot(times, wpd_pos(theta[:,24], theta[:,16]), color='orange',  label=r'$\theta_{24}-\theta_{16}$ FL vs LH (target $\pi$)')
    axes3[1].axhline( np.pi, color='gray', linestyle='--', alpha=0.7)
    axes3[1].set_ylim([0, 2*np.pi]); axes3[1].set_ylabel('Phase [rad]'); axes3[1].set_xlabel('Time [s]')
    axes3[1].set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    axes3[1].set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    axes3[1].set_title('Same-side pairs — should be anti-phase'); axes3[1].legend(fontsize=8); axes3[1].grid(alpha=0.2)
    plt.tight_layout()

    if np.isscalar(drive):
        out_dir = os.path.join("logs", "ex1a")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, 'interlimb_fix.png')
        fig3.savefig(fname, bbox_inches='tight')
    else:
        out_dir = os.path.join("logs", "ex1b")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, 'interlimb_ramp.png')
        fig3.savefig(fname, bbox_inches='tight')

    # Figure 4: Adjacent body segment phase lags
    fig4, axes4 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Left side: phase lag between adjacent segments (0→2, 2→4, ..., 12→14)
    left_oscs  = [0, 2, 4, 6, 8, 10, 12, 14]
    right_oscs = [1, 3, 5, 7, 9, 11, 13, 15]
    colors = plt.cm.viridis(np.linspace(0, 1, len(left_oscs) - 1))

    for i in range(len(left_oscs) - 1):
        a, b = left_oscs[i], left_oscs[i+1]
        axes4[0].plot(times, wpd(theta[:, b], theta[:, a]),
                    color=colors[i], label=rf'$\theta_{{{b}}}-\theta_{{{a}}}$')

    axes4[0].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes4[0].axhline(-np.pi, color='red', linestyle='--', alpha=0.7)
    if not np.isscalar(drive):
        axes4[0].axhline(-np.pi/4, color='red', linestyle='--', alpha=0.7)
    axes4[0].set_ylim([-np.pi-0.3, np.pi+0.3])
    axes4[0].set_yticks([-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi])
    axes4[0].set_yticklabels(['-π', '-π/2', '-π/4', '0', 'π/4', 'π/2', 'π'])
    axes4[0].set_ylabel('Phase lag [rad]')
    axes4[0].set_title('Left body — adjacent segment phase lags')
    axes4[0].legend(fontsize=7, ncol=4, loc='upper right')
    axes4[0].grid(alpha=0.2)

    for i in range(len(right_oscs) - 1):
        a, b = right_oscs[i], right_oscs[i+1]
        axes4[1].plot(times, wpd(theta[:, b], theta[:, a]),
                    color=colors[i], label=rf'$\theta_{{{b}}}-\theta_{{{a}}}$')

    axes4[1].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes4[1].axhline(-np.pi, color='red', linestyle='--', alpha=0.7)
    if not np.isscalar(drive):
        axes4[1].axhline(-np.pi/4, color='red', linestyle='--', alpha=0.7)
    axes4[1].set_ylim([-np.pi-0.3, np.pi+0.3])
    axes4[1].set_yticks([-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi])
    axes4[1].set_yticklabels(['-π', '-π/2', '-π/4', '0', 'π/4', 'π/2', 'π'])
    axes4[1].set_ylabel('Phase lag [rad]')
    axes4[1].set_xlabel('Time [s]')
    axes4[1].set_title('Right body — adjacent segment phase lags')
    axes4[1].legend(fontsize=7, ncol=4, loc='upper right')
    axes4[1].grid(alpha=0.2)

    fig4.suptitle('Adjacent body segment phase lags', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if np.isscalar(drive):
        out_dir = os.path.join("logs", "ex1a")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, 'body_fix.png')
        fig4.savefig(fname, bbox_inches='tight')
    else:
        out_dir = os.path.join("logs", "ex1b")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, 'body_ramp.png')
        fig4.savefig(fname, bbox_inches='tight')

    return


def exercise_1a_networks(plot, timestep=1e-2, amplitude_gradient=None, phase_lag_body=None, drive = 2.5):
    """[Project 1] Exercise 1: """

    run_network(
        duration=20, 
        drive=drive, 
        decouple=False, 
        timestep=timestep, 
        amplitude_gradient=amplitude_gradient, 
        phase_lag_body=phase_lag_body)      

    # Show plots
    if True:
        if plot:
            plt.show()
        else:
            save_plots()
        return


def exercise_1b_networks(plot, timestep=1e-2, amplitude_gradient=None, phase_lag_body=None):
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
        amplitude_gradient=amplitude_gradient,
        phase_lag_body=phase_lag_body,
        decouple=False
    )

    # Show plots
    if True:
        if plot:
            plt.show()
        else:
            save_plots()
        return


if __name__ == '__main__':
    exercise_1a_networks(plot=not save_plots())
    exercise_1b_networks(plot=not save_plots())

    # R_head = 0.25
    # R_tail = 1
    # amplitude_gradient = R_head + (R_tail - R_head) * (np.arange(8) / 7) # Linear gradient from head to tail
    # amplitude_gradient = np.repeat(amplitude_gradient, 2)  # Same gradient for left and right body; limbs have gradient of 1
    # exercise_1a_networks(plot=not save_plots(), drive = 4)
    # exercise_1a_networks(plot=not save_plots(), drive = 4, amplitude_gradient=amplitude_gradient)

