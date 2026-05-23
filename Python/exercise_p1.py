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
        amplitude_gradient=full_gradient,
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
    
    osc_left = np.arange(0, 16, 2)      # Left oscillators indices
    osc_right = np.arange(1, 16, 2)     # Right oscillators indices
    osc_legs_left = np.concatenate([np.arange(16, 20), np.arange(24, 28)])
    osc_legs_right = np.concatenate([np.arange(20, 24), np.arange(28, 32)])    
    
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

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Body muscle outputs (left side: even indices 0,2,4,...,14)
    ax = axes[0]
    for i in osc_left:
        ax.plot(times, outputs[:, i], label=f'x_{i}')
    ax.set_ylabel('x Body')
    ax.set_title('Left body muscle outputs')
    ax.legend(loc='upper right', fontsize=7, ncol=4)

    # Limb muscle outputs
    ax = axes[1]
    for i in osc_right:
        ax.plot(times, outputs[:, i], label=f'x_{i}')
    ax.set_ylabel('x Body')
    ax.set_title('Rights body muscle outputs')
    ax.legend(loc='upper right', fontsize=7, ncol=4)

    # Instantaneous frequencies (convert from rad/s to Hz)
    ax = axes[2]
    for i in osc_legs_left:
        ax.plot(times, outputs[:, i], label=f'x_{i}')
    ax.set_ylabel('x Limb')
    ax.set_title('Left limb muscle outputs')
    ax.legend(loc='upper right', fontsize=7, ncol=4)

    """
    ax.plot(times, freqs_log[:, 0] / (2*np.pi), label='Body', color='black')
    ax.plot(times, freqs_log[:, 16] / (2*np.pi), label='Limb', color='black', linestyle='--')
    ax.set_ylabel('Freq [Hz]')
    ax.set_title('Oscillator frequencies')

    ax.legend()
    """
    # Drive
    ax = axes[3]
    for i in osc_legs_right:
        ax.plot(times, outputs[:, i], label=f'x_{i}')
    ax.set_ylabel('x Limb')
    ax.set_title('Right limb muscle outputs')
    ax.legend(loc='upper right', fontsize=7, ncol=4)

    """
    if np.isscalar(drive):
        ax.plot(times, np.full_like(times, drive), color='black')
    else:
        ax.plot(times, drive[:n_iterations], color='black')
    ax.set_ylabel('drive d')
    ax.set_xlabel('Time [s]')
    ax.set_title('Drive signal')
    """
    plt.tight_layout()
# =====================================================================
    # PHASE LAG ANALYSIS DIAGNOSTIC PLOT
    # =====================================================================
    # Extract the raw phase arrays for your target oscillators over time
    theta_16 = phases_log[:, 16]  # LF Girdle Extensor
    theta_17 = phases_log[:, 17]  # LF Girdle Flexor / Antagonist
    theta_18 = phases_log[:, 18]  # LF Knee Extensor
    theta_19 = phases_log[:, 19]  # LF Knee Flexor / Antagonist

    # Calculate wrapped phase differences: Delta_Phi = Wrapped(theta_j - theta_i)
    def wrapped_phase_difference(theta_j, theta_i):
        complex_phase_diff = np.exp(1j * (theta_j - theta_i))
        return np.angle(complex_phase_diff)

    # 1. Intra-limb Girdle-to-Knee Lag (Should converge near -pi/2 or -1.57)
    girdle_knee_lag = wrapped_phase_difference(theta_18, theta_16)

    # 2. Antagonist Pair Phase Lag (Should converge near pi or 3.14)
    girdle_antagonist_lag = wrapped_phase_difference(theta_17, theta_16)
    knee_antagonist_lag = wrapped_phase_difference(theta_19, theta_18)

    # Create diagnostic visualization window
    fig_diag, axes_diag = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Panel A: Girdle to Knee Spatial Lag Tracking
    ax = axes_diag[0]
    ax.plot(times, girdle_knee_lag, color='purple', linewidth=2, label=r'$\theta_{18} - \theta_{16}$ (Knee vs Girdle)')
    ax.axhline(y=-np.pi/2, color='red', linestyle='--', alpha=0.7, label=r'Target $-\pi/2$')
    ax.set_ylabel('Phase Lag [rad]')
    ax.set_ylim([-np.pi, np.pi])
    ax.set_title('Intra-Limb Spatial Phase Locking Progression')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel B: Antagonist Symmetric Phase Locking
    ax = axes_diag[1]
    ax.plot(times, np.abs(girdle_antagonist_lag), color='orange', label=r'$|\theta_{17} - \theta_{16}|$ (Girdle Antagonists)')
    ax.plot(times, np.abs(knee_antagonist_lag), color='crimson', linestyle=':', label=r'$|\theta_{19} - \theta_{18}|$ (Knee Antagonists)')
    ax.axhline(y=np.pi, color='green', linestyle='--', alpha=0.7, label=r'Target $\pi$')
    ax.set_ylabel('Phase Lag [rad]')
    ax.set_xlabel('Time [s]')
    ax.set_ylim([0, np.pi + 0.5])
    ax.set_title('Antagonist Co-Axial Anti-Phase Synchronization')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # =====================================================================
# INTER-LIMB TROT PATTERN DIAGNOSTIC
# =====================================================================
    theta_20 = phases_log[:, 20]  # RF Girdle
    theta_24 = phases_log[:, 24]  # LH Girdle  
    theta_28 = phases_log[:, 28]  # RH Girdle

    # Trot: FL(16) and RH(28) should be in phase → difference ≈ 0
    # Trot: RF(20) and LH(24) should be in phase → difference ≈ 0
    # Trot: FL(16) and RF(20) should be antiphase → difference ≈ π
    # Trot: FL(16) and LH(24) should be antiphase → difference ≈ π

    fl_rh_lag = wrapped_phase_difference(theta_28, theta_16)   # target 0
    rf_lh_lag = wrapped_phase_difference(theta_24, theta_20)   # target 0
    fl_rf_lag = wrapped_phase_difference(theta_20, theta_28)   # target π
    fl_lh_lag = wrapped_phase_difference(theta_24, theta_16)   # target π

    fig_trot, axes_trot = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes_trot[0]
    ax.plot(times, fl_rh_lag, color='blue',   linewidth=2, label=r'$\theta_{28} - \theta_{16}$ (FL vs RH, target 0)')
    ax.plot(times, rf_lh_lag, color='green',  linewidth=2, label=r'$\theta_{24} - \theta_{20}$ (RF vs LH, target 0)')
    ax.axhline(y=0,      color='blue',  linestyle='--', alpha=0.7, label='Target 0')
    ax.set_ylabel('Phase Lag [rad]')
    ax.set_ylim([-np.pi, np.pi])
    ax.set_title('Diagonal Pairs (should be in-phase for trot)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes_trot[1]
    ax.plot(times, fl_rf_lag, color='red',    linewidth=2, label=r'$\theta_{20} - \theta_{28}$ (FL vs RF, target π)')
    ax.plot(times, fl_lh_lag, color='orange', linewidth=2, label=r'$\theta_{24} - \theta_{16}$ (FL vs LH, target π)')
    ax.axhline(y=np.pi,  color='red',   linestyle='--', alpha=0.7, label='Target π')
    ax.axhline(y=-np.pi, color='red',   linestyle='--', alpha=0.7)
    ax.set_ylabel('Phase Lag [rad]')
    ax.set_xlabel('Time [s]')
    ax.set_ylim([-np.pi, np.pi])
    ax.set_title('Same-side Pairs (should be antiphase for trot)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return


def exercise_1a_networks(plot, timestep=1e-2):
    """[Project 1] Exercise 1: """

    run_network(duration=10, drive=3)       # NOTE: FIXED DRIVE FOR PART A

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

