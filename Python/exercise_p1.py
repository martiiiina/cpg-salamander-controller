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
    
    osc_left = np.arange(0, 16, 2)      # Left oscillators indices
    osc_right = np.arange(1, 16, 2)     # Right oscillators indices
    osc_legs = np.arange(16, 32)        # Leg oscillators indices

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
    for i in osc_legs:
        ax.plot(times, outputs[:, i], label=f'x_{i}')
    ax.set_ylabel('x Limb')
    ax.set_title('Limb muscle outputs')
    ax.legend(loc='upper right', fontsize=7, ncol=4)

    # Instantaneous frequencies (convert from rad/s to Hz)
    ax = axes[2]
    ax.plot(times, freqs_log[:, 0] / (2*np.pi), label='Body', color='black')
    ax.plot(times, freqs_log[:, 16] / (2*np.pi), label='Limb', color='black', linestyle='--')
    ax.set_ylabel('Freq [Hz]')
    ax.set_title('Oscillator frequencies')
    ax.legend()

    # Drive
    ax = axes[3]
    if np.isscalar(drive):
        ax.plot(times, np.full_like(times, drive), color='black')
    else:
        ax.plot(times, drive[:n_iterations], color='black')
    ax.set_ylabel('drive d')
    ax.set_xlabel('Time [s]')
    ax.set_title('Drive signal')

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

