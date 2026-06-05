import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster plotting and headless environments

"""Exercise 3: Limb and Spine Coordination while walking"""

import os
import numpy as np
import matplotlib.pyplot as plt
from exercise_p1 import run_network
from plot_traj import load_simulation
from utils import compute_fws, compute_cot, compute_lat_deviation, compute_phase_locking_error, compute_heading_variance, get_com
from salamandra_simulation.save_figures import save_figures
from salamandra_simulation.simulation import simulation, simulation_sweep
from simulation_parameters import SimulationParameters
import farms_core as pylog


def exercise_3_disable_limb_spine_coupling(timestep):
    """ Walk with disabled limb-spine limbs """
    # Use exercise_example.py for reference
    plot=True

    run_network(duration=10, drive=2, decouple=True) 
    if True:
        if plot:
            plt.show()
        else:
            save_figures()

    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=2,          # fixed drive
        phase_lag_body=None,
        w_limb_body=0,
        limb_body_phase_offset=0
    )
    os.makedirs('./logs/ex3.2_walk/', exist_ok=True)
    sim, data = simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=True,
        output='logs/ex3.2_walk/sim_0',
        record=True,
        record_path='logs/ex3.2_walk/video_walking.mp4',
    )

    # Compute forward speed and cost of transport 
    log_path = './logs/ex3.2_walk/sim_0/simulation.hdf5'
    try:
        links, joints, times, data = load_simulation(log_path)
    except Exception as e:
        print(f"Failed to load simulation: {e}")
        return

    # links_positions   = links[:, :, :3]
    links_positions = get_com(data, times)  # directly use CoM trajectory
    joints_torques    = joints[:, :, 2]
    joints_velocities = joints[:, :, 1]

    fws = compute_fws(links_positions, times)
    _, cot = compute_cot(links_positions, joints_torques, joints_velocities, times)

    print(f"Forward speed: {fws}")
    print(f"Cost of Transport (CoT): {cot}")
    
    # Stability metrics

    lat_deviation = compute_lat_deviation(links_positions, times)

    # Oscillator phases are not saved directly; estimate them from joint data.
    # The instantaneous phase of a harmonic oscillator can be read from the
    # phase plane: phi = arctan2(velocity, position).
    # Joint indices: spine joints 0-7 (axial), limb girdle joints 8/10/12/14.
    # Use joint 0 as the reference axial oscillator and joint 8 (front-left
    # girdle) as the reference limb oscillator.
    joint_angles     = joints[:, :, 0]   # shape (n_steps, n_joints)
    joint_velocities = joints[:, :, 1]
    phi_axial  = np.arctan2(joint_velocities[:, 0],  joint_angles[:, 0])
    phi_limb   = np.arctan2(joint_velocities[:, 8],  joint_angles[:, 8])
    psi_desired = sim_parameters.limb_body_phase_offset   # desired phase lag [rad]
    phase_locking_error_mean, phase_locking_error_rms = compute_phase_locking_error(phi_axial, phi_limb, psi_desired)

    heading_variance = compute_heading_variance(links_positions, times)
    
    print(f"Lateral path deviation (RMS): {lat_deviation:.4f} m")
    print(f"Phase locking error (MAE): {phase_locking_error_mean:.4f} rad")
    print(f"Phase locking error (RMS): {phase_locking_error_rms:.4f} rad")
    print(f"Heading variance: {heading_variance:.4f}")

    return


def exercise_3_limb_spine_antiphase(timestep, best_parameters):
    """ Walk with limb-spine in anti-phase """
    # Use exercise_example.py for reference
    # Simulation in antiphase
    drive, phi_deg = best_parameters
    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=drive,          # fixed drive
        phase_lag_body=None,
        w_limb_body=150,
        limb_body_phase_offset=phi_deg + np.pi
    )
    os.makedirs('./logs/ex3.3_walk/', exist_ok=True)
    sim, data = simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=True,
        output='logs/ex3.3_walk/sim_0',
        record=True,
        record_path='logs/ex3.3_walk/video_walking_antiphase.mp4',
    )



def exercise_3a_coordination(timestep):
    """Exercise 3a Limb and Spine coordination

    This exercise explores how phase difference between spine and legs
    affects locomotion.

    Run the simulations for different walking drives and phase lag between body
    and limb oscillators.

    """
    # Use exercise_example.py for reference
    # # For sweeps with many simulations running in parallel
    # parameter_set = [
    #     SimulationParameters(...)
    #     for ... in ...
    #     for ... in ...
    # ]
    # os.makedirs('./logs/sweep_3a/', exist_ok=True)
    # simulation_sweep([
    #     {
    #         'sim_parameters': sim_parameters,
    #         'arena': 'land',
    #         'fast': True,  # For fast mode (not real-time)
    #         'headless': True,  # For headless mode (No GUI, could be faster)
    #         'output': f'logs/ex3a/simulation_{simulation_i}',
    #         'verbose': False,
    #     }
    #     for simulation_i, sim_parameters in enumerate(parameter_set)
    # ], processes=4)  # Adjust based on your hardware

        # Parameter sweep ranges
    drives         = np.linspace(1.0, 3.0, 10)        # walking drive range [1,3]
    phase_offsets  = np.linspace(-np.pi, np.pi, 20)   # limb-body phase offset

    os.makedirs('./logs/sweep_3.3/', exist_ok=True)

    parameter_set = [
        {
            'sim_parameters': SimulationParameters(
                drive=d,
                duration=15,
                timestep=timestep,
                phase_lag_body=None,         
                w_limb_body=150,
                limb_body_phase_offset=phi,        # parameter to sweep
                spawn_position=[0, 0, 0.1],
                spawn_orientation=[0, 0, np.pi/2],
            ),
            'arena': 'land',
            'fast': True,
            'headless': True,
            'output': f'./logs/sweep_3.3/sim_d{d:.2f}_phi{phi:.2f}',
            'verbose': False,
        }
        for d   in drives
        for phi in phase_offsets
    ]

    simulation_sweep(parameter_set, processes=4)

    # Collect results 
    fwd_speed = np.full((len(drives), len(phase_offsets)), np.nan)
    cot       = np.full((len(drives), len(phase_offsets)), np.nan)

    for i, d in enumerate(drives):
        for j, phi in enumerate(phase_offsets):
            log_path = f'./logs/sweep_3.3/sim_d{d:.2f}_phi{phi:.2f}/simulation.hdf5'
            if not os.path.exists(log_path):
                continue
            try:
                links, joints, times, data = load_simulation(log_path)
            except Exception:
                continue

            links_positions   = get_com(data, times)  # directly use CoM trajectory
            joints_torques    = joints[:, :, 2]
            joints_velocities = joints[:, :, 1]

            fwd_speed[i, j] = compute_fws(links_positions, times)
            try:
                _, cot[i, j] = compute_cot(links_positions, joints_torques, joints_velocities, times)
            except Exception as e:
                print(f"CoT failed for d={d:.2f}, phi={phi:.2f}: {e}")
                cot[i, j] = np.nan

    #  Plots 
    phi_deg = np.degrees(phase_offsets)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(
        fwd_speed, aspect='auto', origin='lower', cmap='viridis',
        extent=[phi_deg[0], phi_deg[-1], drives[0], drives[-1]],
    )
    plt.colorbar(im0, ax=axes[0], label='Forward speed [m/s]')
    axes[0].set_xlabel('Limb-body phase offset [deg]')
    axes[0].set_ylabel('Drive')
    axes[0].set_title('Forward speed')
    best = np.unravel_index(np.nanargmax(fwd_speed), fwd_speed.shape)
    axes[0].plot(phi_deg[best[1]], drives[best[0]], 'r*', markersize=14,
                 label=f'Best: {fwd_speed[best]:.3f} m/s\nφ={phi_deg[best[1]]:.0f}°, d={drives[best[0]]:.1f}')
    axes[0].legend(fontsize=8)

    im1 = axes[1].imshow(
        cot, aspect='auto', origin='lower', cmap='plasma_r',
        extent=[phi_deg[0], phi_deg[-1], drives[0], drives[-1]],
    )
    plt.colorbar(im1, ax=axes[1], label='CoT [-]')
    axes[1].set_xlabel('Limb-body phase offset [deg]')
    axes[1].set_ylabel('Drive')
    axes[1].set_title('Cost of Transport (lower = better)')
    best_cot = np.unravel_index(np.nanargmin(cot), cot.shape)
    axes[1].plot(phi_deg[best_cot[1]], drives[best_cot[0]], 'w*', markersize=14,
                 label=f'Best: {cot[best_cot]:.3f}\nφ={phi_deg[best_cot[1]]:.0f}°, d={drives[best_cot[0]]:.1f}')
    axes[1].legend(fontsize=8)

    fig.suptitle('Exercise 3.3: Limb-body phase offset × Drive sweep', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.cool(np.linspace(0, 1, len(drives)))

    for i, d in enumerate(drives):
        axes2[0].plot(phi_deg, fwd_speed[i], color=colors[i], marker='o', label=f'd={d:.1f}')
        axes2[1].plot(phi_deg, cot[i],        color=colors[i], marker='o', label=f'd={d:.1f}')

    for ax, ylabel, title in zip(
        axes2,
        ['Forward speed [m/s]', 'CoT [-]'],
        ['Forward speed vs phase offset', 'CoT vs phase offset'],
    ):
        ax.set_xlabel('Limb-body phase offset [deg]')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    fig2.suptitle('Exercise 3.3: per-drive curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Simulation with best parameters
    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=drives[best[0]],          # fixed drive
        phase_lag_body=None,
        w_limb_body=150,
        limb_body_phase_offset=np.deg2rad(phi_deg[best[1]])
    )
    os.makedirs('./logs/ex3.3_walk/', exist_ok=True)
    sim, data = simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=True,
        output='logs/ex3.3_walk/sim_0',
        record=True,
        record_path='logs/ex3.3_walk/video_walking.mp4',
    )


    return (drives[best[0]], np.deg2rad(phi_deg[best[1]]))


def exercise_3b_coordination(timestep, best_parameters):
    """Exercise 3b Limb and Spine coordination

    This exercise explores how axial and limb amplitudes affect coordination.

    Run the simulations for different axial and limb amplitudes.

    """
    # Parameter sweep ranges
    axial_gains = np.linspace(0, 3, 20)        
    limb_gains  = np.linspace(0, 3, 20)   
    optimal_limb_body_phase_offset = np.rad2deg(best_parameters[1])  # from previous exercise
    drive = best_parameters[0]  # from previous exercise

    os.makedirs('./logs/sweep_3.4/', exist_ok=True)

    parameter_set = [
        {
            'sim_parameters': SimulationParameters(
                drive=drive,
                duration=15,
                timestep=timestep,
                phase_lag_body=None,         
                w_limb_body=150,
                limb_body_phase_offset=optimal_limb_body_phase_offset,        
                spawn_position=[0, 0, 0.1],
                spawn_orientation=[0, 0, np.pi/2],
                body_gain=bg,
                limb_gain=lg
            ),
            'arena': 'land',
            'fast': True,
            'headless': True,
            'output': f'./logs/sweep_3.4/sim_bg{bg:.2f}_lg{lg:.2f}',
            'verbose': False,
        }
        for bg in axial_gains
        for lg in limb_gains
    ]

    simulation_sweep(parameter_set, processes=4)

    # Collect results 
    fwd_speed = np.full((len(axial_gains), len(limb_gains)), np.nan)
    cot       = np.full((len(axial_gains), len(limb_gains)), np.nan)

    for i, bg in enumerate(axial_gains):
        for j, lg in enumerate(limb_gains):
            log_path = f'./logs/sweep_3.4/sim_bg{bg:.2f}_lg{lg:.2f}/simulation.hdf5'
            if not os.path.exists(log_path):
                continue
            try:
                links, joints, times, data = load_simulation(log_path)
            except Exception:
                continue

            # links_positions   = links[:, :, :3]
            links_positions = get_com(data, times)  # directly use CoM trajectory
            joints_torques    = joints[:, :, 2]
            joints_velocities = joints[:, :, 1]

            fwd_speed[i, j] = compute_fws(links_positions, times)
            try:
                _, cot[i, j] = compute_cot(links_positions, joints_torques, joints_velocities, times)
            except Exception as e:
                print(f"CoT failed for bg={bg:.2f}, lg={lg:.2f}: {e}")
                cot[i, j] = np.nan

    #  Plots 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(
        fwd_speed, aspect='auto', origin='lower', cmap='viridis',
        extent=[limb_gains[0], limb_gains[-1], axial_gains[0], axial_gains[-1]],
    )
    plt.colorbar(im0, ax=axes[0], label='Forward speed [m/s]')
    axes[0].set_xlabel('Limb gain')
    axes[0].set_ylabel('Axial gain')
    axes[0].set_title('Forward speed')
    best = np.unravel_index(np.nanargmax(fwd_speed), fwd_speed.shape)
    axes[0].plot(limb_gains[best[1]], axial_gains[best[0]], 'r*', markersize=14,
                 label=f'Best: {fwd_speed[best]:.3f} \nlg={limb_gains[best[1]]:.2f}, bg={axial_gains[best[0]]:.2f}')
    axes[0].legend(fontsize=8)

    im1 = axes[1].imshow(
        cot, aspect='auto', origin='lower', cmap='plasma_r',
        extent=[limb_gains[0], limb_gains[-1], axial_gains[0], axial_gains[-1]],
    )
    plt.colorbar(im1, ax=axes[1], label='CoT [-]')
    axes[1].set_xlabel('Limb gain')
    axes[1].set_ylabel('Axial gain')
    axes[1].set_title('Cost of Transport (lower = better)')
    best_cot = np.unravel_index(np.nanargmin(cot), cot.shape)
    axes[1].plot(limb_gains[best_cot[1]], axial_gains[best_cot[0]], 'w*', markersize=14,
                 label=f'Best: {cot[best_cot]:.3f}\nlg={limb_gains[best_cot[1]]:.2f}, bg={axial_gains[best_cot[0]]:.2f}')
    axes[1].legend(fontsize=8)

    fig.suptitle('Exercise 3.4: Limb-body limb gain × axial gain', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.cool(np.linspace(0, 1, len(axial_gains)))

    for i, bg in enumerate(axial_gains):
        axes2[0].plot(limb_gains, fwd_speed[i], color=colors[i], marker='o', label=f'bg={bg:.2f}')
        axes2[1].plot(limb_gains, cot[i],        color=colors[i], marker='o', label=f'bg={bg:.2f}')

    for ax, ylabel, title in zip(
        axes2,
        ['Forward speed [m/s]', 'CoT [-]'],
        ['Forward speed vs limb gain', 'CoT vs limb gain'],
    ):
        ax.set_xlabel('Limb gain')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    fig2.suptitle('Exercise 3.4: per-drive curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Simulation with best parameters
    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=drive,          
        phase_lag_body=None,
        w_limb_body=150,
        limb_body_phase_offset=optimal_limb_body_phase_offset,
        body_gain=axial_gains[best[0]],
        limb_gain=limb_gains[best[1]],
    )
    os.makedirs('./logs/ex3.4_walk/', exist_ok=True)
    sim, data = simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=True,
        output='logs/ex3.4_walk/sim_0',
        record=True,
        record_path='logs/ex3.4_walk/video_walking.mp4',
    )
    return

if __name__ == '__main__':
    exercise_3_disable_limb_spine_coupling(timestep=5e-3)
    best_params = exercise_3a_coordination(timestep=5e-3)
    exercise_3_limb_spine_antiphase(timestep=5e-3, best_parameters=best_params)
    exercise_3b_coordination(timestep=5e-3, best_parameters=best_params)

