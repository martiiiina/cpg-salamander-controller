"""[Project1] Exercise 2: Swimming & Walking with Salamander Robot"""

import os
import h5py
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_walk(timestep):
    "[Project 1] Q2 Walking with an increasing (ramp) drive"
    # Use exercise_example.py for reference
    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=2.5,          # fixed drive in walking regime
        phase_lag_body=None, # your walking phase lag from ex1
        w_limb_body=150,
        limb_body_phase_offset=np.pi/4,
        body_gain=1,
        limb_gain=1
    )
    os.makedirs('./logs/ex2a_walk/', exist_ok=True)
    sim, data = simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=True,
        output='logs/ex2a_walk/sim_0',
        record=True,
        record_path='logs/ex2a_walk/video_walking.mp4',
    )
    return

def exercise_ramp_swim(timestep):
    "[Project 1] Q2 Swimming with an increasing (ramp) drive"
    # Use exercise_example.py for reference
    # exercise_p2.py
    n_steps = int(40 / timestep)
    sim_parameters = SimulationParameters(
        duration=40,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=0.0,      # Inital value                                    
        phase_lag_body=None,
        w_limb_body=150,
        limb_body_phase_offset=0,
        body_gain=1,
        limb_gain=1
    )
    sim_parameters.drive_ramp = np.linspace(0, 6, n_steps)  # Ramp as a new parameter   
    os.makedirs('./logs/ex2b_swim/', exist_ok=True)
    sim, data = simulation(
        sim_parameters=sim_parameters,
        arena='water',
        fast=True,
        output='logs/ex2b_swim/sim_0',
        record=True,
        record_path='logs/ex2b_swim/video_swimming.mp4',
        record_fps=15,              # halves memory
    )
    return


def exercise_ramp_walk(timestep):
    "[Project 1] Q2 Walking with an increasing (ramp) drive"
    # Use exercise_example.py for reference
    n_steps = int(40 / timestep)
    sim_parameters = SimulationParameters(
        duration=40,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=0,          
        phase_lag_body=None,
        w_limb_body=150,
        limb_body_phase_offset=0,
        body_gain=1,
        limb_gain=1
    )
    sim_parameters.drive_ramp = np.linspace(0, 6, n_steps)  
    os.makedirs('./logs/ex2b_walk/', exist_ok=True)
    sim, data = simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=True,
        output='logs/ex2b_walk/sim_0',
        record=True,
        record_path='logs/ex2b_walk/video_walk.mp4',
        record_fps=15,              # halves memory
    )
    return

def exercise_swim(timestep, amplitude_gradient=None):
    "[Project 1 EXTRA] Q2 Swimming with fixed drive, analyze impact of gradient"
    # Use exercise_example.py for reference
    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=4,          # fixed drive in walking regime
        phase_lag_body=None, # your walking phase lag from ex1
        w_limb_body=150,
        limb_body_phase_offset=np.pi/4,
        body_gain=1,
        limb_gain=1,
        amplitude_gradient=amplitude_gradient
    )
    os.makedirs('./logs/gradient/', exist_ok=True)
    if amplitude_gradient is not None:
        PATH = f'./logs/gradient/sim_ampgrad_{timestep:.0e}'
        VIDEO_PATH = f'./logs/gradient/video_swimming_ampgrad.mp4'
    else:
        PATH = f'./logs/gradient/sim_noampgrad_{timestep:.0e}'
        VIDEO_PATH = f'./logs/gradient/video_swimming_noampgrad.mp4'
    sim, data = simulation(
        sim_parameters=sim_parameters,
        arena='water',
        fast=True,
        output=PATH,
        record=True,
        record_path=VIDEO_PATH,
    )
    return

def exercise_swim(timestep):
    "Additional simulation for pure swimming behaviour"
    # Use exercise_example.py for reference
    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=4,          # fixed drive in walking regime
        phase_lag_body=None, # your walking phase lag from ex1
        w_limb_body=150,
        limb_body_phase_offset=np.pi/4,
        body_gain=1,
        limb_gain=1
    )
    os.makedirs('./logs/ex2a_swim/', exist_ok=True)
    sim, data = simulation(
        sim_parameters=sim_parameters,
        arena='water',
        fast=True,
        output='logs/ex2a_swim/sim_0',
        record=True,
        record_path='logs/ex2a_swim/video_swimming.mp4',
    )
    return
if __name__ == '__main__':
    exercise_walk(timestep=5e-3)
    exercise_ramp_swim(timestep=5e-3)
    exercise_ramp_walk(timestep=5e-3)
    # exercise_swim(timestep=5e-3)
    
    # R_head = 0.25
    # R_tail = 1
    # amplitude_gradient = R_head + (R_tail - R_head) * (np.arange(8) / 7) # Linear gradient from head to tail
    # amplitude_gradient = np.repeat(amplitude_gradient, 2)  # Same gradient for left and right body; limbs have gradient of 1
    # exercise_swim(timestep=5e-3)
    # exercise_swim(timestep=5e-3, amplitude_gradient=amplitude_gradient)    
