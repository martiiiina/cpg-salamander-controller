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
        drive=3,          # fixed drive in walking regime
        phase_lag_body=None # your walking phase lag from ex1
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
    # --- LOG ALL JOINT POSITIONS POINT-BY-POINT ---
    import h5py

    def load_from_hdf5(log_path='./logs/ex2a_walk/sim_0/simulation.hdf5'):
        with h5py.File(log_path, 'r') as f:
            # Explore the structure first
            def print_structure(name, obj):
                print(name)
            f.visititems(print_structure)
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
        phase_lag_body=None
    )
    sim_parameters.drive_ramp = np.linspace(0, 6, n_steps)  # ← ramp stored here    
    os.makedirs('./logs/ex2b_swim/', exist_ok=True)
    sim, data = simulation(
        sim_parameters=sim_parameters,
        arena='water',
        fast=True,
        output='logs/ex2b_swim/sim_0',
        #record=True,
        #record_path='logs/ex2b_swim/video_swimming.mp4',
        #record_fps=15,              # halves memory
    )

    def load_from_hdf5(log_path='./logs/ex2b_swim/sim_0/simulation.hdf5'):
        with h5py.File(log_path, 'r') as f:
            # Explore the structure first
            def print_structure(name, obj):
                print(name)
            f.visititems(print_structure)
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
        drive=0,          # fixed drive in walking regime
        phase_lag_body=None, # your walking phase lag from ex1
    )
    sim_parameters.drive_ramp = np.linspace(0, 6, n_steps)  # ← ramp stored here    
    os.makedirs('./logs/ex2b_walk/', exist_ok=True)
    sim, data = simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=True,
        output='logs/ex2b_walk/sim_0',
        #record=True,
        #record_path='logs/ex2b_walk/video_walk.mp4',
        #record_fps=15,              # halves memory
    )
    def load_from_hdf5(log_path='./logs/ex2b_swim/sim_0/simulation.hdf5'):
        with h5py.File(log_path, 'r') as f:
            # Explore the structure first
            def print_structure(name, obj):
                print(name)
            f.visititems(print_structure)
    return


if __name__ == '__main__':
    exercise_walk(timestep=5e-3)
    exercise_ramp_swim(timestep=5e-3)
    exercise_ramp_walk(timestep=5e-3)

