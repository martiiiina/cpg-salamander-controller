"""[Project1] Exercise 2: Swimming & Walking with Salamander Robot"""

import os
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
        drive=2.0,          # fixed drive in walking regime
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
    return


def exercise_ramp_swim(timestep):
    "[Project 1] Q2 Swimming with an increasing (ramp) drive"
    # Use exercise_example.py for reference
    pass
    return


def exercise_ramp_walk(timestep):
    "[Project 1] Q2 Walking with an increasing (ramp) drive"
    # Use exercise_example.py for reference
    pass
    return


if __name__ == '__main__':
    exercise_walk(timestep=5e-3)
    exercise_ramp_swim(timestep=5e-3)
    exercise_ramp_walk(timestep=5e-3)

