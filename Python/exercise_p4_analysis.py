import matplotlib.pyplot as plt
from plot_traj import load_simulation, plot_joints
from plot_results import plot_positions
from utils import get_com
import numpy as np
import os


def swimming_to_walking_transition_analysis(log_path, title_suffix, times=None):
    links_array, joints_array, _, data = load_simulation(log_path)
    if times is None:
        times = _
    else:
        start = times[0]
        end = times[-1]
        timestep = _[1] - _[0]
        times = np.arange(start, end + timestep, timestep)
    com = get_com(data, times)
    joint_angles = joints_array[:, 0]

    plot_positions(times, com, title_suffix=title_suffix)
    plot_joints(joints_array, _, title_suffix=title_suffix)
    plt.xlim(times[0], times[-1])
    
    # contact of limbs in environment
    n_iterations = _.shape[0]

    contacts_all = np.zeros((n_iterations, 19))  # Adjust the shape as needed
    for index in np.arange(n_iterations):
        contacts_all[index,:] = np.linalg.norm(
            np.array(
                data.sensors.contacts.totals()[index]
            ),
            axis=1
            )
    contacts_body = contacts_all[:, :9]
    contacts_upper_limbs = contacts_all[:, 9:17:2]
    contacts_feet = contacts_all[:, 10:18:2]
    
    fig = plt.figure(figsize=(12, 4))
    plt.plot(_, np.sum(contacts_body, axis=1)/9, label='Body Contacts', color='royalblue')
    plt.plot(_, np.sum(contacts_upper_limbs, axis=1)/4, label='Upper Limb Contacts', color='purple')
    plt.plot(_, np.sum(contacts_feet[:, 0:2], axis=1)/2, label='Fore Feet Contacts', color='seagreen')
    plt.plot(_, np.sum(contacts_feet[:, 2:4], axis=1)/2, label='Hind Feet Contacts', color='tomato')
    plt.xlabel('Time [s]')
    plt.ylabel('Contact Force Magnitude [N]')
    plt.title(f'Contact Forces Over Time {title_suffix}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.xlim(times[0], times[-1])


if __name__ == '__main__':
    swimming_to_walking_transition_analysis(
        log_path='./logs/ex4_swim2walk/sim_0/simulation.hdf5',
        title_suffix='Swim to Walk Transition',
        times = [20, 40]  # Focus on the transition period
    )
    plt.show()
    
    swimming_to_walking_transition_analysis(
        log_path='./logs/ex4_walk2swim/sim_0/simulation.hdf5',
        title_suffix='Walk to Swim Transition',
        times = [15, 35]  # Focus on the transition period
    )
    plt.show()