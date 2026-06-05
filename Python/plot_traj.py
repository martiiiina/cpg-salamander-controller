import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils import compute_fws, LINKS_MASSES, TOTAL_MASS, N_LINKS
from farms_amphibious.data.data import AmphibiousExperimentData
from utils import get_com

def load_simulation(log_path):
    with h5py.File(log_path, 'r') as f:
        links_array = f['FARMSLISTanimats/0/sensors/links/array'][:]
        joints_array = f['FARMSLISTanimats/0/sensors/joints/array'][:]
        times = f['times'][:]
    exp_data = AmphibiousExperimentData.from_file(log_path)
    data = exp_data.animats[0]
    return links_array, joints_array, times, data

def _com_trajectory(links_array):
    """Compute CoM trajectory using link masses."""
    n_links = min(links_array.shape[1], N_LINKS)
    com = np.sum(
        links_array[:, :n_links, :3] * LINKS_MASSES[:n_links][np.newaxis, :, np.newaxis],
        axis=1
    ) / TOTAL_MASS  # (n_steps, 3)
    return com
    
"""
Trajectory directly using com positions with get_com function, which should be more accurate and better plot
"""
def plot_trajectory(com, times, title_suffix):
    positions = com
    forward = positions[:, 1]            # Y = forward (spawn at pi/2)
    lateral = positions[:, 0]            # X = lateral

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(lateral, forward, color='royalblue', linewidth=1.5)
    ax.plot(lateral[0], forward[0], 'go', markersize=8, label='Start')
    ax.plot(lateral[-1], forward[-1], 'rs', markersize=8, label='End')
    ax.set_xlabel('Lateral X [m]')
    ax.set_ylabel('Forward Y [m]')
    ax.set_title('Top-Down COM Trajectory')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times, forward, label='Forward Y', color='royalblue')
    ax.plot(times, lateral, label='Lateral X', color='tomato')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('COM Position vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def compute_metrics(com, times):
    # com = _com_trajectory(links_array)
    forward = com[:, 1]
    lateral = com[:, 0]
    dur = times[-1] - times[0]

    fwd_speed = compute_fws(com, times)

    metrics = {
        'forward_displacement': forward[-1] - forward[0],
        'lateral_drift':        lateral[-1] - lateral[0],
        'forward_speed':        fwd_speed,
        'lateral_speed':        np.abs((lateral[-1] - lateral[0]) / dur),
    }

    print(f"Forward displacement:   {metrics['forward_displacement']:.4f} m")
    print(f"Lateral drift:          {metrics['lateral_drift']:.4f} m")
    print(f"Average forward speed:  {metrics['forward_speed']:.4f} m/s")
    print(f"Average lateral speed:  {metrics['lateral_speed']:.4f} m/s")

    return metrics


def plot_joints(joints_array, times, title_suffix=''):
    joint_angles = joints_array[:, :, 0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for j in range(8):
        axes[0].plot(times, joint_angles[:, j], label=f'Spine {j}', alpha=0.85)
    axes[0].set_ylabel('Angle [rad]')
    axes[0].set_title(f'Spine Joints {title_suffix}')
    axes[0].legend(loc='upper right', fontsize=7, ncol=4)
    axes[0].grid(True, alpha=0.3)

    for idx, lbl in zip([8,9,12,13], ['FL girdle','FL knee','HL girdle','HL knee']):
        axes[1].plot(times, joint_angles[:, idx], label=lbl, alpha=0.85)
    axes[1].set_ylabel('Angle [rad]')
    axes[1].set_title(f'Left Limb Joints {title_suffix}')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    for idx, lbl in zip([10,11,14,15], ['FR girdle','FR knee','HR girdle','HR knee']):
        axes[2].plot(times, joint_angles[:, idx], label=lbl, alpha=0.85)
    axes[2].set_ylabel('Angle [rad]')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_title(f'Right Limb Joints {title_suffix}')
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_simulation(log_path, title_suffix='', show=True):
    # """Load, plot and print metrics for a single simulation."""
    links_array, joints_array, times, data = load_simulation(log_path)
    com = get_com(data, times)
    metrics = compute_metrics(com, times)
    plot_trajectory(com, times, title_suffix=title_suffix)
    plot_joints(joints_array, times, title_suffix=title_suffix)
    if show:
        plt.show()
    return metrics


# Keep standalone execution working
if __name__ == '__main__':
    analyze_simulation('./logs/ex2a_walk/sim_0/simulation.hdf5')
    # analyze_simulation('./logs/ex2b_swim/sim_0/simulation.hdf5')
    # analyze_simulation('./logs/ex2b_walk/sim_0/simulation.hdf5')
    # analyze_simulation('./logs/ex3.2_walk/sim_0/simulation.hdf5')
    
    # analyze_simulation('./logs/gradient/sim_noampgrad_5e-03/simulation.hdf5', title_suffix='(No Amplitude Gradient)')
    # analyze_simulation('./logs/gradient/sim_ampgrad_5e-03/simulation.hdf5', title_suffix='(With Amplitude Gradient)')