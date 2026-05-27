import numpy as np
import matplotlib.pyplot as plt 
import h5py


# log_path = './logs/ex2a_walk/sim_0/simulation.hdf5'
# log_path = './logs/ex2b/swim/sim_0/simulation.hdf5'
log_path = './logs/ex2b/ramp/sim_water/simulation.hdf5'

with h5py.File(log_path, 'r') as f:
    links_array = f['FARMSLISTanimats/0/sensors/links/array'][:]
    joints_array = f['FARMSLISTanimats/0/sensors/joints/array'][:]
    times = f['times'][:]

# Figure out which columns are position
# Print first timestep of link 0 to see what the 20 values represent
print("Link 0 at t=0:", links_array[0, 0, :])
print("Link 0 at t=end:", links_array[-1, 0, :])
# Position is typically columns 0:3 = [x, y, z]
# link_body_04 is index 4, link_body_05 is index 17 (reordered in HDF5)
# Use index 4 as body center

positions = links_array[:, 4, :3]   # shape [3000, 3]
forward = positions[:, 1]            # Y = forward (spawn at pi/2)
lateral = positions[:, 0]            # X = lateral

# Joint positions — column 0 is the angle
# Joint mapping from names:
# 0-7: spine joints (body_00 to body_08, skipping passive)
# 8:  joint_leg_0_L_0  = FL girdle
# 9:  joint_leg_0_L_1  = FL knee
# 10: joint_leg_0_R_0  = FR girdle
# 11: joint_leg_0_R_1  = FR knee
# 12: joint_leg_1_L_0  = HL girdle
# 13: joint_leg_1_L_1  = HL knee
# 14: joint_leg_1_R_0  = HR girdle
# 15: joint_leg_1_R_1  = HR knee
joint_angles = joints_array[:, :, 0]   # shape [3000, 18]

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

fwd = forward[-1] - forward[0]
lat = lateral[-1] - lateral[0]
dur = times[-1] - times[0]
print(f"Forward displacement: {fwd:.4f} m")
print(f"Lateral drift:        {lat:.4f} m")
print(f"Average forward speed:{fwd/dur:.4f} m/s")

plt.tight_layout()
plt.show()

# --- Joint angles plot ---
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

ax = axes[0]
for j in range(8):
    ax.plot(times, joint_angles[:, j], label=f'Spine {j}', alpha=0.85)
ax.set_ylabel('Angle [rad]')
ax.set_title('Spine Joints')
ax.legend(loc='upper right', fontsize=7, ncol=4)
ax.grid(True, alpha=0.3)

ax = axes[1]
for idx, lbl in zip([8,9,12,13], ['FL girdle','FL knee','HL girdle','HL knee']):
    ax.plot(times, joint_angles[:, idx], label=lbl, alpha=0.85)
ax.set_ylabel('Angle [rad]')
ax.set_title('Left Limb Joints')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2]
for idx, lbl in zip([10,11,14,15], ['FR girdle','FR knee','HR girdle','HR knee']):
    ax.plot(times, joint_angles[:, idx], label=lbl, alpha=0.85)
ax.set_ylabel('Angle [rad]')
ax.set_xlabel('Time [s]')
ax.set_title('Right Limb Joints')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()