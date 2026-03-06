# Copyright (c) 2026 Isabel Moore. All rights reserved.
"""
3D animated demonstration of UAV intercepting a maneuvering target.
Shows what the RL agent should learn to do.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Simulation Parameters ---
dt = 0.05
t_max = 60.0
N_nav = 4.0  # Proportional navigation constant

# --- Initial Conditions ---
# UAV: launched in tail-chase — has to run down a fleeing F-16
m_pos = np.array([0.0, 0.0, 7500.0])        # x, y, alt (m)
m_vel = np.array([270.0, 5.0, 15.0])        # m/s (~M0.8, off the rail)
m_speed = np.linalg.norm(m_vel)
m_speed_max = 550.0                           # max speed (~Mach 1.6)

# Target: F-16 running away at M0.75 — UAV barely faster
t_pos = np.array([8000.0, 400.0, 8000.0])    # 8km ahead
t_vel = np.array([250.0, 8.0, 0.0])          # m/s (~M0.75)
t_speed = np.linalg.norm(t_vel)

# --- Storage ---
m_traj = [m_pos.copy()]
t_traj = [t_pos.copy()]
ranges = []
times = []

# --- Simulate Pro-Nav Intercept ---
t = 0.0
prev_los = (t_pos - m_pos) / np.linalg.norm(t_pos - m_pos)
hit = False

while t < t_max:
    # Range and LOS
    r_vec = t_pos - m_pos
    rng = np.linalg.norm(r_vec)
    los = r_vec / rng

    ranges.append(rng)
    times.append(t)

    if rng < 15.0:  # Hit!
        hit = True
        break

    # LOS rate
    los_rate = (los - prev_los) / dt
    prev_los = los.copy()

    # Closing velocity
    v_rel = m_vel - t_vel
    v_closing = np.dot(v_rel, los)

    # Pro-Nav acceleration command: a = N * Vc * LOS_rate
    a_cmd = N_nav * v_closing * los_rate

    # Limit acceleration (45g)
    a_mag = np.linalg.norm(a_cmd)
    if a_mag > 450.0:
        a_cmd = a_cmd / a_mag * 450.0

    # Update UAV — rocket motor burns for ~8s then slow coast
    if t < 8.0:
        m_speed = min(m_speed + 40.0 * dt, m_speed_max)  # boost phase
    else:
        m_speed = max(m_speed - 3.0 * dt, 350.0)  # slow drag bleed
    m_vel += a_cmd * dt
    m_vel = m_vel / np.linalg.norm(m_vel) * m_speed
    m_pos += m_vel * dt

    # F-16 evasive maneuvering — threat-reactive
    # Computes evasion relative to actual UAV position, never flies toward it
    threat_vec = m_pos - t_pos  # vector FROM target TO UAV
    threat_dir = threat_vec / max(np.linalg.norm(threat_vec), 1.0)

    fwd = t_vel / np.linalg.norm(t_vel)

    # "Away" = perpendicular to both threat and gravity, always moving away
    beam = np.cross(threat_dir, np.array([0, 0, 1]))  # perpendicular to threat in horizontal
    if np.linalg.norm(beam) > 0.01:
        beam = beam / np.linalg.norm(beam)
    # Pick the beam direction that moves away from current heading toward UAV
    if np.dot(beam, fwd) < 0:
        beam = -beam  # choose the beam side we're already turning toward

    away = -threat_dir  # directly away from UAV
    down = np.array([0, 0, -1])

    g_load = 0.0
    pull_dir = beam

    if t < 5.0:
        # Running straight — doesn't know UAV is coming yet
        pass
    elif 5.0 < t < 10.0:
        # RWR detects launch — hard beam turn right + climb to break lock
        g_load = 6.0
        pull_dir = beam + 0.3 * np.array([0, 0, 1])
        pull_dir = pull_dir / np.linalg.norm(pull_dir)
    elif 10.0 < t < 15.0:
        # Extend away — dive for speed, move away from UAV
        g_load = 5.0
        pull_dir = away + 0.6 * down + 0.2 * beam
        pull_dir = pull_dir / np.linalg.norm(pull_dir)
    elif 15.0 < t < 20.0:
        # Reverse beam — snap to opposite side to force overshoot
        g_load = 7.0
        pull_dir = -beam + 0.2 * down
        pull_dir = pull_dir / np.linalg.norm(pull_dir)
    elif 20.0 < t < 25.0:
        # Climb + turn — trade speed for altitude, change plane of maneuver
        g_load = 5.0
        pull_dir = beam + 0.8 * np.array([0, 0, 1])
        pull_dir = pull_dir / np.linalg.norm(pull_dir)
    elif 25.0 < t < 30.0:
        # Nose low slice away — dive hard, extend
        g_load = 6.0
        pull_dir = away + down
        pull_dir = pull_dir / np.linalg.norm(pull_dir)
    elif 30.0 < t:
        # Last ditch — max-g perpendicular to UAV's velocity
        m_vel_dir = m_vel / max(np.linalg.norm(m_vel), 1.0)
        perp = np.cross(m_vel_dir, np.array([0, 0, 1]))
        if np.linalg.norm(perp) > 0.01:
            perp = perp / np.linalg.norm(perp)
        if np.dot(perp, away) < 0:
            perp = -perp
        g_load = 9.0
        pull_dir = perp + 0.3 * down
        pull_dir = pull_dir / np.linalg.norm(pull_dir)

    if g_load > 0:
        t_accel = pull_dir * g_load * 9.81
        t_vel += t_accel * dt
        t_vel = t_vel / np.linalg.norm(t_vel) * t_speed

    t_pos += t_vel * dt

    m_traj.append(m_pos.copy())
    t_traj.append(t_pos.copy())
    t += dt

m_traj = np.array(m_traj)
t_traj = np.array(t_traj)
n_frames = len(m_traj)

print(f"Simulation: {t:.1f}s, {n_frames} frames, final range: {ranges[-1]:.1f}m, hit: {hit}")

# --- Colors for white background ---
COL_UAV = '#cc0000'
COL_TARGET = '#0055cc'
COL_UAV_DOT = '#ff2222'
COL_TARGET_DOT = '#2277ff'
COL_LOS = '#999999'
COL_TEXT = '#222222'
COL_GRID = '#cccccc'
COL_HIT = '#00aa00'

# --- Create Animation ---
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_position([0.0, 0.0, 1.0, 1.0])  # fill entire figure
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Style
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor('#aaaaaa')
    axis.label.set_color(COL_TEXT)
    axis.set_tick_params(labelcolor='#444444', labelsize=12, width=1.5)
    for line in axis.get_gridlines():
        line.set_linewidth(1.5)
        line.set_color('#aaaaaa')

ax.set_xlabel('Downrange (m)', fontsize=16, fontweight='bold', labelpad=14)
ax.set_ylabel('Crossrange (m)', fontsize=16, fontweight='bold', labelpad=14)
ax.set_zlabel('Altitude (m)', fontsize=16, fontweight='bold', labelpad=14)

# Compute bounds
all_x = np.concatenate([m_traj[:, 0], t_traj[:, 0]])
all_y = np.concatenate([m_traj[:, 1], t_traj[:, 1]])
all_z = np.concatenate([m_traj[:, 2], t_traj[:, 2]])
pad = 500
ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
ax.set_zlim(all_z.min() - pad, all_z.max() + pad)

# Plot elements
UAV_trail, = ax.plot([], [], [], color=COL_UAV, linewidth=3.5, alpha=0.9, label='Interceptor')
target_trail, = ax.plot([], [], [], color=COL_TARGET, linewidth=3.5, alpha=0.9, label='Target (F-16)')
UAV_dot, = ax.plot([], [], [], 'o', color=COL_UAV_DOT, markersize=14, zorder=10)
target_dot, = ax.plot([], [], [], 's', color=COL_TARGET_DOT, markersize=14, zorder=10)
los_line, = ax.plot([], [], [], '--', color=COL_LOS, linewidth=1.0, alpha=0.4)

# Text overlays
fig.text(0.5, 0.95, '6-DOF Deep RL UAV Guidance', color=COL_TEXT, fontsize=30, fontweight='bold',
         ha='center', va='top')
fig.text(0.5, 0.91, 'Proportional Navigation Intercept', color='#666666', fontsize=24,
         ha='center', va='top')

# Left panel: legend + info box, same style
# Legend as fig text, outside the 3D axes entirely
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=COL_UAV, linewidth=4, label='Interceptor'),
                   Line2D([0], [0], color=COL_TARGET, linewidth=4, label='Target (F-16)')]
fig_legend = fig.legend(handles=legend_elements, loc='lower left', fontsize=22,
                        facecolor='white', edgecolor='#bbbbbb', labelcolor=COL_TEXT,
                        borderpad=1.2, framealpha=0.9, bbox_to_anchor=(0.02, 0.02))
fig_legend.get_frame().set_linewidth(1.5)

info_text = fig.text(0.02, 0.88, '', color=COL_TEXT, fontsize=24, fontfamily='monospace',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='#bbbbbb',
                               alpha=0.9, linewidth=1.5))

# Initial view angle
ax.view_init(elev=25, azim=-60)

def animate(frame):
    # Show every 3rd sim step for faster pace
    i = min(frame * 3, n_frames - 1)

    # Full trails
    UAV_trail.set_data(m_traj[:i+1, 0], m_traj[:i+1, 1])
    UAV_trail.set_3d_properties(m_traj[:i+1, 2])

    target_trail.set_data(t_traj[:i+1, 0], t_traj[:i+1, 1])
    target_trail.set_3d_properties(t_traj[:i+1, 2])

    # Current positions
    UAV_dot.set_data([m_traj[i, 0]], [m_traj[i, 1]])
    UAV_dot.set_3d_properties([m_traj[i, 2]])

    target_dot.set_data([t_traj[i, 0]], [t_traj[i, 1]])
    target_dot.set_3d_properties([t_traj[i, 2]])

    # LOS line
    los_line.set_data([m_traj[i, 0], t_traj[i, 0]], [m_traj[i, 1], t_traj[i, 1]])
    los_line.set_3d_properties([m_traj[i, 2], t_traj[i, 2]])

    # Range
    rng = np.linalg.norm(t_traj[i] - m_traj[i])
    time = i * dt

    # Hit flash
    if i >= n_frames - 1 and hit:
        UAV_dot.set_color(COL_HIT)
        UAV_dot.set_markersize(22)

    # UAV speed at this frame
    if i < n_frames - 1:
        mspd = np.linalg.norm(m_traj[min(i+1, n_frames-1)] - m_traj[i]) / dt
    else:
        mspd = np.linalg.norm(m_traj[i] - m_traj[i-1]) / dt

    # Info — right-align numbers using fixed-width fields
    phase = "terminal" if rng < 2000 else "midcourse" if rng < 5000 else "boost" if time < 6 else "pursuit"
    info_text.set_text(
        f'time:   {time:>6.1f}s\n'
        f'range:  {rng:>6.0f}m\n'
        f'speed:  {mspd:>5.0f} m/s (M{mspd/343:.1f})\n'
        f'phase:  {phase}'
    )

    # Slowly rotate camera
    ax.view_init(elev=25 + 10 * np.sin(frame * 0.03), azim=-60 + frame * 0.4)

    return UAV_trail, target_trail, UAV_dot, target_dot, los_line

total_anim_frames = n_frames // 3 + 8  # +8 frames to hold on intercept
anim = FuncAnimation(fig, animate, frames=total_anim_frames, interval=40, blit=False)

# Save
print("Saving GIF...")
writer = PillowWriter(fps=25)
anim.save('intercept_demo.gif', writer=writer, dpi=100)
print(f"Saved intercept_demo.gif ({total_anim_frames} frames)")
plt.close()

# Also save a static plot
fig2 = plt.figure(figsize=(18, 12))
ax2 = fig2.add_subplot(111, projection='3d')
fig2.patch.set_facecolor('white')
ax2.set_facecolor('white')

for axis in [ax2.xaxis, ax2.yaxis, ax2.zaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor(COL_GRID)
    axis.label.set_color(COL_TEXT)
    axis.set_tick_params(labelcolor='#444444', labelsize=11)

ax2.plot(m_traj[:, 0], m_traj[:, 1], m_traj[:, 2], color=COL_UAV, linewidth=3.5, label='Interceptor')
ax2.plot(t_traj[:, 0], t_traj[:, 1], t_traj[:, 2], color=COL_TARGET, linewidth=3.5, label='Target (F-16)')
ax2.plot(*m_traj[0], 'o', color=COL_UAV_DOT, markersize=16, label='UAV Launch')
ax2.plot(*t_traj[0], 's', color=COL_TARGET_DOT, markersize=16, label='Target Start')
ax2.plot(*m_traj[-1], '*', color=COL_HIT, markersize=26, label='Intercept')

# Draw a few LOS lines
for j in range(0, n_frames, max(1, n_frames // 8)):
    ax2.plot([m_traj[j, 0], t_traj[j, 0]],
             [m_traj[j, 1], t_traj[j, 1]],
             [m_traj[j, 2], t_traj[j, 2]], '--', color=COL_LOS, alpha=0.3, linewidth=0.8)

ax2.set_xlabel('Downrange (m)', fontsize=14, labelpad=12)
ax2.set_ylabel('Crossrange (m)', fontsize=14, labelpad=12)
ax2.set_zlabel('Altitude (m)', fontsize=14, labelpad=12)
ax2.set_title('6-DOF Meta-RL UAV Guidance — Complete Intercept Trajectory', color=COL_TEXT, fontsize=20, fontweight='bold', pad=20)
ax2.legend(loc='upper right', fontsize=14, facecolor='white', edgecolor='#cccccc', labelcolor=COL_TEXT)
ax2.view_init(elev=20, azim=-45)

fig2.savefig('intercept_static.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved intercept_static.png")
plt.close()
