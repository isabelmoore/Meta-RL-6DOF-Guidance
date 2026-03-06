# Copyright (c) 2026 Isabel Moore. All rights reserved.
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Load aircraft data
blue = pd.read_csv("data_output/tacview/F-16 (RL) [Blue].csv")
red = pd.read_csv("data_output/tacview/F-16 (BT) [Red].csv")

# Load UAV data (only Red AIM0 has data)
try:
    UAV_red = pd.read_csv("data_output/tacview/AIM-120 AMRAAM (AIM0) [Red].csv")
    has_UAV = len(UAV_red) > 1
except:
    has_UAV = False

# Convert lat/lon to km relative to midpoint
mid_lat = (blue["Latitude"].iloc[0] + red["Latitude"].iloc[0]) / 2
mid_lon = (blue["Longitude"].iloc[0] + red["Longitude"].iloc[0]) / 2

def to_km(df):
    x = (df["Longitude"] - mid_lon) * 111.32 * np.cos(np.radians(mid_lat))
    y = (df["Latitude"] - mid_lat) * 111.32
    alt = df["Altitude"] / 1000  # to km
    return x.values, y.values, alt.values

blue_x, blue_y, blue_alt = to_km(blue)
red_x, red_y, red_alt = to_km(red)

if has_UAV:
    mis_x, mis_y, mis_alt = to_km(UAV_red)
    mis_times = UAV_red["Time"].values

times = blue["Time"].values

# Subsample for gif speed (every 10th frame)
step = 10
frames = range(0, len(times), step)

fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#1a1a2e')

def animate(i):
    idx = i * step
    if idx >= len(times):
        idx = len(times) - 1

    trail = max(0, idx - 200)

    # Top-down view
    ax_top.clear()
    ax_top.set_facecolor('#0f0f23')
    ax_top.plot(blue_x[trail:idx], blue_y[trail:idx], color='#4fc3f7', alpha=0.4, linewidth=1)
    ax_top.plot(red_x[trail:idx], red_y[trail:idx], color='#ef5350', alpha=0.4, linewidth=1)
    ax_top.scatter(blue_x[idx], blue_y[idx], color='#4fc3f7', s=80, zorder=5, label='Blue (RL)')
    ax_top.scatter(red_x[idx], red_y[idx], color='#ef5350', s=80, zorder=5, label='Red (BT)')

    if has_UAV:
        mask = UAV_red["Time"].values <= times[idx]
        if mask.any():
            mis_trail = max(0, mask.sum() - 50)
            ax_top.plot(mis_x[mis_trail:mask.sum()], mis_y[mis_trail:mask.sum()],
                       color='#ffab40', alpha=0.6, linewidth=1, linestyle='--')
            ax_top.scatter(mis_x[mask.sum()-1], mis_y[mask.sum()-1],
                          color='#ffab40', s=40, marker='d', zorder=5, label='UAV (Red)')

    ax_top.set_xlabel('East-West (km)', color='white')
    ax_top.set_ylabel('North-South (km)', color='white')
    ax_top.set_title(f'Top-Down View  |  t={times[idx]:.0f}s', color='white', fontsize=12)
    ax_top.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
    ax_top.set_xlim(blue_x.min() - 2, max(blue_x.max(), red_x.max()) + 2)
    ax_top.set_ylim(min(blue_y.min(), red_y.min()) - 2, max(blue_y.max(), red_y.max()) + 2)
    ax_top.tick_params(colors='gray')
    ax_top.grid(True, alpha=0.15)
    ax_top.set_aspect('equal')

    # Side view (altitude profile)
    ax_side.clear()
    ax_side.set_facecolor('#0f0f23')
    ax_side.plot(blue_y[trail:idx], blue_alt[trail:idx], color='#4fc3f7', alpha=0.4, linewidth=1)
    ax_side.plot(red_y[trail:idx], red_alt[trail:idx], color='#ef5350', alpha=0.4, linewidth=1)
    ax_side.scatter(blue_y[idx], blue_alt[idx], color='#4fc3f7', s=80, zorder=5)
    ax_side.scatter(red_y[idx], red_alt[idx], color='#ef5350', s=80, zorder=5)

    if has_UAV:
        mask = UAV_red["Time"].values <= times[idx]
        if mask.any():
            mis_trail = max(0, mask.sum() - 50)
            ax_side.plot(mis_y[mis_trail:mask.sum()], mis_alt[mis_trail:mask.sum()],
                        color='#ffab40', alpha=0.6, linewidth=1, linestyle='--')
            ax_side.scatter(mis_y[mask.sum()-1], mis_alt[mask.sum()-1],
                           color='#ffab40', s=40, marker='d', zorder=5)

    ax_side.set_xlabel('North-South (km)', color='white')
    ax_side.set_ylabel('Altitude (km)', color='white')
    ax_side.set_title('Side View (Altitude)', color='white', fontsize=12)
    ax_side.tick_params(colors='gray')
    ax_side.grid(True, alpha=0.15)

    # Distance between aircraft
    dist = np.sqrt((blue_x[idx]-red_x[idx])**2 + (blue_y[idx]-red_y[idx])**2 + (blue_alt[idx]-red_alt[idx])**2)
    fig.suptitle(f'BVR Combat  |  Distance: {dist:.1f} km', color='white', fontsize=14, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])

print("Generating GIF... (this takes a minute)")
anim = animation.FuncAnimation(fig, animate, frames=len(list(frames)), interval=50)
anim.save("data_output/fight_replay.gif", writer='pillow', fps=20)
print("Saved to data_output/fight_replay.gif")
plt.close()
