# Copyright (c) 2026 Isabel Moore. All rights reserved.
"""
Usage:
    python evaluate_meta.py --run training_logs/Feb22_META_A_B_C_20M
    python evaluate_meta.py --run training_logs/... --scenarios A B --gifs 5
    python evaluate_meta.py --model training_logs/.../models/final --scenarios all
"""
import argparse
import json
import os
import re
import sys
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from glob import glob

from sb3_contrib import RecurrentPPO
from simulation.environments.uav_guidance_env import UAVGuidanceEnv
from simulation.core.scenario_loader import load_scenario, find_scenario, list_scenarios

import pymap3d as pm


def find_best_model(run_dir):
    """Return path to the best (or final) model checkpoint in a training run directory.

    Args:
        run_dir: path to the training run directory containing a models/ subfolder.
    """
    models_dir = os.path.join(run_dir, "models")
    if not os.path.isdir(models_dir):
        return None
    best_files = glob(os.path.join(models_dir, "*_best.zip"))
    if not best_files:
        final = os.path.join(models_dir, "final.zip")
        return final if os.path.isfile(final) else None

    def step_num(path):
        m = re.search(r'(\d+)_best\.zip$', path)
        return int(m.group(1)) if m else 0
    return max(best_files, key=step_num)


def evaluate_scenario(model, scenario_name, n_episodes=50, holdout=False):
    """Run n_episodes in a scenario and return hit rate, miss distance, and reward stats.

    Args:
        model: loaded RecurrentPPO model.
        scenario_name: scenario ID string (e.g. 'A', 'B', 'C').
        n_episodes: number of evaluation episodes to run.
        holdout: if True, marks results as holdout (not used in training).
    """
    scenario_path = find_scenario(scenario_name)
    conf, label = load_scenario(scenario_path)

    conf.curriculum_file = None
    conf.reward_params.hit_radius_start = 0.0
    conf.reward_params.hit_radius_end = 0.0

    env = UAVGuidanceEnv(conf=conf)

    miss_distances = []
    termination_reasons = Counter()
    episode_rewards = []
    episode_times = []
    all_trajectories = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        done = False
        total_reward = 0.0

        ref_lat = env.UAV.get_lat_gc_deg()
        ref_lon = env.UAV.get_long_gc_deg()
        ref_alt = env.UAV.get_altitude()

        trajectory = {"UAV": [], "target": [], "time": [], "reason": ""}

        def record_pos():
            me, mn, mu = pm.geodetic2enu(
                env.UAV.get_lat_gc_deg(), env.UAV.get_long_gc_deg(),
                env.UAV.get_altitude(), ref_lat, ref_lon, ref_alt, deg=True)
            trajectory["UAV"].append([me / 1000, mn / 1000, mu / 1000])
            te, tn, tu = pm.geodetic2enu(
                env.target.get_lat_gc_deg(), env.target.get_long_gc_deg(),
                env.target.get_altitude(), ref_lat, ref_lon, ref_alt, deg=True)
            trajectory["target"].append([te / 1000, tn / 1000, tu / 1000])
            trajectory["time"].append(env.sim_time)

        record_pos()
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start,
                deterministic=True)
            episode_start = np.zeros((1,), dtype=bool)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            record_pos()

        reason = info.get("termination_reason", "unknown")
        final_range = info.get("range", float("inf"))
        sim_time = info.get("sim_time", 0.0)

        trajectory["reason"] = reason
        trajectory["final_range"] = final_range
        all_trajectories.append(trajectory)
        termination_reasons[reason] += 1
        miss_distances.append(final_range)
        episode_rewards.append(total_reward)
        episode_times.append(sim_time)

    miss_arr = np.array(miss_distances)
    rew_arr = np.array(episode_rewards)
    hit_count = termination_reasons.get("hit", 0)
    hit_rate = hit_count / n_episodes

    return {
        "scenario": scenario_name,
        "label": label,
        "holdout": holdout,
        "n_episodes": n_episodes,
        "hit_rate": round(hit_rate, 4),
        "hit_count": hit_count,
        "miss_distance": {
            "mean": round(float(np.mean(miss_arr)), 1),
            "median": round(float(np.median(miss_arr)), 1),
            "min": round(float(np.min(miss_arr)), 1),
            "std": round(float(np.std(miss_arr)), 1),
        },
        "reward": {
            "mean": round(float(np.mean(rew_arr)), 2),
            "std": round(float(np.std(rew_arr)), 2),
        },
        "termination_reasons": {
            r: {"count": c, "pct": round(c / n_episodes, 3)}
            for r, c in termination_reasons.most_common()
        },
        "trajectories": all_trajectories,
    }


def print_comparison_table(results):
    """Print a formatted table of evaluation metrics across scenarios.

    Args:
        results: list of result dicts from evaluate_scenario().
    """
    header = f"{'Scenario':<22s} {'Tag':>8s} {'Hit%':>8s} {'MissMean':>10s} {'MissMin':>10s} {'Reward':>10s} {'TopReason':>15s}"
    print(f"\n{'='*90}")
    print("META-RL EVALUATION RESULTS")
    print(f"{'='*90}")
    print(header)
    print("-" * 90)

    for r in results:
        tag = "[HOLDOUT]" if r["holdout"] else ""
        top_reason = ""
        if r["termination_reasons"]:
            top_reason = max(r["termination_reasons"],
                             key=lambda k: r["termination_reasons"][k]["count"])

        print(
            f"{r['label']:<22s} {tag:>8s} "
            f"{r['hit_rate']:>7.1%} "
            f"{r['miss_distance']['mean']:>9.1f}m "
            f"{r['miss_distance']['min']:>9.1f}m "
            f"{r['reward']['mean']:>9.1f} "
            f"{top_reason:>15s}"
        )

    print(f"{'='*90}")

    all_hits = sum(r["hit_count"] for r in results)
    all_eps = sum(r["n_episodes"] for r in results)
    train_results = [r for r in results if not r["holdout"]]
    holdout_results = [r for r in results if r["holdout"]]

    if train_results:
        train_hr = sum(r["hit_count"] for r in train_results) / sum(r["n_episodes"] for r in train_results)
        print(f"Train scenarios aggregate hit rate:   {train_hr:.1%}")
    if holdout_results:
        hold_hr = sum(r["hit_count"] for r in holdout_results) / sum(r["n_episodes"] for r in holdout_results)
        print(f"Holdout scenarios aggregate hit rate:  {hold_hr:.1%}")
    print(f"Overall hit rate: {all_hits}/{all_eps} = {all_hits/all_eps:.1%}")


_COL_UAV = '#cc0000'
_COL_TARGET = '#0055cc'
_COL_UAV_DOT = '#ff2222'
_COL_TARGET_DOT = '#2277ff'


def _mpl_text_overlay(width, height, text_blocks):
    """Render text overlay using matplotlib (matching trajectory GIF fonts).

    text_blocks: list of dicts with keys:
        x, y: position in pixels (y=0 is top)
        text: string
        fontsize: int
        fontfamily: 'monospace' or 'sans-serif'
        color: matplotlib color string
        fontweight: 'normal' or 'bold'
        ha: horizontal alignment ('left', 'center', 'right')
        va: vertical alignment ('top', 'bottom', 'center')
        bbox: optional dict for matplotlib bbox (like boxstyle, facecolor, etc.)
    Returns a PIL RGBA image.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image as PilImg

    dpi = 72
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_alpha(0)

    for block in text_blocks:
        fx = block['x'] / width
        fy = 1.0 - block['y'] / height
        kwargs = dict(
            fontsize=block.get('fontsize', 13),
            color=block.get('color', '#222222'),
            fontweight=block.get('fontweight', 'normal'),
            ha=block.get('ha', 'left'),
            va=block.get('va', 'top'),
        )
        ff = block.get('fontfamily', 'monospace')
        if ff:
            kwargs['fontfamily'] = ff
        bbox = block.get('bbox')
        if bbox:
            kwargs['bbox'] = bbox
        fig.text(fx, fy, block['text'], **kwargs)

    buf = BytesIO()
    fig.savefig(buf, format='png', transparent=True, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return PilImg.open(buf).convert('RGBA')
_COL_LOS = '#999999'
_COL_TEXT = '#222222'
_COL_HIT = '#00aa00'


def make_demo_gif(traj, path, ep_num):
    """Generate trajectory overview GIF with 3D matplotlib plot of UAV and target paths.

    Args:
        traj: trajectory dict with keys 'UAV', 'target', 'attitudes', 'fins',
              'flight_data', 'target_flight_data', 'time', 'reason'.
        path: output GIF file path.
        ep_num: episode number (displayed in title).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D

    m = np.array(traj["UAV"])
    t = np.array(traj["target"])
    times = np.array(traj["time"])
    flight_data = np.array(traj["flight_data"])  # [alt, airspeed, mach, p, q, r]
    tgt_flight_data = np.array(traj["target_flight_data"])  # [airspeed, mach]
    reason = traj["reason"]
    final_range = traj.get("final_range", -1)

    win = min(15, len(t) // 2 * 2 - 1)
    if win >= 3:
        pad = win // 2
        for ax_i in range(3):
            cumsum = np.cumsum(np.insert(t[:, ax_i], 0, 0))
            smoothed = (cumsum[win:] - cumsum[:-win]) / win
            t[pad:-pad, ax_i] = smoothed

    ranges_m = np.linalg.norm(m - t, axis=1)

    total = len(m)
    n_frames = 120
    step = max(1, total // n_frames)
    indices = list(range(0, total, step))
    if indices[-1] != total - 1:
        indices.append(total - 1)
    indices.extend([total - 1] * 8)

    sim_duration = times[-1] - times[0]
    gif_duration = len(indices) * 0.025  # 25ms per frame
    speedup = sim_duration / gif_duration if gif_duration > 0 else 1

    all_pts = np.concatenate([m, t], axis=0)
    pad = 500
    xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    zlim = (all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)

    fig = plt.figure(figsize=(900 / 72, 450 / 72))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('#aaaaaa')
        axis.label.set_color(_COL_TEXT)
        axis.set_tick_params(labelcolor='#444444', labelsize=7, width=1.0)
        for line in axis.get_gridlines():
            line.set_linewidth(1.5)
            line.set_color('#aaaaaa')

    ax.set_xlabel('Longitude (m)', fontsize=9, fontweight='bold', labelpad=8)
    ax.set_ylabel('Latitude (m)', fontsize=9, fontweight='bold', labelpad=8)
    ax.set_zlabel('Altitude (m)', fontsize=9, fontweight='bold', labelpad=8)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)

    uav_trail, = ax.plot([], [], [], color=_COL_UAV, linewidth=2.5, alpha=0.9)
    target_trail, = ax.plot([], [], [], color=_COL_TARGET, linewidth=2.5, alpha=0.9)
    uav_dot, = ax.plot([], [], [], 'o', color=_COL_UAV_DOT, markersize=10, zorder=10)
    target_dot, = ax.plot([], [], [], 's', color=_COL_TARGET_DOT, markersize=10, zorder=10)
    los_line, = ax.plot([], [], [], '--', color=_COL_LOS, linewidth=1.0, alpha=0.4)

    fig.text(0.5, 0.984, 'Trajectory Overview', color=_COL_TEXT, fontsize=15,
             fontweight='bold', ha='center', va='top')
    fig.text(0.5, 0.934, f'Episode {ep_num}  ({speedup:.0f}x real time)', color='#666666',
             fontsize=12, ha='center', va='top')

    legend_elements = [Line2D([0], [0], color=_COL_UAV, linewidth=4, label='UAV'),
                       Line2D([0], [0], color=_COL_TARGET, linewidth=4, label='Target')]
    fig_legend = fig.legend(handles=legend_elements, loc='lower left', fontsize=11,
                            facecolor='white', edgecolor='#bbbbbb', labelcolor=_COL_TEXT,
                            borderpad=1.2, framealpha=0.9, bbox_to_anchor=(0.02, 0.02))
    fig_legend.get_frame().set_linewidth(1.5)

    info_text = fig.text(0.02, 0.88, '', color=_COL_TEXT, fontsize=11, fontfamily='monospace',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.7', facecolor='white',
                                   edgecolor='#bbbbbb', alpha=0.9, linewidth=1.5))

    ax.view_init(elev=25, azim=-60)

    def animate(frame):
        idx = indices[frame]

        uav_trail.set_data(m[:idx+1, 0], m[:idx+1, 1])
        uav_trail.set_3d_properties(m[:idx+1, 2])
        target_trail.set_data(t[:idx+1, 0], t[:idx+1, 1])
        target_trail.set_3d_properties(t[:idx+1, 2])

        uav_dot.set_data([m[idx, 0]], [m[idx, 1]])
        uav_dot.set_3d_properties([m[idx, 2]])
        target_dot.set_data([t[idx, 0]], [t[idx, 1]])
        target_dot.set_3d_properties([t[idx, 2]])

        los_line.set_data([m[idx, 0], t[idx, 0]], [m[idx, 1], t[idx, 1]])
        los_line.set_3d_properties([m[idx, 2], t[idx, 2]])

        uav_speed = flight_data[idx, 1]      # true airspeed from JSBSim
        uav_mach = flight_data[idx, 2]       # mach from JSBSim
        tgt_speed = tgt_flight_data[idx, 0]  # true airspeed from JSBSim
        tgt_mach = tgt_flight_data[idx, 1]   # mach from JSBSim

        rng = ranges_m[idx]
        phase = "terminal" if rng < 2000 else "midcourse" if rng < 5000 else "closing"

        is_final = idx == total - 1
        if is_final and reason == 'hit':
            uav_dot.set_color(_COL_HIT)
            uav_dot.set_markersize(16)
            info_text.set_text(
                f'time:       {times[idx]:>6.1f} s\n'
                f'range:      {final_range:>6.0f} m\n'
                f'UAV speed:  {uav_speed:>5.0f} m/s (M{uav_mach:.1f})\n'
                f'tgt speed:  {tgt_speed:>5.0f} m/s (M{tgt_mach:.1f})\n'
                f'result:     HIT'
            )
        elif is_final:
            uav_dot.set_color('#ff8800')
            info_text.set_text(
                f'time:       {times[idx]:>6.1f} s\n'
                f'range:      {final_range:>6.0f} m\n'
                f'UAV speed:  {uav_speed:>5.0f} m/s (M{uav_mach:.1f})\n'
                f'tgt speed:  {tgt_speed:>5.0f} m/s (M{tgt_mach:.1f})\n'
                f'result:     {reason}'
            )
        else:
            info_text.set_text(
                f'time:       {times[idx]:>6.1f} s\n'
                f'range:      {rng:>6.0f} m\n'
                f'UAV speed:  {uav_speed:>5.0f} m/s (M{uav_mach:.1f})\n'
                f'tgt speed:  {tgt_speed:>5.0f} m/s (M{tgt_mach:.1f})\n'
                f'phase:      {phase}'
            )

        progress = frame / max(len(indices) - 1, 1)
        ax.view_init(elev=25, azim=-60 + 45 * progress)
        return uav_trail, target_trail, uav_dot, target_dot, los_line

    anim = FuncAnimation(fig, animate, frames=len(indices), interval=25, blit=False)
    writer = PillowWriter(fps=40)
    anim.save(path, writer=writer, dpi=72)
    plt.close()
    print(f"  Saved: {path}  ({reason}, {final_range:.0f}m, {times[-1]:.1f}s)")


def make_vehicle_gif(traj, path, ep_num):
    """Generate vehicle close-up GIF with 3D body mesh, ADI, gimbal, and telemetry overlay.

    Args:
        traj: trajectory dict with keys 'UAV', 'target', 'attitudes', 'fins',
              'flight_data', 'target_flight_data', 'time', 'reason'.
        path: output GIF file path.
        ep_num: episode number (displayed in title).
    """
    try:
        import pyvista as pv
        from PIL import Image, ImageDraw
    except ImportError:
        print(f"  PyVista/Pillow not available, skipping vehicle GIF: {path}")
        return

    try:
        pv.start_xvfb()
    except Exception:
        pass

    pos_arr = np.array(traj["UAV"])
    tgt_arr = np.array(traj["target"])
    att_arr = np.array(traj["attitudes"])
    fin_arr = np.array(traj["fins"])
    flight_data_arr = np.array(traj["flight_data"])
    times = np.array(traj["time"])
    reason = traj["reason"]

    vg = traj.get("vehicle_geometry", {})
    L = vg.get("length_m", 6.25)
    R_b = vg.get("radius_m", 0.155)
    CG = vg.get("cg_m", 3.13)
    nose_len = vg.get("nose_frac", 0.15) * L
    fin_span = vg.get("fin_span_m", 0.63)
    fin_start = 0.88 * L
    fin_chord = L - fin_start
    fin_root_x = CG - fin_start

    total = len(pos_arr)
    n_frames = 120
    step = max(1, total // n_frames)
    indices = list(range(0, total, step))
    if indices[-1] != total - 1:
        indices.append(total - 1)
    indices.extend([total - 1] * 8)

    sim_duration = times[-1] - times[0]
    gif_duration = len(indices) * 0.025
    speedup = sim_duration / gif_duration if gif_duration > 0 else 1

    rho_ogive = (R_b ** 2 + nose_len ** 2) / (2 * R_b)
    x_nose = np.linspace(0, nose_len, 30)
    r_nose = np.sqrt(rho_ogive ** 2 - (nose_len - x_nose) ** 2) + R_b - rho_ogive
    r_nose[0] = 0.001
    x_cyl = np.linspace(nose_len, L, 10)[1:]
    r_cyl = np.full_like(x_cyl, R_b)
    x_prof = CG - np.concatenate([x_nose, x_cyl])
    r_prof = np.concatenate([r_nose, r_cyl])

    n_theta = 33
    theta = np.linspace(0, 2 * np.pi, n_theta)
    X = np.outer(np.ones_like(theta), x_prof).T
    Y = np.outer(np.cos(theta), r_prof).T
    Z = np.outer(np.sin(theta), r_prof).T
    body_mesh_base = pv.StructuredGrid(X, Y, Z).extract_surface(algorithm=None)
    body_pts_orig = body_mesh_base.points.copy()

    t = 0.008
    sweep = 0.18
    tip_chord = fin_chord * 0.55
    tail_pts = np.array([
        [0, 0, -t], [fin_chord, 0, -t],
        [sweep + tip_chord, fin_span, -t], [sweep, fin_span, -t],
        [0, 0, t], [fin_chord, 0, t],
        [sweep + tip_chord, fin_span, t], [sweep, fin_span, t],
    ])
    fin8_faces = np.hstack([
        [4, 0, 3, 2, 1], [4, 4, 5, 6, 7],
        [4, 0, 1, 5, 4], [4, 2, 3, 7, 6],
        [4, 0, 4, 7, 3], [4, 1, 2, 6, 5],
    ])
    fin_configs = [
        (0, 'rud'), (np.pi / 2, 'elev'),
        (np.pi, 'rud'), (3 * np.pi / 2, 'elev'),
    ]

    wing_x = CG - 2.0
    wing_chord = 0.7
    wing_span = 0.40
    wing_sweep = 0.35
    wing_tip_chord = wing_chord * 0.3
    wing_pts = np.array([
        [wing_x, 0, -t], [wing_x + wing_chord, 0, -t],
        [wing_x + wing_sweep + wing_tip_chord, wing_span, -t],
        [wing_x + wing_sweep, wing_span, -t],
        [wing_x, 0, t], [wing_x + wing_chord, 0, t],
        [wing_x + wing_sweep + wing_tip_chord, wing_span, t],
        [wing_x + wing_sweep, wing_span, t],
    ])
    wing_radials = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    def rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def body_to_enu_mat(phi, th, psi):
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(th), np.sin(th)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        R_nb = np.array([
            [cth * cpsi, cth * spsi, -sth],
            [sphi * sth * cpsi - cphi * spsi, sphi * sth * spsi + cphi * cpsi, sphi * cth],
            [cphi * sth * cpsi + sphi * spsi, cphi * sth * spsi - sphi * cpsi, cphi * cth],
        ])
        T_ne = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        return T_ne @ R_nb.T

    def place_tail_fin(pts, radial, defl):
        p = pts.copy()
        p = (rx(defl) @ p.T).T
        p[:, 1] += R_b
        p = (rx(radial) @ p.T).T
        p[:, 0] += fin_root_x
        return p

    def place_wing(pts, radial):
        p = pts.copy()
        p[:, 1] += R_b
        p = (rx(radial) @ p.T).T
        return p

    smooth_w = min(21, max(1, total // 10))
    fin_smooth = np.copy(fin_arr)
    att_smooth = np.copy(att_arr)
    fd_smooth = np.copy(flight_data_arr)

    def _moving_avg(arr, w):
        pad = w // 2
        cs = np.cumsum(np.insert(arr, 0, 0))
        smoothed = (cs[w:] - cs[:-w]) / w
        out = arr.copy()
        out[pad:pad + len(smoothed)] = smoothed
        return out

    if smooth_w >= 3:
        for col in range(fin_arr.shape[1]):
            fin_smooth[:, col] = _moving_avg(fin_arr[:, col], smooth_w)
        for col in range(att_arr.shape[1]):
            unwrapped = np.unwrap(att_arr[:, col])
            att_smooth[:, col] = _moving_avg(unwrapped, smooth_w)
        for col in range(flight_data_arr.shape[1]):
            fd_smooth[:, col] = _moving_avg(flight_data_arr[:, col], smooth_w)

    body_color = '#a8c4d8'
    fin_color = '#90b0c4'

    ring_n = 80
    ring_r = 3.0
    ring_angles = np.linspace(0, 2 * np.pi, ring_n)

    horiz_pts = np.column_stack([
        ring_r * np.cos(ring_angles),
        ring_r * np.sin(ring_angles),
        np.zeros(ring_n),
    ])
    ns_pts = np.column_stack([
        np.zeros(ring_n),
        ring_r * np.cos(ring_angles),
        ring_r * np.sin(ring_angles),
    ])
    ew_pts = np.column_stack([
        ring_r * np.cos(ring_angles),
        np.zeros(ring_n),
        ring_r * np.sin(ring_angles),
    ])

    def make_ring_mesh(pts):

        lines = np.zeros((ring_n, 3), dtype=int)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(ring_n)
        lines[:, 2] = (np.arange(ring_n) + 1) % ring_n
        poly = pv.PolyData(pts, lines=lines.ravel())
        return poly.tube(radius=0.04)

    horiz_ring = make_ring_mesh(horiz_pts)
    ns_ring = make_ring_mesh(ns_pts)
    ew_ring = make_ring_mesh(ew_pts)

    def draw_adi_fast(phi_r, theta_r, psi_r, size=140):

        import math
        from PIL import Image as PilImg
        r = size // 2

        adi = PilImg.new('RGB', (size, size), (255, 255, 255))
        d = ImageDraw.Draw(adi)

        d.ellipse([4, 4, size - 5, size - 5], fill=(200, 225, 250))

        cos_r, sin_r = math.cos(phi_r), math.sin(phi_r)
        pitch_px = theta_r * r * 1.2
        far = size * 2
        gnd_pts = []
        for lx, ly in [(-far, pitch_px), (far, pitch_px), (far, far), (-far, far)]:
            gnd_pts.append((r + lx * cos_r - ly * sin_r,
                            r + lx * sin_r + ly * cos_r))
        d.polygon(gnd_pts, fill=(215, 195, 160))

        corners = PilImg.new('RGB', (size, size), (255, 255, 255))
        cd = ImageDraw.Draw(corners)
        cd.ellipse([4, 4, size - 5, size - 5], fill=(0, 0, 0))
        for y_px in range(size):
            for x_px in range(size):
                if corners.getpixel((x_px, y_px))[0] > 128:
                    adi.putpixel((x_px, y_px), (255, 255, 255))

        hx1 = r + int(-r * cos_r - pitch_px * sin_r)
        hy1 = r + int(-r * sin_r + pitch_px * cos_r)
        hx2 = r + int(r * cos_r - pitch_px * sin_r)
        hy2 = r + int(r * sin_r + pitch_px * cos_r)
        d.line([(hx1, hy1), (hx2, hy2)], fill=(80, 80, 80), width=2)

        for pitch_deg in [-20, -10, 10, 20]:
            pp = pitch_px + pitch_deg * (r * 1.2 / 57.3)
            lw = r // 3 if abs(pitch_deg) == 10 else r // 4
            lx1 = r + int(-lw * cos_r - pp * sin_r)
            ly1 = r + int(-lw * sin_r + pp * cos_r)
            lx2 = r + int(lw * cos_r - pp * sin_r)
            ly2 = r + int(lw * sin_r + pp * cos_r)
            d.line([(lx1, ly1), (lx2, ly2)], fill=(140, 140, 140), width=1)

        d.line([(r - 22, r), (r - 6, r)], fill=(200, 50, 50), width=3)
        d.line([(r + 6, r), (r + 22, r)], fill=(200, 50, 50), width=3)
        d.ellipse([r - 3, r - 3, r + 3, r + 3], outline=(200, 50, 50), width=2)

        hdg_deg = math.degrees(psi_r) % 360
        d.text((r - 14, size - 18), f"{hdg_deg:.0f}\u00b0", fill=(60, 60, 60))

        d.ellipse([2, 2, size - 3, size - 3], outline=(170, 175, 185), width=2)

        adi_rgba = adi.convert('RGBA')
        alpha = PilImg.new('L', (size, size), 0)
        ImageDraw.Draw(alpha).ellipse([2, 2, size - 3, size - 3], fill=255)
        adi_rgba.putalpha(alpha)
        return adi_rgba

    def render_attitude_sphere(body_mesh, body_pts, fins_template, fin8f, fin_cfgs,
                               wings_template, wing_rads, place_fin_fn, place_wing_fn,
                               phi_r, theta_r, psi_r, elev_r, rud_r,
                               body_col, fin_col, size=240):

        R_be = body_to_enu_mat(phi_r, theta_r, psi_r)
        R_display = R_be @ np.diag([1.0, 1.0, -1.0])

        pl2 = pv.Plotter(off_screen=True, window_size=[size, size])
        pl2.set_background('white')

        pl2.add_mesh(horiz_ring.copy(), color='#5599dd')
        pl2.add_mesh(ns_ring.copy(), color='#55bb55')
        pl2.add_mesh(ew_ring.copy(), color='#dd5555')

        bm = body_mesh.copy()
        bm.points = (R_display @ body_pts.T).T
        pl2.add_mesh(bm, color=body_col, smooth_shading=True,
                     show_edges=False, specular=0.3)

        for radial, ctrl_type in fin_cfgs:
            defl = elev_r if ctrl_type == 'elev' else rud_r
            local = place_fin_fn(fins_template, radial, defl)
            rotated = (R_display @ local.T).T
            pl2.add_mesh(pv.PolyData(rotated, fin8f.copy()),
                         color=fin_col, smooth_shading=True, show_edges=False)

        for radial in wing_rads:
            local = place_wing_fn(wings_template, radial)
            rotated = (R_display @ local.T).T
            pl2.add_mesh(pv.PolyData(rotated, fin8f.copy()),
                         color=fin_col, smooth_shading=True, show_edges=False)

        d_lbl = ring_r + 0.5
        for lbl, local_pos, col in [
            ('N', [0, d_lbl, 0], '#55bb55'),
            ('S', [0, -d_lbl, 0], '#55bb55'),
            ('E', [d_lbl, 0, 0], '#dd5555'),
            ('W', [-d_lbl, 0, 0], '#dd5555'),
        ]:
            pl2.add_point_labels(
                np.array([local_pos]), [lbl], font_size=14, text_color=col,
                point_size=0, shape=None, always_visible=True)

        pl2.camera.position = (-3, -5, 3)
        pl2.camera.focal_point = (0, 0, 0)
        pl2.camera.up = (0, 0, 1)
        pl2.camera.clipping_range = (0.1, 100)
        pl2.reset_camera()
        pl2.camera.zoom(1.0)

        img2 = pl2.screenshot(return_img=True)
        pl2.close()
        return Image.fromarray(img2)

    frames = []
    for fi, idx in enumerate(indices):
        p = pos_arr[idx]
        phi, theta_att, psi = att_smooth[idx]
        elev_rad, rud_rad = fin_smooth[idx]

        R_be = body_to_enu_mat(phi, theta_att, psi)

        world_body = (R_be @ body_pts_orig.T).T + p

        world_fins = []
        for radial, ctrl_type in fin_configs:
            defl = elev_rad if ctrl_type == 'elev' else rud_rad
            local = place_tail_fin(tail_pts, radial, defl)
            world_fins.append((R_be @ local.T).T + p)

        world_wings = []
        for radial in wing_radials:
            local = place_wing(wing_pts, radial)
            world_wings.append((R_be @ local.T).T + p)

        pl = pv.Plotter(off_screen=True, window_size=[900, 450])
        pl.set_background('white')

        bm = body_mesh_base.copy()
        bm.points = world_body
        pl.add_mesh(bm, color=body_color, smooth_shading=True,
                    show_edges=False, specular=0.3, specular_power=15)

        for wf in world_fins:
            pl.add_mesh(pv.PolyData(wf, fin8_faces.copy()),
                        color=fin_color, smooth_shading=True,
                        show_edges=False, specular=0.2)
        for ww in world_wings:
            pl.add_mesh(pv.PolyData(ww, fin8_faces.copy()),
                        color=fin_color, smooth_shading=True,
                        show_edges=False, specular=0.2)

        progress = fi / max(len(indices) - 1, 1)
        orbit = -0.3 + 0.6 * progress
        cam_body = np.array([
            8,
            3 * np.cos(orbit),
            -3.5 - 0.7 * np.sin(orbit),
        ])
        cam_pos = p + R_be @ cam_body
        focal = p + R_be @ np.array([-2, -1.2, 0.8])
        pl.camera.position = cam_pos
        pl.camera.focal_point = focal
        pl.camera.up = R_be @ np.array([0, 0, -1])
        pl.camera.clipping_range = (0.1, 10000)

        img = pl.screenshot(return_img=True)
        pl.close()

        pil_img = Image.fromarray(img)

        fd = fd_smooth[idx]
        alt, airspeed, mach = fd[0], fd[1], fd[2]
        p_dps, q_dps, r_dps = np.degrees(fd[3]), np.degrees(fd[4]), np.degrees(fd[5])
        elev_deg = np.degrees(elev_rad)
        rud_deg = np.degrees(rud_rad)
        px, py, pz = pos_arr[idx]

        rng_to_tgt = np.linalg.norm(pos_arr[idx] - tgt_arr[idx])
        if idx > 0:
            rng_prev = np.linalg.norm(pos_arr[idx - 1] - tgt_arr[idx - 1])
            dt = times[idx] - times[idx - 1]
            closing_speed = (rng_prev - rng_to_tgt) / dt if dt > 0 else 0
        else:
            closing_speed = 0

        g_load = airspeed * np.sqrt(fd[4]**2 + fd[5]**2) / 9.81 if airspeed > 1 else 0

        w, h = pil_img.size

        left_lines = [
            f"time:      {times[idx]:>8.2f} s",
            f"altitude:  {alt:>8.1f} m",
            f"speed:     {airspeed:>8.1f} m/s",
            f"mach:      {mach:>8.2f}",
            f"range:     {rng_to_tgt:>8.0f} m",
            f"closing:   {closing_speed:>+8.0f} m/s",
            f"load:      {g_load:>8.1f} g",
        ]
        if idx == total - 1:
            result_str = "HIT" if reason == "hit" else reason.upper()
            left_lines.append(f"result:    {result_str:>8s}")

        phi_d, th_d, psi_d = np.degrees(phi), np.degrees(theta_att), np.degrees(psi)
        right_lines = [
            f"position (m):    {px:>+9.1f} {py:>+9.1f} {pz:>+9.1f}",
            f"attitude (deg):  {phi_d:>+9.1f} {th_d:>+9.1f} {psi_d:>+9.1f}",
            f"rates (deg/s):   {p_dps:>+9.1f} {q_dps:>+9.1f} {r_dps:>+9.1f}",
            f"elevator (deg):  {elev_deg:>+9.2f}",
            f"rudder (deg):    {rud_deg:>+9.2f}",
        ]

        text_blocks = [
            dict(x=w // 2, y=6, text='Vehicle Close-Up',
                 fontsize=15, fontweight='bold', ha='center', fontfamily='sans-serif',
                 color=_COL_TEXT),
            dict(x=w // 2, y=28, text=f'Episode {ep_num}  ({speedup:.0f}x real time)',
                 fontsize=12, ha='center', fontfamily='sans-serif', color='#666666'),
            dict(x=12, y=60, text='\n'.join(left_lines),
                 fontsize=12, fontfamily='monospace', color=_COL_TEXT,
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                           edgecolor='#bbbbbb', alpha=0.9, linewidth=1.2)),
            dict(x=w - 12, y=60, text='\n'.join(right_lines),
                 fontsize=11, fontfamily='monospace', color=_COL_TEXT, ha='right',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                           edgecolor='#bbbbbb', alpha=0.9, linewidth=1.2)),
        ]
        overlay = _mpl_text_overlay(w, h, text_blocks)
        pil_img = pil_img.convert('RGBA')
        pil_img = Image.alpha_composite(pil_img, overlay)
        pil_img = pil_img.convert('RGB')

        adi = draw_adi_fast(phi, theta_att, psi, size=140)
        pil_img.paste(adi, (12, pil_img.height - 150), adi)

        att_sphere = render_attitude_sphere(
            body_mesh_base, body_pts_orig, tail_pts, fin8_faces,
            fin_configs, wing_pts, wing_radials,
            place_tail_fin, place_wing,
            phi, theta_att, psi, elev_rad, rud_rad,
            body_color, fin_color, size=260)
        pil_img.paste(att_sphere, (pil_img.width - 265, pil_img.height - 265))

        d_hdg = ImageDraw.Draw(pil_img)
        sphere_x = pil_img.width - 265
        sphere_y = pil_img.height - 265
        hdg = np.degrees(psi) % 360
        phi_d_g = np.degrees(phi)
        th_d_g = np.degrees(theta_att)

        dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        dir_label = dirs[int((hdg + 22.5) % 360 / 45)]
        hdg_line = f"{dir_label} {hdg:.0f}°"

        box_y = sphere_y - 38
        box_x = sphere_x
        box_w = 260
        box_h = 34
        d_hdg.rounded_rectangle([box_x, box_y, box_x + box_w, box_y + box_h],
                                radius=5, fill=(255, 255, 255), outline=(187, 187, 187))
        d_hdg.text((box_x + 8, box_y + 4), f"HDG  {hdg_line}", fill=(34, 34, 34))
        d_hdg.text((box_x + 8, box_y + 18), f"Roll {phi_d_g:+.1f}°", fill='#dd5555')
        d_hdg.text((box_x + 115, box_y + 18), f"Pitch {th_d_g:+.1f}°", fill='#55bb55')

        frames.append(pil_img)

    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=25, loop=0, optimize=True)
    print(f"  Saved: {path}  (vehicle view, {reason})")



def make_combined_gif(traj, path, ep_num):
    """Generate engagement view GIF with UAV body, target sphere, and flight trails.

    Args:
        traj: trajectory dict with keys 'UAV', 'target', 'attitudes', 'fins',
              'flight_data', 'target_flight_data', 'time', 'reason'.
        path: output GIF file path.
        ep_num: episode number (displayed in title).
    """
    try:
        import pyvista as pv
        from PIL import Image, ImageDraw
    except ImportError:
        print(f"  PyVista/Pillow not available, skipping combined GIF: {path}")
        return

    try:
        pv.start_xvfb()
    except Exception:
        pass

    pos_arr = np.array(traj["UAV"])
    tgt_arr = np.array(traj["target"])
    att_arr = np.array(traj["attitudes"])
    fin_arr = np.array(traj["fins"])
    flight_data_arr = np.array(traj["flight_data"])
    tgt_flight_data_arr = np.array(traj["target_flight_data"])
    times = np.array(traj["time"])
    reason = traj["reason"]

    vg = traj.get("vehicle_geometry", {})
    L = vg.get("length_m", 6.25)
    R_b = vg.get("radius_m", 0.155)
    CG = vg.get("cg_m", 3.13)
    nose_len = vg.get("nose_frac", 0.15) * L
    fin_span = vg.get("fin_span_m", 0.63)
    fin_start = 0.88 * L
    fin_chord = L - fin_start
    fin_root_x = CG - fin_start

    total = len(pos_arr)
    n_frames = 120
    step = max(1, total // n_frames)
    indices = list(range(0, total, step))
    if indices[-1] != total - 1:
        indices.append(total - 1)
    indices.extend([total - 1] * 8)

    sim_duration = times[-1] - times[0]
    gif_duration = len(indices) * 0.025
    speedup = sim_duration / gif_duration if gif_duration > 0 else 1

    rho_ogive = (R_b ** 2 + nose_len ** 2) / (2 * R_b)
    x_nose = np.linspace(0, nose_len, 30)
    r_nose = np.sqrt(rho_ogive ** 2 - (nose_len - x_nose) ** 2) + R_b - rho_ogive
    r_nose[0] = 0.001
    x_cyl = np.linspace(nose_len, L, 10)[1:]
    r_cyl = np.full_like(x_cyl, R_b)
    x_prof = CG - np.concatenate([x_nose, x_cyl])
    r_prof = np.concatenate([r_nose, r_cyl])

    n_theta = 33
    theta = np.linspace(0, 2 * np.pi, n_theta)
    X = np.outer(np.ones_like(theta), x_prof).T
    Y = np.outer(np.cos(theta), r_prof).T
    Z = np.outer(np.sin(theta), r_prof).T
    body_mesh_base = pv.StructuredGrid(X, Y, Z).extract_surface(algorithm=None)
    body_pts_orig = body_mesh_base.points.copy()

    t_thick = 0.008
    sweep = 0.18
    tip_chord = fin_chord * 0.55
    tail_pts = np.array([
        [0, 0, -t_thick], [fin_chord, 0, -t_thick],
        [sweep + tip_chord, fin_span, -t_thick], [sweep, fin_span, -t_thick],
        [0, 0, t_thick], [fin_chord, 0, t_thick],
        [sweep + tip_chord, fin_span, t_thick], [sweep, fin_span, t_thick],
    ])
    fin8_faces = np.hstack([
        [4, 0, 3, 2, 1], [4, 4, 5, 6, 7],
        [4, 0, 1, 5, 4], [4, 2, 3, 7, 6],
        [4, 0, 4, 7, 3], [4, 1, 2, 6, 5],
    ])
    fin_configs = [
        (0, 'rud'), (np.pi / 2, 'elev'),
        (np.pi, 'rud'), (3 * np.pi / 2, 'elev'),
    ]

    wing_x = CG - 2.0
    wing_chord = 0.7
    wing_span = 0.40
    wing_sweep = 0.35
    wing_tip_chord = wing_chord * 0.3
    wing_pts = np.array([
        [wing_x, 0, -t_thick], [wing_x + wing_chord, 0, -t_thick],
        [wing_x + wing_sweep + wing_tip_chord, wing_span, -t_thick],
        [wing_x + wing_sweep, wing_span, -t_thick],
        [wing_x, 0, t_thick], [wing_x + wing_chord, 0, t_thick],
        [wing_x + wing_sweep + wing_tip_chord, wing_span, t_thick],
        [wing_x + wing_sweep, wing_span, t_thick],
    ])
    wing_radials = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    def rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def body_to_enu_mat(phi, th, psi):
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(th), np.sin(th)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        R_nb = np.array([
            [cth * cpsi, cth * spsi, -sth],
            [sphi * sth * cpsi - cphi * spsi, sphi * sth * spsi + cphi * cpsi, sphi * cth],
            [cphi * sth * cpsi + sphi * spsi, cphi * sth * spsi - sphi * cpsi, cphi * cth],
        ])
        T_ne = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        return T_ne @ R_nb.T

    def place_tail_fin(pts, radial, defl):
        p = pts.copy()
        p = (rx(defl) @ p.T).T
        p[:, 1] += R_b
        p = (rx(radial) @ p.T).T
        p[:, 0] += fin_root_x
        return p

    def place_wing(pts, radial):
        p = pts.copy()
        p[:, 1] += R_b
        p = (rx(radial) @ p.T).T
        return p

    smooth_w = min(21, max(1, total // 10))
    fin_smooth = np.copy(fin_arr)
    att_smooth = np.copy(att_arr)
    fd_smooth = np.copy(flight_data_arr)

    def _moving_avg(arr, w):
        pad = w // 2
        cs = np.cumsum(np.insert(arr, 0, 0))
        smoothed = (cs[w:] - cs[:-w]) / w
        out = arr.copy()
        out[pad:pad + len(smoothed)] = smoothed
        return out

    if smooth_w >= 3:
        for col in range(fin_arr.shape[1]):
            fin_smooth[:, col] = _moving_avg(fin_arr[:, col], smooth_w)
        for col in range(att_arr.shape[1]):
            unwrapped = np.unwrap(att_arr[:, col])
            att_smooth[:, col] = _moving_avg(unwrapped, smooth_w)
        for col in range(flight_data_arr.shape[1]):
            fd_smooth[:, col] = _moving_avg(flight_data_arr[:, col], smooth_w)

    body_color = '#a8c4d8'
    fin_color = '#90b0c4'

    all_pts = np.vstack([pos_arr, tgt_arr])
    scene_size = np.max(np.ptp(all_pts, axis=0))
    uav_scale = scene_size / 40.0

    frames = []
    for fi, idx in enumerate(indices):
        p = pos_arr[idx]
        phi, theta_att, psi = att_smooth[idx]
        elev_rad, rud_rad = fin_smooth[idx]
        fd = fd_smooth[idx]

        R_be = body_to_enu_mat(phi, theta_att, psi)

        pl = pv.Plotter(off_screen=True, window_size=[900, 450])
        pl.set_background('white')

        if idx > 0:
            uav_pts = pos_arr[:idx+1]
            uav_line = pv.Spline(uav_pts, min(500, len(uav_pts)))
            pl.add_mesh(uav_line.tube(radius=scene_size * 0.003),
                        color=_COL_UAV, smooth_shading=True)

        if idx > 0:
            tgt_pts = tgt_arr[:idx+1]
            tgt_line = pv.Spline(tgt_pts, min(500, len(tgt_pts)))
            pl.add_mesh(tgt_line.tube(radius=scene_size * 0.003),
                        color=_COL_TARGET, smooth_shading=True)

        tgt_sphere = pv.Sphere(radius=scene_size * 0.012, center=tgt_arr[idx])
        pl.add_mesh(tgt_sphere, color=_COL_TARGET, smooth_shading=True)

        world_body = (R_be @ (body_pts_orig * uav_scale).T).T + p
        bm = body_mesh_base.copy()
        bm.points = world_body
        pl.add_mesh(bm, color=body_color, smooth_shading=True,
                     show_edges=False, specular=0.3)

        for radial, ctrl_type in fin_configs:
            defl = elev_rad if ctrl_type == 'elev' else rud_rad
            local = place_tail_fin(tail_pts, radial, defl) * uav_scale
            wf = (R_be @ local.T).T + p
            pl.add_mesh(pv.PolyData(wf, fin8_faces.copy()),
                        color=fin_color, smooth_shading=True, show_edges=False)

        for radial in wing_radials:
            local = place_wing(wing_pts, radial) * uav_scale
            ww = (R_be @ local.T).T + p
            pl.add_mesh(pv.PolyData(ww, fin8_faces.copy()),
                        color=fin_color, smooth_shading=True, show_edges=False)

        los_pts = np.array([pos_arr[idx], tgt_arr[idx]])
        los_line = pv.Line(los_pts[0], los_pts[1])
        pl.add_mesh(los_line, color='#999999', line_width=1, opacity=0.4)

        fwd = R_be @ np.array([1, 0, 0])
        fwd_horiz = np.array([fwd[0], fwd[1], 0])
        fwd_norm = np.linalg.norm(fwd_horiz)
        if fwd_norm > 1e-6:
            fwd_horiz /= fwd_norm
        else:
            fwd_horiz = np.array([1, 0, 0])
        cam_back = -fwd_horiz
        cam_offset = (cam_back + np.array([0, 0, 0.4])) * scene_size * 0.6
        cam_offset += np.cross(np.array([0, 0, 1]), cam_back) * scene_size * 0.15
        cam_pos = p + cam_offset
        focal = p + fwd_horiz * scene_size * 0.2
        pl.camera.position = cam_pos
        pl.camera.focal_point = focal
        pl.camera.up = (0, 0, 1)
        pl.camera.clipping_range = (0.1, scene_size * 10)

        img = pl.screenshot(return_img=True)
        pl.close()

        pil_img = Image.fromarray(img)
        w, h = pil_img.size

        alt, airspeed, mach = fd[0], fd[1], fd[2]
        p_dps, q_dps, r_dps = np.degrees(fd[3]), np.degrees(fd[4]), np.degrees(fd[5])
        elev_deg = np.degrees(elev_rad)
        rud_deg = np.degrees(rud_rad)
        rng_to_tgt = np.linalg.norm(pos_arr[idx] - tgt_arr[idx])
        if idx > 0:
            rng_prev = np.linalg.norm(pos_arr[idx - 1] - tgt_arr[idx - 1])
            dt_f = times[idx] - times[idx - 1]
            closing_speed = (rng_prev - rng_to_tgt) / dt_f if dt_f > 0 else 0
        else:
            closing_speed = 0
        g_load = airspeed * np.sqrt(fd[4]**2 + fd[5]**2) / 9.81 if airspeed > 1 else 0

        tfd = tgt_flight_data_arr[idx]
        tgt_speed = tfd[0]

        px, py, pz = p
        phi_d, th_d, psi_d = np.degrees(phi), np.degrees(theta_att), np.degrees(psi)
        left_lines = [
            f"time:      {times[idx]:>8.2f} s",
            f"altitude:  {alt:>8.1f} m",
            f"speed:     {airspeed:>8.1f} m/s",
            f"mach:      {mach:>8.2f}",
            f"range:     {rng_to_tgt:>8.0f} m",
            f"closing:   {closing_speed:>+8.0f} m/s",
            f"load:      {g_load:>8.1f} g",
            f"tgt speed: {tgt_speed:>8.0f} m/s",
        ]
        if idx == total - 1:
            result_str = "HIT" if reason == "hit" else reason.upper()
            left_lines.append(f"result:    {result_str:>8s}")

        right_lines = [
            f"position (m):    {px:>+9.1f} {py:>+9.1f} {pz:>+9.1f}",
            f"attitude (deg):  {phi_d:>+9.1f} {th_d:>+9.1f} {psi_d:>+9.1f}",
            f"rates (deg/s):   {p_dps:>+9.1f} {q_dps:>+9.1f} {r_dps:>+9.1f}",
            f"elevator (deg):  {elev_deg:>+9.2f}",
            f"rudder (deg):    {rud_deg:>+9.2f}",
        ]

        text_blocks = [
            dict(x=w // 2, y=6, text='Engagement View',
                 fontsize=15, fontweight='bold', ha='center', fontfamily='sans-serif',
                 color=_COL_TEXT),
            dict(x=w // 2, y=28, text=f'Episode {ep_num}  ({speedup:.0f}x real time)',
                 fontsize=12, ha='center', fontfamily='sans-serif', color='#666666'),
            dict(x=12, y=60, text='\n'.join(left_lines),
                 fontsize=12, fontfamily='monospace', color=_COL_TEXT,
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                           edgecolor='#bbbbbb', alpha=0.9, linewidth=1.2)),
            dict(x=w - 12, y=60, text='\n'.join(right_lines),
                 fontsize=11, fontfamily='monospace', color=_COL_TEXT, ha='right',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                           edgecolor='#bbbbbb', alpha=0.9, linewidth=1.2)),
        ]

        if idx == total - 1:
            result_str = "HIT" if reason == "hit" else reason.upper()
            text_blocks.append(
                dict(x=w // 2, y=h - 40, text=result_str,
                     fontsize=20, fontweight='bold', ha='center', fontfamily='sans-serif',
                     color=_COL_HIT if reason == 'hit' else '#ff8800'))

        overlay = _mpl_text_overlay(w, h, text_blocks)
        pil_img = pil_img.convert('RGBA')
        pil_img = Image.alpha_composite(pil_img, overlay)
        pil_img = pil_img.convert('RGB')

        frames.append(pil_img)

    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=25, loop=0, optimize=True)
    print(f"  Saved: {path}  (combined view, {reason})")


def _run_demo_episodes(model, scenario_name, n_episodes):
    """Run demo episodes with aggressive target maneuvers for GIF generation.

    Args:
        model: loaded RecurrentPPO model.
        scenario_name: scenario ID string (e.g. 'A', 'B', 'C').
        n_episodes: number of episodes to run.
    """
    scenario_path = find_scenario(scenario_name)
    conf, label = load_scenario(scenario_path)

    conf.initial_conditions.range_min = 3000.0
    conf.initial_conditions.range_max = 5000.0
    conf.initial_conditions.azimuth_min = -40.0
    conf.initial_conditions.azimuth_max = 40.0
    conf.initial_conditions.elevation_min = -25.0
    conf.initial_conditions.elevation_max = 25.0
    conf.target_maneuver.maneuver_interval_min = 0.0
    conf.target_maneuver.maneuver_interval_max = 2.0
    conf.target_maneuver.heading_change_max = 60.0
    conf.target_maneuver.alt_change_max = 2500.0
    conf.target_maneuver.throttle = 0.7
    conf.initial_conditions.target_speed_min = 300.0
    conf.initial_conditions.target_speed_max = 450.0
    conf.curriculum_file = None
    conf.reward_params.hit_radius_start = 0.0
    conf.reward_params.hit_radius_end = 0.0

    env = UAVGuidanceEnv(conf=conf)
    trajectories = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        done = False

        ref_lat = env.UAV.get_lat_gc_deg()
        ref_lon = env.UAV.get_long_gc_deg()
        ref_alt = env.UAV.get_altitude()

        traj = {"UAV": [], "target": [], "time": [], "reason": "",
                "attitudes": [], "fins": [], "flight_data": [],
                "target_flight_data": []}

        def record_pos():
            me, mn, mu = pm.geodetic2enu(
                env.UAV.get_lat_gc_deg(), env.UAV.get_long_gc_deg(),
                env.UAV.get_altitude(), ref_lat, ref_lon, ref_alt, deg=True)
            traj["UAV"].append([me, mn, mu])
            te, tn, tu = pm.geodetic2enu(
                env.target.get_lat_gc_deg(), env.target.get_long_gc_deg(),
                env.target.get_altitude(), ref_lat, ref_lon, ref_alt, deg=True)
            traj["target"].append([te, tn, tu])
            traj["time"].append(env.sim_time)
            traj["attitudes"].append([
                env.UAV.get_phi(in_deg=False),
                env.UAV.get_theta(in_deg=False),
                env.UAV.get_psi(in_deg=False),
            ])
            traj["fins"].append([
                env.UAV.fdm['fcs/elevator-pos-rad'],
                env.UAV.fdm['fcs/rudder-pos-rad'],
            ])
            traj["target_flight_data"].append([
                env.target.get_true_airspeed(),
                env.target.get_mach(),
            ])
            traj["flight_data"].append([
                env.UAV.get_altitude(),
                env.UAV.get_true_airspeed(),
                env.UAV.get_mach(),
                env.UAV.get_p_rad_sec(),
                env.UAV.get_q_rad_sec(),
                env.UAV.get_r_rad_sec(),
            ])

        record_pos()
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start,
                deterministic=True)
            episode_start = np.zeros((1,), dtype=bool)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            record_pos()

        traj["reason"] = info.get("termination_reason", "unknown")
        traj["final_range"] = info.get("range", float("inf"))
        geom = getattr(env.UAV_config, 'geometry', None)
        if geom:
            traj["vehicle_geometry"] = {
                "length_m": geom.get('length_in', 246.0) * 0.0254,
                "radius_m": geom.get('diameter_ft', 1.02) * 0.3048 / 2,
                "cg_m": geom.get('cg_x_in', 123.2) * 0.0254,
                "nose_frac": 0.15,
                "fin_span_m": geom.get('wingspan_ft', 4.13) * 0.3048 / 2,
            }
        trajectories.append(traj)

    env.close()
    return trajectories, label


def generate_demo_gifs(model, scenario_names, n_gifs, eval_dir):
    """Select best episodes per scenario and generate all three GIF types.

    Args:
        model: loaded RecurrentPPO model.
        scenario_names: list of scenario ID strings.
        n_gifs: number of GIFs to generate per scenario.
        eval_dir: output directory for GIF files.
    """
    n_demo = max(n_gifs * 3, 15)

    total = 0
    for scenario in scenario_names:
        print(f"\nRunning {n_demo} demo episodes for {scenario}...")
        trajectories, label = _run_demo_episodes(model, scenario, n_demo)

        candidates = []
        for i, traj in enumerate(trajectories):
            candidates.append({
                "traj": traj,
                "is_hit": traj["reason"] == "hit",
                "length": len(traj["UAV"]),
            })

        hits = [c for c in candidates if c["is_hit"]]
        hits.sort(key=lambda x: x["length"], reverse=True)
        if hits:
            selected = hits[:n_gifs]
        else:
            candidates.sort(key=lambda x: x["length"], reverse=True)
            selected = candidates[:n_gifs]
            print(f"  Warning: no hits for {scenario}, using best non-hit episodes")

        if not selected:
            print(f"No trajectories for {scenario}, skipping.")
            continue

        gif_dir = os.path.join(eval_dir, f"{label}_gifs")
        os.makedirs(gif_dir, exist_ok=True)

        for i, c in enumerate(selected):
            tag = "hit" if c["is_hit"] else c["traj"]["reason"]
            fname = f"{label}_{i+1}_{tag}_trajectory_overview.gif"
            make_demo_gif(c["traj"], os.path.join(gif_dir, fname), ep_num=i + 1)
            veh_fname = f"{label}_{i+1}_{tag}_vehicle_closeup.gif"
            make_vehicle_gif(c["traj"], os.path.join(gif_dir, veh_fname), ep_num=i + 1)
            comb_fname = f"{label}_{i+1}_{tag}_engagement_view.gif"
            make_combined_gif(c["traj"], os.path.join(gif_dir, comb_fname), ep_num=i + 1)
        print(f"Generated {len(selected)} GIFs for {scenario} in {gif_dir}/")
        total += len(selected)

    print(f"Total: {total} demo GIFs")


def main():
    """CLI entry point: parse args, load model, evaluate scenarios, generate GIFs."""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--run", type=str)

    parser.add_argument("--scenarios", nargs="+", default=["all"])
    parser.add_argument("--holdout", nargs="*", default=[])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-dir", type=str, default=None)
    parser.add_argument("--gifs", type=int, default=10, metavar="N")
    args = parser.parse_args()

    if args.run:
        model_path = find_best_model(args.run)
        if not model_path:
            print(f"No model found in {args.run}")
            sys.exit(1)
        run_dir = args.run
    else:
        model_path = args.model
        if not model_path.endswith(".zip"):
            model_path_zip = model_path + ".zip"
        else:
            model_path_zip = model_path
        if not os.path.isfile(model_path_zip):
            print(f"Model not found: {model_path_zip}")
            sys.exit(1)
        run_dir = os.path.join(os.path.dirname(model_path), "..")

    if args.eval_dir:
        eval_dir = args.eval_dir
    else:
        run_name = os.path.basename(os.path.normpath(run_dir))
        scen_tag = "_".join(args.scenarios)
        gifs_tag = f"_{args.gifs}gifs" if args.gifs > 0 else ""
        timestamp = datetime.now().strftime("%b%d_%H%M")
        eval_name = f"{timestamp}_{run_name}_eval_{scen_tag}_{args.episodes}ep{gifs_tag}"
        eval_dir = os.path.join("evaluate_logs", eval_name)
    os.makedirs(eval_dir, exist_ok=True)

    meta_config_path = os.path.join(run_dir, "meta_config.json")
    holdout_set = set(args.holdout)
    if os.path.isfile(meta_config_path) and not args.holdout:
        with open(meta_config_path) as f:
            mc = json.load(f)
        holdout_set = set(mc.get("holdout_scenarios", []))

    if args.scenarios == ["all"]:
        scenario_names = list_scenarios()
    else:
        scenario_names = args.scenarios

    print(f"Loading model: {model_path}")
    model = RecurrentPPO.load(model_path)

    results = []
    for name in scenario_names:
        is_holdout = name in holdout_set
        tag = " [HOLDOUT]" if is_holdout else ""
        print(f"\nEvaluating scenario {name}{tag} ({args.episodes} episodes)...")
        result = evaluate_scenario(
            model, name, n_episodes=args.episodes, holdout=is_holdout)
        results.append(result)
        print(f"  -> hit_rate={result['hit_rate']:.1%}  "
              f"miss_mean={result['miss_distance']['mean']:.1f}m  "
              f"reward={result['reward']['mean']:.1f}")

    print_comparison_table(results)

    save_results = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != "trajectories"}
        save_results.append(r_copy)

    results_path = os.path.join(eval_dir, "meta_eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
            "n_episodes_per_scenario": args.episodes,
            "holdout_scenarios": list(holdout_set),
            "results": save_results,
        }, f, indent=2)
    print(f"\nSaved: {results_path}")

    for r in save_results:
        name = r["scenario"]
        label = r["label"]
        tag = "_holdout" if r["holdout"] else ""
        txt_path = os.path.join(eval_dir, f"{name}_{label}{tag}_evaluation.txt")
        with open(txt_path, "w") as f:
            f.write(f"{'='*50}\n")
            f.write(f"EVALUATION: {label}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Model:      {model_path}\n")
            f.write(f"Timestamp:  {datetime.now().isoformat()}\n")
            f.write(f"Scenario:   {name} ({label})\n")
            f.write(f"Holdout:    {r['holdout']}\n")
            f.write(f"Episodes:   {r['n_episodes']}\n\n")
            f.write(f"{'='*50}\n")
            f.write(f"HIT RATE\n")
            f.write(f"{'='*50}\n")
            f.write(f"Hit Rate:   {r['hit_rate']:.1%}\n")
            f.write(f"Hits:       {r['hit_count']} / {r['n_episodes']}\n\n")
            f.write(f"{'='*50}\n")
            f.write(f"MISS DISTANCE\n")
            f.write(f"{'='*50}\n")
            md = r["miss_distance"]
            f.write(f"Mean:       {md['mean']:.1f} m\n")
            f.write(f"Median:     {md['median']:.1f} m\n")
            f.write(f"Min:        {md['min']:.1f} m\n")
            f.write(f"Std:        {md['std']:.1f} m\n\n")
            f.write(f"{'='*50}\n")
            f.write(f"REWARD\n")
            f.write(f"{'='*50}\n")
            rw = r["reward"]
            f.write(f"Mean:       {rw['mean']:.2f}\n")
            f.write(f"Std:        {rw['std']:.2f}\n\n")
            f.write(f"{'='*50}\n")
            f.write(f"TERMINATION REASONS\n")
            f.write(f"{'='*50}\n")
            for reason, data in r["termination_reasons"].items():
                f.write(f"{reason:<20s} {data['count']:>4d}  ({data['pct']:.1%})\n")
            f.write(f"{'='*50}\n")
        print(f"Saved: {txt_path}")

    if args.gifs > 0:
        generate_demo_gifs(model, scenario_names, args.gifs, eval_dir)


if __name__ == "__main__":
    main()
