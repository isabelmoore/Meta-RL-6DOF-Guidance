# Copyright (c) 2026 Isabel Moore. All rights reserved.

import os
import sys
import glob
import warnings
import logging
import numpy as np
import yaml
warnings.filterwarnings("ignore")
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["VTK_SILENCE_GET_VOID_POINTER_WARNINGS"] = "1"
logging.getLogger("vtk").setLevel(logging.CRITICAL)
import vtkmodules.vtkRenderingOpenGL2  # noqa
import pyvista as pv
pv.global_theme.allow_empty_mesh = True
from PIL import Image, ImageDraw, ImageFont


def _progress(current, total, label="", width=40):
    frac = current / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r  {label} [{bar}] {current}/{total}")
    if current == total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def _span(arr):
    return float(arr.max() - arr.min())


def simulate_target(config, duration=25.0, dt=0.2):
    maneuver_type = config.get("maneuver_type", "straight")
    interval_min = config.get("maneuver_interval_min", 999.0)
    interval_max = config.get("maneuver_interval_max", 999.0)
    heading_max = config.get("heading_change_max", 0.0)
    alt_max = config.get("alt_change_max", 0.0)
    rng = np.random.RandomState(42)

    if maneuver_type == "ballistic":
        pos = np.array([0.0, 0.0, 25000.0])
        vel = np.array([320.0, 0.0, -120.0])
        pts = [pos.copy()]
        for _ in range(int(duration / dt)):
            vel[2] -= 9.81 * dt
            pos = pos + vel * dt
            if pos[2] < 0:
                pos[2] = 0
            pts.append(pos.copy())
        return np.array(pts)

    speed = 300.0
    heading = 0.0
    pos = np.array([0.0, 0.0, 7000.0])
    target_alt = 7000.0
    pts = [pos.copy()]
    next_maneuver = rng.uniform(interval_min, interval_max)
    t = 0.0

    for _ in range(int(duration / dt)):
        t += dt
        if t >= next_maneuver and maneuver_type == "evasive":
            heading += np.radians(rng.uniform(-heading_max, heading_max))
            target_alt += rng.uniform(-alt_max, alt_max) * 0.4
            target_alt = max(3000, min(target_alt, 11000))
            next_maneuver = t + rng.uniform(interval_min, interval_max)
        alt_err = target_alt - pos[2]
        pos[0] += speed * np.cos(heading) * dt
        pos[1] += speed * np.sin(heading) * dt
        pos[2] += np.clip(alt_err * 0.3, -200 * dt, 200 * dt)
        pts.append(pos.copy())

    return np.array(pts)



def _behavior_info(config):
    mt = config.get("maneuver_type", "unknown")
    if mt == "straight":
        return "Holds heading and altitude"
    elif mt == "ballistic":
        return "No thrust, gravity arc"
    else:
        hm = config.get("heading_change_max", 0)
        am = config.get("alt_change_max", 0)
        iv = f"{config.get('maneuver_interval_min', 0)}-{config.get('maneuver_interval_max', 0)}s"
        return f"hdg +/-{hm:.0f} deg, alt +/-{am:.0f}m, every {iv}"


def _suppress_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    return old_stderr


def _restore_stderr(old_stderr):
    os.dup2(old_stderr, 2)
    os.close(old_stderr)


def render_panel(pts_list, cmap_list, color_list, full_pts, progress, angle, w, h):
    """Render one panel. pts_list has paths truncated to current progress."""
    extent = max(_span(full_pts[:, 0]), _span(full_pts[:, 1]),
                 _span(full_pts[:, 2]), 1000)
    center = full_pts.mean(axis=0)
    view_dist = extent * 2.2
    tube_r = extent * 0.005
    sphere_r = extent * 0.015

    cam_x = center[0] + view_dist * 0.7 * np.cos(angle)
    cam_y = center[1] + view_dist * 0.7 * np.sin(angle)
    cam_z = center[2] + view_dist * 0.4

    old_stderr = _suppress_stderr()
    try:
        pl = pv.Plotter(off_screen=True, window_size=[w, h])
        pl.set_background('white')

        # ground + grid
        cx = (full_pts[:, 0].min() + full_pts[:, 0].max()) / 2
        cy = (full_pts[:, 1].min() + full_pts[:, 1].max()) / 2
        gsize = max(_span(full_pts[:, 0]), _span(full_pts[:, 1]), 1000) * 1.6
        ground = pv.Plane(center=(cx, cy, 0), i_size=gsize, j_size=gsize,
                          i_resolution=1, j_resolution=1)
        pl.add_mesh(ground, color='#f0f0f0', opacity=0.4)
        for i in range(6):
            frac = i / 5
            gx = cx - gsize / 2 + frac * gsize
            pl.add_mesh(pv.Line((gx, cy - gsize / 2, 1), (gx, cy + gsize / 2, 1)),
                        color='#e0e0e0', line_width=1)
            gy = cy - gsize / 2 + frac * gsize
            pl.add_mesh(pv.Line((cx - gsize / 2, gy, 1), (cx + gsize / 2, gy, 1)),
                        color='#e0e0e0', line_width=1)

        for pts, cmap, col in zip(pts_list, cmap_list, color_list):
            if len(pts) < 3:
                continue
            # trail
            spline = pv.Spline(pts, n_points=max(len(pts) * 2, 10))
            tube = spline.tube(radius=tube_r)
            tube["t"] = np.linspace(0, 1, tube.n_points)
            pl.add_mesh(tube, scalars="t", cmap=cmap,
                        smooth_shading=True, show_scalar_bar=False,
                        specular=0.3, specular_power=15)
            # shadow
            shadow = pts.copy()
            shadow[:, 2] = 2.0
            sh = pv.Spline(shadow, n_points=len(shadow))
            pl.add_mesh(sh.tube(radius=tube_r * 0.4), color='#dddddd', opacity=0.3)
            # current position marker
            pl.add_mesh(pv.Sphere(radius=sphere_r, center=pts[-1]),
                        color=col, smooth_shading=True)

        pl.camera_position = [(cam_x, cam_y, cam_z), tuple(center), (0, 0, 1)]
        pl.render()
        img = pl.screenshot(return_img=True)
        pl.close()
    finally:
        _restore_stderr(old_stderr)
    return Image.fromarray(img)


def main():
    yaml_files = sorted(glob.glob("simulation/config/behaviors/*.yaml"))
    configs = []
    for yf in yaml_files:
        try:
            with open(yf) as f:
                data = yaml.safe_load(f)
            if data and "maneuver_type" in data:
                configs.append((os.path.basename(yf), data))
        except Exception:
            continue

    if not configs:
        print("No behavior configs found.")
        return

    try:
        ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        ft = ImageFont.load_default()
        fs = ft

    print("Simulating trajectories...")
    target_data = []
    for name, config in configs:
        label = os.path.splitext(name)[0]
        info = _behavior_info(config)
        tgt_pts = simulate_target(config)
        target_data.append((label, info, tgt_pts))

    cols = min(len(configs), 2)
    rows = (len(configs) + cols - 1) // cols
    pw, ph = 500, 400
    n_frames = 30
    frame_ms = 120  # slow

    # --- target only ---
    print(f"\nRendering target GIF ({n_frames} frames)...")
    target_frames = []
    for fi in range(n_frames):
        progress = (fi + 1) / n_frames
        angle = np.pi * 0.3 * progress  # slow partial rotation
        canvas = Image.new('RGB', (pw * cols, ph * rows), (255, 255, 255))
        for idx, (label, info, tgt_pts) in enumerate(target_data):
            r, c = divmod(idx, cols)
            end = max(3, int(len(tgt_pts) * progress))
            panel = render_panel(
                [tgt_pts[:end]], ["YlOrRd"], ["#cc3333"],
                tgt_pts, progress, angle, pw, ph)
            draw = ImageDraw.Draw(panel)
            draw.text((10, 6), label, fill='#222222', font=ft)
            draw.text((10, 42), info, fill='#666666', font=fs)
            canvas.paste(panel, (c * pw, r * ph))
        _progress(fi + 1, n_frames, "target")
        target_frames.append(canvas)

    out = "simulation/config/behaviors/behavior_target.gif"
    target_frames[0].save(out, save_all=True, append_images=target_frames[1:],
                          duration=frame_ms, loop=0, optimize=True)
    print(f"  -> {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
