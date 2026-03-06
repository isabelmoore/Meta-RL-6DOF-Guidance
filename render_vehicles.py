# Copyright (c) 2026 Isabel Moore. All rights reserved.
"""Render a visual catalog of all UAV configs as a single comparison image."""

import os
import glob
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
import pyvista as pv
pv.global_theme.allow_empty_mesh = True
from PIL import Image, ImageDraw, ImageFont
from simulation.core.config_loader import ConfigLoader


def _make_fin_set(chord, span, sweep, t, root_x, count=4, offset_deg=0):
    """Build a set of fins evenly spaced around the body."""
    angles = [np.radians(offset_deg) + i * 2 * np.pi / count for i in range(count)]
    tip_chord = chord * 0.55
    pts = np.array([
        [0, 0, -t], [chord, 0, -t],
        [sweep + tip_chord, span, -t], [sweep, span, -t],
        [0, 0, t], [chord, 0, t],
        [sweep + tip_chord, span, t], [sweep, span, t],
    ])
    faces = np.hstack([
        [4, 0, 3, 2, 1], [4, 4, 5, 6, 7],
        [4, 0, 1, 5, 4], [4, 2, 3, 7, 6],
        [4, 0, 4, 7, 3], [4, 1, 2, 6, 5],
    ])
    fins = []
    for angle in angles:
        ca, sa = np.cos(angle), np.sin(angle)
        rot = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        rotated = (pts @ rot.T)
        rotated[:, 0] += root_x
        fins.append(pv.PolyData(rotated, faces))
    return fins


def build_mesh(L, R_b, CG, nose_len, fin_span, fin_chord, fin_start, tail_fin_count=4, wings=None):
    """Build ogive body + tail fins + optional mid-body wings as PyVista meshes."""
    rho = (R_b**2 + nose_len**2) / (2 * R_b)
    x_nose = np.linspace(0, nose_len, 40)
    r_nose = np.sqrt(rho**2 - (nose_len - x_nose)**2) + R_b - rho
    r_nose[0] = 0.001
    x_cyl = np.linspace(nose_len, L, 12)[1:]
    r_cyl = np.full_like(x_cyl, R_b)
    x_prof = CG - np.concatenate([x_nose, x_cyl])
    r_prof = np.concatenate([r_nose, r_cyl])

    n_theta = 33
    theta = np.linspace(0, 2 * np.pi, n_theta)
    X = np.outer(np.ones_like(theta), x_prof).T
    Y = np.outer(np.cos(theta), r_prof).T
    Z = np.outer(np.sin(theta), r_prof).T
    try:
        body = pv.StructuredGrid(X, Y, Z).extract_surface(algorithm=None)
    except TypeError:
        body = pv.StructuredGrid(X, Y, Z).extract_surface()

    # Tail fins
    fin_root_x = CG - fin_start
    fins = _make_fin_set(fin_chord, fin_span, 0.18, 0.008, fin_root_x,
                         count=tail_fin_count)

    # Mid-body wings (if configured)
    if wings:
        wing_count = wings.get('count', 4)
        wing_pos = wings.get('position_frac', 0.4) * L
        wing_span = wings.get('span_m', 0.5)
        wing_chord = wings.get('chord_m', 0.5)
        wing_sweep = wings.get('sweep', 0.3)
        wing_root_x = CG - wing_pos
        # Offset wings 45° from tail fins
        fins += _make_fin_set(wing_chord, wing_span, wing_sweep, 0.006,
                              wing_root_x, count=wing_count, offset_deg=45)

    return body, fins


def render_vehicle(conf, width=600, height=400):
    """Render a single vehicle config and return a PIL Image."""
    geom = conf.geometry
    L = geom.get('length_in', 144.0) * 0.0254
    R_b = geom.get('diameter_ft', 0.667) * 0.3048 / 2
    CG = geom.get('cg_x_in', 72.0) * 0.0254
    nose_len = 0.15 * L
    fin_span = geom.get('wingspan_ft', 3.33) * 0.3048 / 2
    fin_start = 0.88 * L
    fin_chord = L - fin_start

    tail_fin_count = geom.get('tail_fins', 4)
    wings = geom.get('wings', None)
    body, fins = build_mesh(L, R_b, CG, nose_len, fin_span, fin_chord, fin_start,
                            tail_fin_count=tail_fin_count, wings=wings)

    pl = pv.Plotter(off_screen=True, window_size=[width, height])
    pl.set_background('white')

    pl.add_mesh(body, color='#888888', smooth_shading=True,
                specular=0.5, specular_power=30)
    for f in fins:
        pl.add_mesh(f, color='#666666', smooth_shading=True,
                    specular=0.3, specular_power=20)

    cam_dist = L * 3.5
    pl.camera_position = [
        (0, -cam_dist * 0.4, cam_dist * 0.12),
        (0, 0, 0),
        (0, 0, 1),
    ]
    pl.render()
    img = pl.screenshot(return_img=True)
    pl.close()

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    mass = conf.mass
    name = conf.name
    weight_kg = mass.get('weight_lbs', 0) * 0.4536
    length_m = L
    diam_m = R_b * 2

    lines = [
        name,
        f"{weight_kg:.0f} kg  ({mass.get('weight_lbs', 0):.0f} lbs)",
        f"L = {length_m:.2f} m  |  D = {diam_m:.3f} m",
        f"Fins: {geom.get('wingspan_ft', 0):.2f} ft span",
    ]

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_title = ImageFont.load_default()
        font = font_title

    y = 12
    draw.text((14, y), lines[0], fill='#222222', font=font_title)
    y += 28
    for line in lines[1:]:
        draw.text((14, y), line, fill='#555555', font=font)
        y += 20

    return pil_img


def main():
    yaml_files = sorted(glob.glob("simulation/config/*.yaml"))
    uav_configs = []

    for yf in yaml_files:
        try:
            conf = ConfigLoader.load_config(yf)
            if conf.type == "UAV" and hasattr(conf, 'geometry') and conf.geometry:
                uav_configs.append(conf)
        except Exception:
            continue

    if not uav_configs:
        print("No UAV configs with geometry found.")
        return

    print(f"Rendering {len(uav_configs)} vehicles...")

    w, h = 600, 400
    images = []
    for conf in uav_configs:
        print(f"  {conf.name}")
        images.append(render_vehicle(conf, w, h))

    cols = min(len(images), 2)
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new('RGB', (w * cols, h * rows), color=(255, 255, 255))

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(img, (c * w, r * h))

    os.makedirs("demo", exist_ok=True)
    out = "demo/vehicle_catalog.png"
    canvas.save(out)
    print(f"Saved: {out} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
