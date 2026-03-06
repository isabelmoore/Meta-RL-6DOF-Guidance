from data_classes.uav_guidance_dataclass import (
    UAVGuidanceInitialConditions,
    UAVGuidancePathConstraints,
    UAVGuidanceRewardParams,
    UAVGuidanceRewardParams,
    TargetManeuverParams,
)

import os
# Default configuration files
UAV_config_file = os.path.join("simulation", "config", "aim7.yaml")
target_config_file = os.path.join("simulation", "config", "f16.yaml")

observation_shape = (23,)
action_shape = (3,)

fdm_steps_per_action = 10
max_episode_time = 30.0  # seconds
action_scale = 1.0       # FCS handles rate scaling (80 deg/s → ±20° via integrator)

# Paper Table 1: engagement geometry ranges
initial_conditions = UAVGuidanceInitialConditions(
    range_min=5000.0,
    range_max=10000.0,
    UAV_speed_min=800.0,
    UAV_speed_max=1000.0,
    target_speed_min=250.0,
    target_speed_max=600.0,
    UAV_alt_min=5000.0,
    UAV_alt_max=10000.0,
    target_alt_min=4000.0,
    target_alt_max=11000.0,
    elevation_min=-30.0,
    elevation_max=30.0,
    azimuth_min=-30.0,
    azimuth_max=30.0,
    ref_lat=58.0,
    ref_lon=18.0,
)

# Paper Table 2 — relaxed for exploration (tighten via curriculum later)
path_constraints = UAVGuidancePathConstraints(
    speed_min=400.0,
    pitch_max=85.0,
    yaw_max=85.0,
    roll_max=100.0,
    look_angle_max=90.0,     # relaxed from paper's 45 for exploration
    load_max=80.0,           # relaxed from paper's 45 for exploration
    alt_min=0.0,
)

# Paper Eq. 30a-30f — EXACT coefficients
reward_params = UAVGuidanceRewardParams(
    alpha=1.0,               # paper: shaping weight α = 1
    sigma_omega=0.02,        # paper: σ_Ω = 0.02
    roll_rate_penalty=0.05,  # paper: β = 0.05
    ctrl_penalty=0.01,       # paper: γ = 0.01
    hit_bonus=10.0,          # paper: κ = 10
    violation_penalty=10.0,  # paper: Z = 10
    hit_radius=10.0,         # paper: r_lim = 10 m
)

target_maneuver = TargetManeuverParams(
    maneuver_interval_min=2.0,
    maneuver_interval_max=6.0,
    heading_change_max=45.0,
    alt_change_max=2000.0,
    throttle=0.49,
)
