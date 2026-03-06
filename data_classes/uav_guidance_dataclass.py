from dataclasses import dataclass


@dataclass
class UAVGuidanceInitialConditions:
    """Randomization bounds for engagement initial conditions (ranges, speeds, angles)."""
    # Engagement geometry (paper Table 1)
    range_min: float          # meters (e.g. 5000)
    range_max: float          # meters (e.g. 10000)
    UAV_speed_min: float  # m/s (e.g. 800)
    UAV_speed_max: float  # m/s (e.g. 1000)
    target_speed_min: float   # m/s (e.g. 250)
    target_speed_max: float   # m/s (e.g. 600)
    UAV_alt_min: float    # meters
    UAV_alt_max: float    # meters
    target_alt_min: float     # meters
    target_alt_max: float     # meters
    elevation_min: float      # degrees (LOS elevation angle)
    elevation_max: float      # degrees
    azimuth_min: float        # degrees (LOS azimuth offset)
    azimuth_max: float        # degrees
    ref_lat: float            # reference latitude for UAV spawn
    ref_lon: float            # reference longitude for UAV spawn


@dataclass
class UAVGuidancePathConstraints:
    """Flight envelope constraints that trigger episode termination if violated."""
    speed_min: float          # m/s — below this: violation (paper: 400)
    pitch_max: float          # degrees — |pitch| > this: violation (paper: 85)
    yaw_max: float            # degrees — |beta| > this: violation (paper: 85)
    roll_max: float           # degrees — |roll| > this: violation (paper: 100)
    look_angle_max: float     # degrees — hard termination (paper: 45)
    load_max: float           # g's — hard termination (paper: 45)
    alt_min: float            # meters — below this: ground impact


@dataclass
class UAVGuidanceRewardParams:
    """Reward function weights and thresholds (paper Eq. 30a-30f)."""
    # Paper Eq. 30a-30f
    alpha: float              # shaping weight (paper: 1.0)
    sigma_omega: float        # LOS rate shaping bandwidth (paper: 0.02)
    roll_rate_penalty: float  # β — weight on |ωx| penalty (paper: 0.05)
    ctrl_penalty: float       # γ — weight on ||δ|| penalty (paper: 0.01)
    hit_bonus: float          # κ — terminal reward for hit (paper: 10)
    violation_penalty: float  # Z — terminal penalty for constraint violation (paper: 10)
    hit_radius: float         # r_lim — meters, range < this = hit (paper: 10)
    miss_reward_scale: float = 0.0   # exp decay scale for graded terminal reward (m), 0=off
    proximity_weight: float = 0.0    # per-step proximity shaping weight, 0=off
    # Curriculum hit radius (default 0 = disabled, use fixed hit_radius)
    hit_radius_start: float = 0.0    # starting hit radius for curriculum (m)
    hit_radius_end: float = 0.0      # ending hit radius (should match hit_radius)
    curriculum_steps: float = 4e6    # env steps over which to anneal radius


@dataclass
class TargetManeuverParams:
    """Parameters controlling randomized target evasive maneuvers."""
    maneuver_interval_min: float   # seconds between command changes
    maneuver_interval_max: float   # seconds between command changes
    heading_change_max: float      # degrees max heading change
    alt_change_max: float          # meters max altitude change
    throttle: float                # fixed target throttle (e.g. 0.49)
