# Copyright (c) 2026 Isabel Moore. All rights reserved.
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymap3d as pm

from simulation.models.uavs import UAV
from simulation.models.aircraft import Aircraft
from simulation.core.units import f2m
from simulation.core.navigation import delta_heading
from simulation.core.config_loader import ConfigLoader
from simulation.core.factory import ComponentFactory


class UAVGuidanceEnv(gym.Env):
    """
    RL agent IS the UAV guidance system.
    Action: [aileron, elevator, rudder] in [-1, 1] — direct fin commands.
    Observation: 23-dim vector (LOS, rates, attitude, body rates, accel, fins, throttle, speed).
    """

    def __init__(self, conf):
        """Initialize environment from a scenario configuration.

        Args:
            conf: SimpleNamespace with observation_shape, action_shape, initial_conditions,
                  path_constraints, reward_params, target_maneuver, and vehicle config paths.
        """
        super().__init__()
        self.conf = conf
        self.obs_shape = conf.observation_shape
        self.act_shape = conf.action_shape
        
        # Load UAV and target configurations
        UAV_conf_path = getattr(conf, 'UAV_config_file', 'simulation/config/aim7.yaml')
        target_conf_path = getattr(conf, 'target_config_file', 'simulation/config/f16.yaml')
        
        self.UAV_config = ConfigLoader.load_config(UAV_conf_path)
        self.target_config = ConfigLoader.load_config(target_conf_path)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.act_shape, dtype=np.float32
        )

        self.ic = conf.initial_conditions
        self.pc = conf.path_constraints
        self.rp = conf.reward_params
        self.tm = conf.target_maneuver
        
        # Determine reward type (default to 'paper' if not specified)
        self.reward_type = getattr(conf, 'reward_type', 'paper')

        # Curriculum: adaptive (file-based) or fixed-schedule (step-based)
        self._step_count = 0
        self._n_envs = getattr(conf, 'n_envs', 4)
        self._total_timesteps = getattr(conf, 'total_timesteps', 5_000_000)
        self._curriculum_file = getattr(conf, 'curriculum_file', None)

        # Will be initialized in reset()
        self.UAV = None
        self.target = None
        self.done = False
        self.termination_reason = ""

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Reset environment with randomized engagement geometry.

        Args:
            seed: Optional RNG seed for reproducibility.
            options: Unused, present for Gymnasium API compatibility.

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)
        rng = self.np_random

        # --- Resolve Components ---
        # Allow overrides from the environment config (conf)
        m_ap_name = getattr(self.conf, 'autopilot_type', self.UAV_config.autopilot_type)
        m_guid_name = getattr(self.conf, 'guidance_type', self.UAV_config.guidance_type)
        
        # Simple alias handling
        if m_guid_name == "pro_nav": m_guid_name = "PN"

        UAVAutopilot = ComponentFactory.get_autopilot(m_ap_name)
        UAVGuidance = ComponentFactory.get_guidance(m_guid_name)
        
        # --- Create FDM objects ---
        self.UAV = UAV(self.UAV_config, autopilot_cls=UAVAutopilot, guidance_cls=UAVGuidance)
        
        if self.target_config.type == 'UAV':
            # Check target overrides if we ever need them (e.g. conf.target_autopilot_type)
            # For now, stick to yaml defaults for target to avoid confusion
            t_ap_name = self.target_config.autopilot_type
            t_guid_name = self.target_config.guidance_type
            
            TargetAutopilot = ComponentFactory.get_autopilot(t_ap_name)
            TargetGuidance = ComponentFactory.get_guidance(t_guid_name)
            self.target = UAV(self.target_config, autopilot_cls=TargetAutopilot, guidance_cls=TargetGuidance)
        else:
            t_ap_name = self.target_config.autopilot_type
            TargetAutopilot = ComponentFactory.get_autopilot(t_ap_name)
            self.target = Aircraft(self.target_config, autopilot_cls=TargetAutopilot)

        # --- Random engagement geometry (UAV heading points at target) ---
        UAV_alt = rng.uniform(self.ic.UAV_alt_min, self.ic.UAV_alt_max)
        UAV_speed = rng.uniform(self.ic.UAV_speed_min, self.ic.UAV_speed_max)

        target_speed = rng.uniform(self.ic.target_speed_min, self.ic.target_speed_max)
        target_heading = rng.uniform(0.0, 360.0)

        engagement_range = rng.uniform(self.ic.range_min, self.ic.range_max)
        elevation_deg = rng.uniform(self.ic.elevation_min, self.ic.elevation_max)
        azimuth_deg = rng.uniform(self.ic.azimuth_min, self.ic.azimuth_max)

        # Engagement axis: random world direction, target at axis + azimuth offset
        engagement_axis = rng.uniform(0.0, 360.0)
        target_bearing = engagement_axis + azimuth_deg
        # UAV heading points at target — ensures initial look angle ≈ 0
        UAV_heading = target_bearing % 360.0

        # UAV at reference position
        UAV_lat = self.ic.ref_lat
        UAV_lon = self.ic.ref_lon

        # Target position from engagement geometry (range, elevation, azimuth)
        az_rad = np.radians(target_bearing)
        el_rad = np.radians(elevation_deg)
        horiz_range = engagement_range * np.cos(el_rad)
        vert_offset = engagement_range * np.sin(el_rad)

        # ENU offset from UAV to target
        target_e = horiz_range * np.sin(az_rad)
        target_n = horiz_range * np.cos(az_rad)
        target_alt = UAV_alt + vert_offset
        target_alt = np.clip(target_alt, self.ic.target_alt_min, self.ic.target_alt_max)

        # Convert ENU offset to lat/lon
        target_lat, target_lon, _ = pm.enu2geodetic(
            target_e, target_n, 0.0,
            UAV_lat, UAV_lon, UAV_alt,
            deg=True,
        )

        # Reset FDMs
        self.UAV.reset(UAV_lat, UAV_lon, UAV_alt, UAV_speed, UAV_heading)
        self.target.reset(target_lat, target_lon, target_alt, target_speed, target_heading)
        self.target.set_target(self.UAV) # UAV target knows about the agent (for simulation mechanics if needed)

        # --- Internal state ---
        self.done = False
        self.termination_reason = ""
        self.sim_time = 0.0
        self.throttle = 0.0  # Post-burnout: no engine (paper-matched)

        # LOS rate tracking (finite diff) — for observation
        self._prev_los_unit = None
        self._prev_range = None
        self._diverge_timer = 0.0

        # Separate tracking for shaping reward (independent of obs computation)
        self._prev_los_unit_shaping = None
        self._prev_range_shaping = None

        # Track minimum range achieved (for graded terminal reward)
        self._min_range = float("inf")

        # Curriculum milestone: bonus given once when crossing curriculum radius
        self._crossed_curriculum = False

        # Curriculum hit radius (read from file if adaptive, else step-based)
        self._current_hit_radius = self._read_curriculum_radius()

        # Target maneuver command state
        self._target_heading_cmd = target_heading
        self._target_alt_cmd = target_alt
        self._next_maneuver_time = rng.uniform(
            self.tm.maneuver_interval_min, self.tm.maneuver_interval_max
        )

        # Compute initial observation
        obs = self._compute_observation()
        return obs, {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        """Advance simulation by one agent step (multiple FDM sub-steps).

        Args:
            action: Array [aileron, elevator, rudder] in [-1, 1].

        Returns:
            Tuple of (obs, reward, done, truncated, info).
        """
        action = np.clip(action, -1.0, 1.0)

        self._step_count += 1
        # Step-based curriculum (only used if no adaptive file)
        if self._curriculum_file is None:
            self._current_hit_radius = self._get_curriculum_hit_radius()

        reward = 0.0

        # Paper: agent outputs rate commands, FCS integrates to positions
        # Clamp aileron to zero — cruciform UAV, roll not needed for guidance
        # (roll dynamics too aggressive for exploration: Cl_da*q*S*b/Ixx >> constraint)
        action[0] = 0.0
        scaled_action = action * self.conf.action_scale

        for _ in range(self.conf.fdm_steps_per_action):
            # Command UAV fins (rate commands — FCS integrates to positions)
            self.UAV.command_UAV(
                scaled_action[0], scaled_action[1], scaled_action[2], 0.0
            )
            self.UAV.fdm.run()

            # Step target with PID autopilot
            self._step_target()

            self.sim_time = self.UAV.get_sim_time_sec()

            # Check termination (also returns milestone bonuses even if not done)
            term_reward = self._check_termination()
            reward += term_reward
            if self.done:
                break

            # Per-substep shaping reward
            reward += self._compute_shaping_reward(action)

        # Guard against NaN rewards
        if not np.isfinite(reward):
            reward = 0.0
        reward = float(reward)  # ensure plain Python float for SB3

        obs = self._compute_observation()
        truncated = self.sim_time >= self.conf.max_episode_time and not self.done
        if truncated:
            self.done = True
            self.termination_reason = "timeout"

        info = {
            "termination_reason": self.termination_reason,
            "sim_time": self.sim_time,
            "range": self._get_range(),
            "min_range": self._min_range if np.isfinite(self._min_range) else -1.0,
            "current_hit_radius": self._current_hit_radius,
            "crossed_curriculum": self._crossed_curriculum,
            "scenario_label": getattr(self, '_current_scenario_label', ''),
        }

        return obs, reward, self.done, truncated, info

    # ------------------------------------------------------------------
    # Target stepping (clock-synchronized with UAV FDM)
    # ------------------------------------------------------------------
    def _step_target(self):
        """Step target FDM one frame, issuing randomized maneuver commands on schedule."""
        # Randomize target maneuver commands periodically
        if self.sim_time >= self._next_maneuver_time:
            rng = self.np_random
            heading_change = rng.uniform(
                -self.tm.heading_change_max, self.tm.heading_change_max
            )
            self._target_heading_cmd = (self.target.get_psi() + heading_change) % 360.0
            alt_change = rng.uniform(
                -self.tm.alt_change_max, self.tm.alt_change_max
            )
            self._target_alt_cmd = np.clip(
                self.target.get_altitude() + alt_change,
                self.conf.initial_conditions.target_alt_min,
                self.conf.initial_conditions.target_alt_max,
            )
            self._next_maneuver_time = self.sim_time + rng.uniform(
                self.tm.maneuver_interval_min, self.tm.maneuver_interval_max
            )

        
        # Unified autopilot control
        ail, elev, rud, thr = self.target.get_autopilot_control(self._target_heading_cmd, self._target_alt_cmd)
        self.target.command_flight(ail, elev, rud, thr)
        
        self.target.fdm.run()

    # ------------------------------------------------------------------
    # Observation (22-dim)
    # ------------------------------------------------------------------
    def _compute_observation(self):
        """Build 23-dim observation vector from LOS geometry, attitude, rates, and fins."""
        # --- LOS vector in ENU frame ---
        m_lat = self.UAV.get_lat_gc_deg()
        m_lon = self.UAV.get_long_gc_deg()
        m_alt = self.UAV.get_altitude()
        t_lat = self.target.get_lat_gc_deg()
        t_lon = self.target.get_long_gc_deg()
        t_alt = self.target.get_altitude()

        e, n, u = pm.geodetic2enu(t_lat, t_lon, t_alt, m_lat, m_lon, m_alt, deg=True)
        los_enu = np.array([e, n, u])
        rng_dist = np.linalg.norm(los_enu)
        if rng_dist < 1e-6:
            rng_dist = 1e-6
        los_unit_enu = los_enu / rng_dist

        # Convert ENU to NED
        los_unit_ned = np.array([los_unit_enu[1], los_unit_enu[0], -los_unit_enu[2]])

        # Rotate NED to body frame using Euler angles
        los_body = self._ned_to_body(los_unit_ned)

        # [0:3] LOS unit vector in body frame
        obs_los = los_body

        # [3:6] LOS rotation rate (finite diff)
        if self._prev_los_unit is not None:
            dt = self.UAV.fdm['simulation/dt']
            if dt < 1e-9:
                dt = 0.00833
            los_rate = (los_body - self._prev_los_unit) / dt
        else:
            los_rate = np.zeros(3)
        self._prev_los_unit = los_body.copy()

        # [6] Closing speed (normalized)
        if self._prev_range is not None:
            dt = self.UAV.fdm['simulation/dt']
            if dt < 1e-9:
                dt = 0.00833
            closing_speed = -(rng_dist - self._prev_range) / dt
        else:
            closing_speed = 0.0
        self._prev_range = rng_dist
        # Normalize closing speed: typical range ~0-2000 m/s
        closing_speed_norm = np.clip(closing_speed / 2000.0, -1.0, 1.0)

        # [7] Range (normalized)
        range_norm = np.clip(rng_dist / self.ic.range_max, 0.0, 2.0) - 1.0  # maps [0, 2*range_max] to [-1, 1]

        # [8:12] Quaternion attitude
        quat = self.UAV.get_quaternion()

        # [12:15] Body angular velocity [p, q, r]
        p = self.UAV.get_p_rad_sec()
        q = self.UAV.get_q_rad_sec()
        r = self.UAV.get_r_rad_sec()
        # Normalize: typical max ~10 rad/s
        body_rates = np.array([p, q, r]) / 10.0

        # [15:18] Body acceleration [ax, ay, az]
        accel = self.UAV.get_body_accel_mps2()
        # Normalize: ~450 m/s^2 = 45g
        accel_norm = accel / 450.0

        # [18:21] Current fin deflections
        fins = np.array([
            self.UAV.get_aileron_pos(),
            self.UAV.get_elevator_pos(),
            self.UAV.get_rudder_pos(),
        ])

        # [21] Throttle
        thr = np.array([self.throttle / 0.7])  # normalized: 1.0 during burn, 0.0 after

        # [22] UAV speed (critical for energy management)
        UAV_speed = self.UAV.get_true_airspeed()
        speed_norm = np.clip(UAV_speed / 1000.0, 0.0, 2.0) - 1.0  # maps [0, 2000] to [-1, 1]

        obs = np.concatenate([
            obs_los,             # 0:3
            los_rate / 10.0,     # 3:6  normalize LOS rate
            [closing_speed_norm],  # 6
            [range_norm],        # 7
            quat,                # 8:12
            body_rates,          # 12:15
            accel_norm,          # 15:18
            fins,                # 18:21
            thr,                 # 21
            [speed_norm],        # 22
        ]).astype(np.float32)

        obs = np.clip(obs, -1.0, 1.0)
        # Guard against NaN from unstable FDM
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    # ------------------------------------------------------------------
    # NED to body frame rotation
    # ------------------------------------------------------------------
    def _ned_to_body(self, vec_ned):
        """Rotate a vector from NED frame to body frame using ZYX Euler DCM.

        Args:
            vec_ned: 3-element array in NED coordinates.

        Returns:
            3-element array in body-frame coordinates.
        """
        phi = self.UAV.get_phi(in_deg=False)
        theta = self.UAV.get_theta(in_deg=False)
        psi = self.UAV.get_psi(in_deg=False)

        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        # DCM: NED to Body (ZYX rotation)
        R = np.array([
            [cth * cpsi, cth * spsi, -sth],
            [sphi * sth * cpsi - cphi * spsi, sphi * sth * spsi + cphi * cpsi, sphi * cth],
            [cphi * sth * cpsi + sphi * spsi, cphi * sth * spsi - sphi * cpsi, cphi * cth],
        ])
        return R @ vec_ned

    # ------------------------------------------------------------------
    # Shaping reward — Paper Eq. 30a-30f (Gaudet & Furfaro 2023)
    # R = α·exp(-||Ω||²/σ²) - β·|ωx| - γ·||δ||
    # ------------------------------------------------------------------
    def _compute_shaping_reward(self, action):
        """Compute per-substep shaping reward (LOS-rate, closing, proximity, penalties).

        Args:
            action: Current action array [aileron, elevator, rudder].

        Returns:
            Scalar shaping reward for this sub-step.
        """
        # --- R_shaping = α · exp(-||Ω||² / σ²)  (Eq. 30a) ---
        m_lat = self.UAV.get_lat_gc_deg()
        m_lon = self.UAV.get_long_gc_deg()
        m_alt = self.UAV.get_altitude()
        t_lat = self.target.get_lat_gc_deg()
        t_lon = self.target.get_long_gc_deg()
        t_alt = self.target.get_altitude()
        e, n, u = pm.geodetic2enu(t_lat, t_lon, t_alt, m_lat, m_lon, m_alt, deg=True)
        los_enu = np.array([e, n, u])
        rng_dist = np.linalg.norm(los_enu)
        if rng_dist < 1e-6:
            rng_dist = 1e-6
        los_unit = los_enu / rng_dist

        # Track minimum range for graded terminal reward
        if rng_dist < self._min_range:
            self._min_range = rng_dist

        if self._prev_los_unit_shaping is not None:
            dt = self.UAV.fdm['simulation/dt']
            if dt < 1e-9:
                dt = 0.00833
            omega = (los_unit - self._prev_los_unit_shaping) / dt
            omega_sq = np.dot(omega, omega)
        else:
            omega_sq = 0.0
        self._prev_los_unit_shaping = los_unit.copy()
        prev_range = self._prev_range_shaping  # save before overwriting
        self._prev_range_shaping = rng_dist

        sigma = self.rp.sigma_omega
        sigma_sq = sigma ** 2
        r_shaping = self.rp.alpha * np.exp(-omega_sq / sigma_sq) if sigma_sq > 0 else 0.0

        # --- R_closing: reward for reducing range to target ---
        r_closing = 0.0
        if prev_range is not None:
            delta_range = prev_range - rng_dist  # positive = closing
            r_closing = 3.0 * delta_range / 1000.0  # 3.0 reward per km closed

        # --- R_proximity: funnel effect — stronger reward as range shrinks ---
        r_proximity = 0.0
        prox_w = self.rp.proximity_weight if self.rp.proximity_weight > 0 else 0.5
        r_proximity = prox_w / (1.0 + rng_dist / 1000.0)

        # --- R_rollrate = -β · |ωx|  (Eq. 30b) ---
        p = abs(self.UAV.get_p_rad_sec())
        r_roll = -self.rp.roll_rate_penalty * p

        # --- R_ctrl = -γ · ||action||  (Eq. 30c) ---
        r_ctrl = -self.rp.ctrl_penalty * np.linalg.norm(action)

        total = r_shaping + r_closing + r_proximity + r_roll + r_ctrl
        # Guard against NaN from unstable FDM states
        if not np.isfinite(total):
            return 0.0
        return total

    # ------------------------------------------------------------------
    # Termination checks
    # ------------------------------------------------------------------
    def _check_termination(self):
        """Check hit, fly-by, constraint violations, and timeout conditions.

        Returns:
            Scalar terminal/milestone reward (0.0 if no event this step).
        """
        rng_dist = self._get_range()
        reward = 0.0

        # Guard against NaN range
        if not np.isfinite(rng_dist):
            self.done = True
            self.termination_reason = "nan_state"
            return 0.0

        # --- Real hit: range < fixed hit radius → terminate with full bonus ---
        real_hit_radius = self.rp.hit_radius
        if rng_dist < real_hit_radius:
            self.done = True
            self.termination_reason = "hit"
            self._crossed_curriculum = True  # also counts as curriculum crossing

            return self.rp.hit_bonus

        # --- Curriculum milestone: crossed curriculum radius → one-time bonus, NO termination ---
        if not self._crossed_curriculum and rng_dist < self._current_hit_radius:
            self._crossed_curriculum = True
            # Give a milestone bonus (fraction of hit bonus) but keep episode going
            reward += self.rp.hit_bonus * 0.3

        # --- Paper termination: closing velocity negative (UAV passed target) ---
        if self._prev_range is not None:
            dt = self.UAV.fdm['simulation/dt']
            if dt < 1e-9:
                dt = 0.00833
            closing_vel = -(rng_dist - self._prev_range) / dt
            # Only trigger fly_by when close — use 500m fixed gate
            if closing_vel < -50.0 and rng_dist < 500.0:
                self.done = True
                self.termination_reason = "fly_by"
                # Graded terminal reward: closer miss = more reward
                miss = min(self._min_range, rng_dist)
                miss_scale = getattr(self.rp, 'miss_reward_scale', 500.0) or 500.0
                graded = self.rp.hit_bonus * np.exp(-miss / miss_scale)
                return reward + graded

        # --- Paper Eq. 30e: Path constraint violations → -Z penalty ---
        if self.UAV.get_true_airspeed() < self.pc.speed_min:
            self.done = True
            self.termination_reason = "speed_low"
            return reward - self.rp.violation_penalty

        if abs(self.UAV.get_theta()) > self.pc.pitch_max:
            self.done = True
            self.termination_reason = "pitch_violation"
            return reward - self.rp.violation_penalty

        if abs(self.UAV.get_phi()) > self.pc.roll_max:
            self.done = True
            self.termination_reason = "roll_violation"
            return reward - self.rp.violation_penalty

        beta_deg = abs(np.degrees(self.UAV.fdm['aero/beta-rad']))
        if beta_deg > self.pc.yaw_max:
            self.done = True
            self.termination_reason = "yaw_violation"
            return reward - self.rp.violation_penalty

        look_angle = self._get_look_angle()
        if look_angle > self.pc.look_angle_max:
            self.done = True
            self.termination_reason = "look_angle_violation"
            return reward - self.rp.violation_penalty

        nz = abs(self.UAV.get_n_pilot())
        if nz > self.pc.load_max:
            self.done = True
            self.termination_reason = "load_violation"
            return reward - self.rp.violation_penalty

        if self.UAV.get_altitude() < self.pc.alt_min:
            self.done = True
            self.termination_reason = "ground_impact"
            return reward - self.rp.violation_penalty

        # Timeout
        if self.sim_time >= self.conf.max_episode_time:
            self.done = True
            self.termination_reason = "timeout"
            # Graded reward based on closest approach — closer = more reward
            miss_scale = getattr(self.rp, 'miss_reward_scale', 500.0) or 500.0
            graded = self.rp.hit_bonus * np.exp(-self._min_range / miss_scale)
            return reward + graded

        return reward

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    def _read_curriculum_radius(self):
        """Read adaptive curriculum radius from file (written by callback).
        Falls back to step-based schedule if no file exists."""
        if self._curriculum_file is not None:
            try:
                with open(self._curriculum_file, 'r') as f:
                    return float(f.read().strip())
            except (FileNotFoundError, ValueError):
                pass  # file not yet created, use step-based
        return self._get_curriculum_hit_radius()

    def _get_curriculum_hit_radius(self):
        """Log-linear anneal from hit_radius_start → hit_radius_end.
        Each worker tracks its own steps, scaled by n_envs to estimate global progress."""
        if self.rp.hit_radius_start <= 0:
            return self.rp.hit_radius  # no curriculum, use fixed radius
        global_steps_est = self._step_count * self._n_envs
        progress = min(1.0, global_steps_est / (self._total_timesteps * 0.8))  # anneal over 80%, fine-tune 20%
        log_start = np.log(self.rp.hit_radius_start)
        log_end = np.log(max(self.rp.hit_radius_end, 1.0))  # guard log(0)
        return np.exp(log_start + progress * (log_end - log_start))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_range(self):
        """Return Euclidean distance from UAV to target in meters."""
        m_lat = self.UAV.get_lat_gc_deg()
        m_lon = self.UAV.get_long_gc_deg()
        m_alt = self.UAV.get_altitude()
        t_lat = self.target.get_lat_gc_deg()
        t_lon = self.target.get_long_gc_deg()
        t_alt = self.target.get_altitude()
        e, n, u = pm.geodetic2enu(t_lat, t_lon, t_alt, m_lat, m_lon, m_alt, deg=True)
        return np.linalg.norm(np.array([e, n, u]))

    def _get_look_angle(self):
        """Angle between UAV body X-axis and LOS vector, in degrees."""
        m_lat = self.UAV.get_lat_gc_deg()
        m_lon = self.UAV.get_long_gc_deg()
        m_alt = self.UAV.get_altitude()
        t_lat = self.target.get_lat_gc_deg()
        t_lon = self.target.get_long_gc_deg()
        t_alt = self.target.get_altitude()

        e, n, u = pm.geodetic2enu(t_lat, t_lon, t_alt, m_lat, m_lon, m_alt, deg=True)
        los_enu = np.array([e, n, u])
        rng_dist = np.linalg.norm(los_enu)
        if rng_dist < 1e-6:
            return 0.0
        los_unit_ned = np.array([los_enu[1], los_enu[0], -los_enu[2]])
        los_body = self._ned_to_body(los_unit_ned / rng_dist)

        # Body X-axis is [1, 0, 0], so look angle = acos(los_body[0])
        cos_angle = np.clip(los_body[0], -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
