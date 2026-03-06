# Copyright (c) 2026 Isabel Moore. All rights reserved.
"""
Meta-RL (RL²) wrapper around UAVGuidanceEnv.

Each reset() samples a random scenario from the pool. The LSTM policy must
infer the task (vehicle dynamics, target behavior, engagement geometry)
from within-episode experience.

Usage:
    env = MetaUAVGuidanceEnv(
        scenario_paths=["scenarios/A.yaml", "scenarios/D.yaml", ...],
    )
"""
import numpy as np
from simulation.environments.uav_guidance_env import UAVGuidanceEnv
from simulation.core.scenario_loader import load_scenario
from simulation.core.config_loader import ConfigLoader


class MetaUAVGuidanceEnv(UAVGuidanceEnv):
    """
    RL² meta-learning environment.

    On each reset(), a scenario is sampled uniformly (or weighted) from the
    pool.  The parent class already recreates FDM objects every reset(), so
    swapping configs before super().reset() is safe.
    """

    def __init__(self, scenario_paths, scenario_weights=None,
                 curriculum_file=None, n_envs=8, total_timesteps=20_000_000):
        # ---- Load all scenario configs + vehicle configs once ----
        self._scenarios = []       # list of (conf, label)
        self._vehicle_cache = {}   # path -> loaded config (avoid reloading)

        for path in scenario_paths:
            conf, label = load_scenario(path)
            # Pre-load vehicle configs
            uav_path = conf.UAV_config_file
            tgt_path = conf.target_config_file
            if uav_path not in self._vehicle_cache:
                self._vehicle_cache[uav_path] = ConfigLoader.load_config(uav_path)
            if tgt_path not in self._vehicle_cache:
                self._vehicle_cache[tgt_path] = ConfigLoader.load_config(tgt_path)
            self._scenarios.append((conf, label))

        if not self._scenarios:
            raise ValueError("MetaUAVGuidanceEnv needs at least one scenario")

        # Sampling weights (uniform if not specified)
        if scenario_weights is not None:
            w = np.array(scenario_weights, dtype=np.float64)
            self._scenario_probs = w / w.sum()
        else:
            self._scenario_probs = None  # uniform

        # Shared training params
        self._meta_curriculum_file = curriculum_file
        self._meta_n_envs = n_envs
        self._meta_total_timesteps = total_timesteps

        # ---- Initialise parent with the first scenario ----
        first_conf, first_label = self._scenarios[0]
        first_conf.curriculum_file = curriculum_file
        first_conf.n_envs = n_envs
        first_conf.total_timesteps = total_timesteps
        super().__init__(first_conf)
        self._current_scenario_label = first_label
        self._current_scenario_idx = 0

    # ------------------------------------------------------------------
    # Reset — sample a new scenario, swap configs, then delegate
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        # Sample scenario
        rng = np.random.default_rng(seed)
        idx = int(rng.choice(len(self._scenarios), p=self._scenario_probs))
        conf, label = self._scenarios[idx]

        # Swap all config attributes the parent reads
        self.conf = conf
        self.ic = conf.initial_conditions
        self.rp = conf.reward_params
        self.tm = conf.target_maneuver
        self.pc = conf.path_constraints
        self.reward_type = getattr(conf, 'reward_type', 'gaudet')

        # Swap vehicle configs from cache
        self.UAV_config = self._vehicle_cache[conf.UAV_config_file]
        self.target_config = self._vehicle_cache[conf.target_config_file]

        # Propagate shared training params for curriculum
        self.conf.curriculum_file = self._meta_curriculum_file
        self.conf.n_envs = self._meta_n_envs
        self.conf.total_timesteps = self._meta_total_timesteps
        self._curriculum_file = self._meta_curriculum_file
        self._n_envs = self._meta_n_envs
        self._total_timesteps = self._meta_total_timesteps

        # Track which scenario is active
        self._current_scenario_label = label
        self._current_scenario_idx = idx

        return super().reset(seed=seed, options=options)

    # ------------------------------------------------------------------
    # Step — delegate to parent, enrich info
    # ------------------------------------------------------------------
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info["scenario_label"] = self._current_scenario_label
        info["scenario_idx"] = self._current_scenario_idx
        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def scenario_labels(self):
        """Return list of all scenario labels in the pool."""
        return [label for _, label in self._scenarios]

    @property
    def n_scenarios(self):
        return len(self._scenarios)
