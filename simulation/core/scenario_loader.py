# Copyright (c) 2026 Isabel Moore. All rights reserved.
"""
Load scenario configurations from YAML files.

Usage:
    from simulation.core.scenario_loader import load_scenario, find_scenario, list_scenarios

    conf, label = load_scenario("scenarios/A.yaml")
    conf, label = load_scenario(find_scenario("A"))
"""
import os
import types
import yaml

from data_classes.uav_guidance_dataclass import (
    UAVGuidanceInitialConditions,
    UAVGuidancePathConstraints,
    UAVGuidanceRewardParams,
    TargetManeuverParams,
)

SCENARIOS_DIR = os.path.join(os.path.dirname(__file__), "../../scenarios")


def find_scenario(name_or_path: str) -> str:
    """Resolve a scenario name (e.g. 'A') or path to a full YAML path."""
    if os.path.isfile(name_or_path):
        return name_or_path

    candidate = os.path.join(SCENARIOS_DIR, f"{name_or_path}.yaml")
    if os.path.isfile(candidate):
        return candidate

    raise FileNotFoundError(
        f"Scenario '{name_or_path}' not found. "
        f"Tried: {name_or_path}, {candidate}"
    )


def list_scenarios() -> list:
    """List all available scenario names from the scenarios/ directory."""
    scenarios_dir = os.path.normpath(SCENARIOS_DIR)
    if not os.path.isdir(scenarios_dir):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(scenarios_dir)
        if f.endswith(".yaml")
    )


def load_scenario(yaml_path: str):
    """
    Read a scenario YAML and return (conf, label) identical to make_config_X().

    Returns:
        conf: SimpleNamespace with all fields the environment expects
        label: str scenario label (e.g. "A_paper_ICs")
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    conf = types.SimpleNamespace()

    conf.observation_shape = tuple(data["observation_shape"])
    conf.action_shape = tuple(data["action_shape"])
    conf.fdm_steps_per_action = data["fdm_steps_per_action"]
    conf.max_episode_time = data["max_episode_time"]
    conf.action_scale = data["action_scale"]

    conf.UAV_config_file = data["UAV_config_file"]
    conf.target_config_file = data["target_config_file"]
    conf.guidance_type = data["guidance_type"]
    conf.autopilot_type = data["autopilot_type"]
    conf.reward_type = data["reward_type"]

    conf.reward_params = UAVGuidanceRewardParams(**data["reward_params"])
    conf.path_constraints = UAVGuidancePathConstraints(**data["path_constraints"])
    conf.initial_conditions = UAVGuidanceInitialConditions(**data["initial_conditions"])
    conf.target_maneuver = TargetManeuverParams(**data["target_maneuver"])

    label = data.get("label", data.get("name", os.path.splitext(os.path.basename(yaml_path))[0]))
    return conf, label
