# Copyright (c) 2026 Isabel Moore. All rights reserved.
"""
Load scenario configurations from YAML files.

Usage:
    from simulation.core.scenario_loader import load_scenario, find_scenario, list_scenarios

    conf, label = load_scenario("scenarios/A.yaml")
    conf, label = load_scenario(find_scenario("A"))

Scenarios can inline reward_params/target_maneuver or reference external files:
    reward_config: "simulation/config/rewards/paper.yaml"
    behavior_config: "simulation/config/behaviors/evasive.yaml"
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
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../..")


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
        if f.endswith(".yaml") and not f.startswith("README")
    )


def _load_yaml_file(rel_path: str) -> dict:
    """Load a YAML file relative to project root."""
    full = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.isfile(full):
        raise FileNotFoundError(f"Config file not found: {rel_path} (resolved: {full})")
    with open(full) as f:
        return yaml.safe_load(f)


def _resolve_reward_params(data: dict) -> dict:
    """Get reward params from inline dict or external file reference."""
    if "reward_config" in data:
        params = _load_yaml_file(data["reward_config"])
        if "reward_params" in data:
            params.update(data["reward_params"])
        return params
    elif "reward_params" in data:
        return data["reward_params"]
    else:
        raise KeyError("Scenario must have either 'reward_config' or 'reward_params'")


def _resolve_behavior_params(data: dict) -> dict:
    """Get target behavior params from inline dict or external file reference."""
    if "behavior_config" in data:
        params = _load_yaml_file(data["behavior_config"])
        if "target_maneuver" in data:
            params.update(data["target_maneuver"])
        return params
    elif "target_maneuver" in data:
        return data["target_maneuver"]
    else:
        raise KeyError("Scenario must have either 'behavior_config' or 'target_maneuver'")


def load_scenario(yaml_path: str):
    """
    Read a scenario YAML and return (conf, label) identical to make_config_X().

    Supports two styles for reward_params and target_maneuver:
      1. Inline: reward_params: {alpha: 0.1, ...}
      2. File reference: reward_config: "simulation/config/rewards/paper.yaml"
         (inline values override the file if both are present)

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

    # reward and behavior: inline or file reference (file + inline overrides supported)
    reward_dict = _resolve_reward_params(data)
    behavior_dict = _resolve_behavior_params(data)

    conf.reward_params = UAVGuidanceRewardParams(**reward_dict)
    conf.path_constraints = UAVGuidancePathConstraints(**data["path_constraints"])
    conf.initial_conditions = UAVGuidanceInitialConditions(**data["initial_conditions"])
    conf.target_maneuver = TargetManeuverParams(**behavior_dict)

    # store config names for titles/logging
    conf.reward_config_name = data.get("reward_config", "inline")
    conf.behavior_config_name = data.get("behavior_config", "inline")
    conf.scenario_name = data.get("name", os.path.splitext(os.path.basename(yaml_path))[0])
    conf.description = data.get("description", "")

    # build a descriptive tag from vehicle configs and behavior
    uav_name = os.path.splitext(os.path.basename(conf.UAV_config_file))[0]
    target_name = os.path.splitext(os.path.basename(conf.target_config_file))[0]
    behavior_name = behavior_dict.get("maneuver_type", "evasive")
    conf.summary = f"{uav_name} vs {target_name} ({behavior_name})"

    label = data.get("label", data.get("name", os.path.splitext(os.path.basename(yaml_path))[0]))
    return conf, label
