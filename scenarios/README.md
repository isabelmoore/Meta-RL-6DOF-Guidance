# Scenarios

Each YAML here defines an engagement — vehicles, guidance, reward,
target behavior, and initial conditions.


## Fields

name                 scenario ID for --scenarios (A, B, C, ...)
label                shows up in logs and plot titles
description          one-liner about the scenario

UAV_config_file      interceptor vehicle (aim7.yaml, gaudet.yaml)
target_config_file   target vehicle (f16.yaml, rs28_sarmat.yaml)

guidance_type        pro_nav, APN, ZEM, pure_pursuit
autopilot_type       UAVPIDAutopilot, AircraftPIDAutopilot
reward_type          reward function name (gaudet)

reward_config        path to reward YAML
behavior_config      path to behavior YAML


## Reward and behavior configs

Instead of inlining everything, scenarios point to external YAMLs:

  reward_config: "simulation/config/rewards/gaudet.yaml"
  behavior_config: "simulation/config/behaviors/evasive.yaml"

You can still add inline reward_params or target_maneuver to override
specific values from the file.

See simulation/config/rewards/README.md and
simulation/config/behaviors/README.md for what's available.


## Guidance laws

pro_nav / PN             proportional navigation, nulls LOS rate
APN / augmented_pro_nav  PN + target acceleration compensation
ZEM / zero_effort_miss   optimal for constant-velocity targets
pure_pursuit             always points at target, simple but suboptimal


## Target maneuver types

evasive      random heading/alt changes on a timer
straight     holds initial heading and altitude
ballistic    no thrust, no control, gravity arc


## Initial conditions

Randomization bounds, sampled fresh each episode:

  range_min/max             separation (meters)
  UAV_speed_min/max         interceptor speed (m/s)
  target_speed_min/max      target speed (m/s)
  UAV_alt_min/max           interceptor altitude (meters)
  target_alt_min/max        target altitude (meters)
  elevation_min/max         LOS elevation angle (degrees)
  azimuth_min/max           LOS azimuth offset (degrees)


## Path constraints

Flight envelope limits. Episode terminates if violated.


## Example

  name: F
  label: my_scenario
  description: "AIM-7 vs RS-28 with APN guidance"

  UAV_config_file: "simulation/config/vehicles/aim7.yaml"
  target_config_file: "simulation/config/vehicles/rs28_sarmat.yaml"
  guidance_type: "APN"
  autopilot_type: "UAVPIDAutopilot"
  reward_type: "gaudet"
  reward_config: "simulation/config/rewards/gaudet.yaml"
  behavior_config: "simulation/config/behaviors/ballistic.yaml"

  # copy initial_conditions and path_constraints from an existing scenario

Reference: https://arxiv.org/pdf/2109.03880
