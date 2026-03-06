# Reward Configs

Reward YAMLs live here. Scenarios point to them with reward_config.


## Files

gaudet.yaml
  Standard Gaudet & Furfaro reward. 50m hit radius,
  curriculum shrinks from 500m to 50m. Has proximity
  shaping and graded miss penalty. Good starting point.

gaudet_tight.yaml
  Same reward but 10m hit radius. Curriculum ramps from
  3000m down to 10m. No proximity/miss shaping. Use this
  for extended range or wide angle scenarios.


## Parameters

alpha                LOS rate shaping weight
sigma_omega          LOS rate shaping bandwidth
roll_rate_penalty    penalizes roll rate
ctrl_penalty         penalizes control effort
hit_bonus            reward when the UAV hits the target
violation_penalty    penalty for constraint violations
hit_radius           range below this = hit (meters)
hit_radius_start     curriculum start radius (m)
hit_radius_end       curriculum end radius (m)
curriculum_steps     steps over which radius anneals
proximity_weight     per-step closeness reward (0 = off)
miss_reward_scale    graded miss penalty scale (0 = off)


## Making a new one

Copy gaudet.yaml, change what you want, point your scenario at it:

  reward_config: "simulation/config/rewards/my_reward.yaml"

Reference: https://arxiv.org/pdf/2109.03880
