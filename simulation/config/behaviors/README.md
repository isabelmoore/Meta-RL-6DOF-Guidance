# Target Behavior Configs

These control how the target flies. Scenarios point to them with behavior_config.


## Files

evasive.yaml
  Random heading/alt changes every 2-6s. Up to 45 deg
  heading, 2000m alt. Fighter-style.

evasive_mild.yaml
  Same idea but gentler. 3-8s intervals, 30 deg heading,
  1500m alt.

straight.yaml
  Target holds initial heading and altitude. No maneuvers.

ballistic.yaml
  No thrust, no control. Pure gravity arc. For ICBM-type
  targets like the RS-28.


## Parameters

maneuver_type           evasive, straight, or ballistic
maneuver_interval_min   min seconds between maneuvers
maneuver_interval_max   max seconds between maneuvers
heading_change_max      degrees per maneuver (evasive only)
alt_change_max          meters per maneuver (evasive only)
throttle                target throttle (0.0 = engine off)


## Making a new one

Copy any file, tweak the numbers, point your scenario at it:

  behavior_config: "simulation/config/behaviors/my_behavior.yaml"
