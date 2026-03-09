# Target Behavior Configs

Controls how the target flies. Set per-scenario via `behavior_config`.

## Files

| File | Type | Description |
|------|------|-------------|
| evasive.yaml | evasive | +/-45 deg heading, +/-2000 m alt, every 2-6s |
| evasive_mild.yaml | evasive | +/-30 deg heading, +/-1500 m alt, every 3-8s |
| straight.yaml | straight | Holds heading and altitude |
| ballistic.yaml | ballistic | No thrust, no control, gravity arc |

To add a new behavior, copy an existing file and update your scenario:

    behavior_config: "simulation/config/behaviors/my_behavior.yaml"
