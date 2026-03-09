# Scenarios

Each YAML defines an engagement. Copy one to make a new scenario.

## Fields

| Field | Options |
|-------|---------|
| UAV_config_file | aim7.yaml, gaudet.yaml |
| target_config_file | f16.yaml, rs28_sarmat.yaml |
| guidance_type | pro_nav, APN, ZEM, pure_pursuit |
| autopilot_type | UAVPIDAutopilot, AircraftPIDAutopilot |
| reward_config | simulation/config/rewards/*.yaml |
| behavior_config | simulation/config/behaviors/*.yaml |

## Initial conditions

Randomized each episode:

| Field | Unit |
|-------|------|
| range_min/max | meters |
| UAV_speed_min/max | m/s |
| target_speed_min/max | m/s |
| UAV_alt_min/max | meters |
| target_alt_min/max | meters |
| elevation_min/max | degrees |
| azimuth_min/max | degrees |
