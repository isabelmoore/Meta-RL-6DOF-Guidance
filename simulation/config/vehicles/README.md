# Vehicle Configs

Each YAML defines mass, geometry, aero coefficients, and FCS limits.
JSBSim XML is auto-generated at runtime.

## Files

| File | Description |
|------|-------------|
| aim7.yaml | AIM-7 Sparrow, 231 kg, 3.66 m, 4 tail + 4 mid-body fins |
| gaudet.yaml | Gaudet vehicle, 455 kg, 6.25 m, 3 swept tail fins |
| f16.yaml | F-16 target, JSBSim built-in flight model |
| rs28_sarmat.yaml | RS-28 Sarmat ballistic target, 208100 kg, 35.5 m |

To add a new vehicle, copy an existing file and update your scenario:

    UAV_config_file: "simulation/config/vehicles/my_vehicle.yaml"
