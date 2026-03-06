<div align="center">

# Meta-RL 6-DOF UAV Guidance

Deep reinforcement learning for 6-DOF UAV guidance-to-intercept against maneuvering targets.

A single LSTM policy (RecurrentPPO) learns to control UAV fin commands across multiple engagement scenarios using meta-RL (RL²).

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-red.svg)](https://pytorch.org/)
[![JSBSim](https://img.shields.io/badge/sim-JSBSim-green.svg)](https://jsbsim.sourceforge.net/)
[![SB3](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)


<table align="center"><tr>
<td align="center"><img src="demo/vehicle_closeup.gif" width="420"/><br/><em>Chase camera with telemetry HUD</em></td>
<td align="center"><img src="demo/engagement_view.gif" width="420"/><br/><em>Engagement view with UAV body</em></td>
</tr></table>

</div>

## How It Works

[JSBSim](https://jsbsim.sourceforge.net/) provides 6-DOF flight dynamics. The RL agent outputs fin deflection commands each timestep, which are passed through a configurable guidance law and PID autopilot before reaching the flight dynamics model. The guidance law (e.g. proportional navigation) computes desired acceleration commands from the line-of-sight geometry, and the autopilot converts those into pitch/yaw fin deflections. The agent learns to augment or override these commands to minimize miss distance.

The policy is a RecurrentPPO with a 256-unit LSTM. Instead of training one policy per scenario, we use meta-RL (RL²) - a single policy trains across all scenarios simultaneously. Each episode the env randomly picks a scenario, so the LSTM hidden state learns to adapt on-the-fly to different engagement geometries, target behaviors, and speed regimes. The hidden state resets every episode, so the policy has to re-identify the scenario from observations alone.

## Vehicles

Vehicle physics are fully defined in YAML (`simulation/config/vehicles/`) - JSBSim XML is auto-generated at runtime from these configs. Each config specifies mass properties, body geometry (length, diameter, nose ogive shape), aerodynamic coefficients, propulsion (thrust curve, burn time, specific impulse), and fin layout (count, position, span, sweep, chord). To render a comparison image of all vehicles, run `python3 simulation/config/vehicles/render_vehicles.py`.

| Config | Name | Type | Mass | Length | Diameter | Fins |
|--------|------|------|------|--------|----------|------|
| `gaudet.yaml` | Gaudet | UAV | 455 kg | 6.25 m | 0.31 m | 3 tail (swept, at 82%) |
| `aim7.yaml` | AIM-7 Sparrow | UAV | 231 kg | 3.66 m | 0.20 m | 4 tail + 4 mid-body |
| `f16.yaml` | F-16 | aircraft | 9072 kg | 15.0 m | - | target only |
| `rs28_sarmat.yaml` | RS-28 Sarmat | UAV | 208100 kg | 35.5 m | 3.0 m | 4 tail (ballistic target) |

The Gaudet vehicle is based on [Gaudet & Furfaro (2023)](https://arxiv.org/pdf/2109.03880). The F-16 is not shown - its aerodynamics and geometry are handled natively by JSBSim's built-in F-16 flight model, not generated from YAML.

<p align="center">
  <img src="demo/vehicle_catalog.png" width="600" /><br/>
  <em>Vehicle catalog - all UAV configs rendered from geometry YAML</em>
</p>

## Target Behavior

Target maneuver patterns (`simulation/config/behaviors/`) define how the target aircraft moves during an episode. The target autopilot flies the aircraft (e.g. F-16) while the behavior config triggers random heading and altitude changes at configurable intervals. Each behavior YAML specifies the maneuver type, maximum heading/altitude deltas, maneuver interval range, and throttle setting. To render an animated preview of all behaviors, run `python3 simulation/config/behaviors/render_behaviors.py`.

| Config | Type | Heading | Altitude | Interval | Throttle |
|--------|------|---------|----------|----------|----------|
| `evasive.yaml` | evasive | ±45° | ±2000 m | 2–6 s | 0.49 |
| `evasive_mild.yaml` | evasive | ±30° | ±1500 m | 3–8 s | 0.49 |
| `straight.yaml` | straight | 0° | 0 m | - | 0.49 |
| `ballistic.yaml` | ballistic | 0° | 0 m | - | 0.0 |

<table align="center"><tr>
<td align="center"><img src="demo/behavior_target.gif" width="420"/><br/><em>Target behavior patterns</em></td>
<td align="center"><img src="demo/trajectory_overview.gif" width="420"/><br/><em>Full engagement trajectory</em></td>
</tr></table>

## Reward

Reward configs (`simulation/config/rewards/`) define the shaping weights, hit radius, and adaptive curriculum parameters used during training. Each config sets the terminal hit bonus, constraint violation penalty, LOS-rate shaping weights, and the curriculum schedule that shrinks the effective hit radius over training. See the reward function details under Scenarios below.

| Config | Hit Radius | Curriculum Start | Curriculum End | Steps |
|--------|-----------|-----------------|----------------|-------|
| `gaudet.yaml` | 50 m | 500 m | 50 m | 4M |
| `gaudet_tight.yaml` | 10 m | 3000 m | 10 m | 4M |

Both use hit_bonus=500, violation_penalty=25, α=0.1, σ=0.05.

## Navigation

The guidance law (`simulation/core/guidance_laws.py`) computes desired acceleration commands from the UAV-target line-of-sight geometry. At each timestep the navigation module measures the LOS vector and its rate of change, then the selected guidance law outputs an acceleration command that the autopilot converts into fin deflections. Configured per-scenario via `guidance_type`. Navigation gain `N` is set in the vehicle config (typically 3–5).

| Key | Name | Description |
|-----|------|-------------|
| `pro_nav` | Proportional Navigation | Steers to nullify LOS rate. `a_cmd = N * V_c * Ω` |
| `APN` | Augmented PN | PN + target acceleration compensation term |
| `ZEM` | Zero Effort Miss | Optimal for constant-velocity targets, minimizes predicted miss |
| `pure_pursuit` | Pure Pursuit | Points velocity vector directly at target position |

## Autopilot

The autopilot (`simulation/core/autopilot.py`) converts guidance acceleration commands into fin deflections using cascaded PID loops. The UAV autopilot takes pitch and yaw acceleration commands from the guidance law and produces elevator and rudder deflections through separate PID controllers. The aircraft autopilot is used for the target and provides full heading/altitude hold with roll/pitch/yaw control. Configured per-scenario via `autopilot_type`.

| Type | Used For | Description |
|------|----------|-------------|
| `UAVPIDAutopilot` | UAV interceptors | Pitch/yaw PID loops, aileron clamped to 0 |
| `AircraftPIDAutopilot` | Aircraft targets (F-16) | Full roll/pitch/yaw PID with heading and altitude hold |

## Configuration

Everything is YAML-driven. Each scenario references configs from the sections above. A scenario YAML ties everything together:

```yaml
# scenarios/A.yaml
UAV_config_file: "simulation/config/vehicles/gaudet.yaml"
target_config_file: "simulation/config/vehicles/f16.yaml"
guidance_type: "pro_nav"
autopilot_type: "UAVPIDAutopilot"
reward_type: "gaudet"
reward_config: "simulation/config/rewards/gaudet.yaml"
behavior_config: "simulation/config/behaviors/evasive_mild.yaml"

initial_conditions:
  range_min: 3000.0
  range_max: 5000.0
  # ... engagement geometry bounds
```

## Scenarios

| ID | UAV | Target | Behavior | Reward | Range | Angles |
|----|-----|--------|----------|--------|-------|--------|
| A | Gaudet (455 kg, 6.25 m) | F-16 | evasive_mild | gaudet | 3–5 km | ±15° |
| B | AIM-7 (231 kg, 3.66 m) | F-16 | evasive | gaudet_tight | 8–12 km | ±30° |
| C | AIM-7 | F-16 | evasive | gaudet_tight | 5–10 km | ±45° |
| D | AIM-7 | F-16 | straight | gaudet | 5–10 km | ±30° |
| E | AIM-7 | RS-28 Sarmat | ballistic | gaudet | 8–15 km | +10–45° |

- **A**: Short range, mild evasion, near head-on. Proof of concept.
- **B**: Long range intercept. Target at 250–600 m/s with aggressive evasion (±45° heading, ±2000m altitude every 2–6s).
- **C**: Wide off-boresight angles (±45° elevation and azimuth). Tests target acquisition from large initial look angles.
- **D**: Non-maneuvering target. Simplest intercept geometry.
- **E**: Ballistic intercept. RS-28 starts at 15–30 km altitude, 600–1200 m/s, diving under gravity. UAV looks up (+10–45° elevation).

All scenarios use `pro_nav` guidance and `UAVPIDAutopilot`. UAV speed is 800–1000 m/s across all scenarios. Scenarios A/D/E use 60s max episode time with 5 FDM steps per action. Scenarios B/C use 30s max episode time with 10 FDM steps per action.


### State Space (23-dim)

| Index | Variable | Description | Normalization |
|-------|----------|-------------|---------------|
| 0:3 | **LOS** | Line-of-sight unit vector (body frame) | unit vector |
| 3:6 | **Ω** | LOS rate (body frame) | ÷ 10 |
| 6 | **V_c** | Closing speed | ÷ 2000 m/s |
| 7 | **r** | Range to target | r / r_max, mapped to [-1, 1] |
| 8:12 | **q** | Quaternion attitude | unit quaternion |
| 12:15 | **[p, q, r]** | Body angular rates | ÷ 10 rad/s |
| 15:18 | **[a_x, a_y, a_z]** | Body accelerations | ÷ 450 m/s² |
| 18:21 | **[δ_a, δ_e, δ_r]** | Current fin deflections | [-1, 1] |
| 21 | **τ** | Throttle (1 = burning, 0 = coast) | ÷ 0.7 |
| 22 | **V** | UAV airspeed | V / 1000, mapped to [-1, 1] |

All observations clipped to [-1, 1].

### Action Space (3-dim)

| Channel | Control | Notes |
|---------|---------|-------|
| 0 | Aileron (δ_a) | Clamped to 0 (no roll command) |
| 1 | Elevator (δ_e) | Pitch fin deflection [-1, 1] |
| 2 | Rudder (δ_r) | Yaw fin deflection [-1, 1] |

### Reward Function

The shaping reward at each timestep is:

$R = \alpha \exp\!\left(-\frac{\|\dot{\hat{\lambda}}\|^2}{\sigma^2}\right) + R_{\text{closing}} + R_{\text{proximity}} - \beta\,|p| - \gamma\,\|\delta\|$

| Term | Expression | Purpose |
|------|-----------|---------|
| LOS rate | α · exp(-‖Ω‖² / σ²) | Reward small LOS rates (proportional navigation) |
| Closing | 3 · Δr / 1000 | Reward for reducing range (per km) |
| Proximity | w / (1 + r/1000) | Stronger reward as range shrinks |
| Roll penalty | -β · \|p\| | Penalize roll rate |
| Control penalty | -γ · ‖δ‖ | Penalize large fin deflections |

Default weights: α=0.1, σ=0.05, β=0.05, γ=0.01, w=0.5.

Terminal rewards:
- **Hit** (r < r_hit): +500
- **Fly-by** (closing velocity < -50 m/s within 500m): +500 · exp(-miss / 300)
- **Timeout**: +500 · exp(-min_range / 300)
- **Constraint violation** (speed, pitch, roll, yaw, load factor, altitude): -25

### Curriculum

Adaptive hit radius shared via file. Starts large and shrinks over `curriculum_steps` (4M) timesteps. Crossing the curriculum radius mid-episode gives a milestone bonus (+150) without terminating.

### Path Constraints

All scenarios terminate the episode if the UAV violates:
- Speed < 400 m/s
- Pitch or yaw > 85°
- Roll > 100°
- Look angle > 90°
- Load factor > 80 g
- Altitude < 0 m

## Quick Start

### 1. Start the container

```bash
docker compose up -d
docker exec -it meta-rl bash
```

The repo is mounted at `/wizard` inside the container. Base image is `python:3.11-slim` with NVIDIA GPU support.

### 2. Train

Edit `train.sh` to set your scenarios, GPU, timesteps, and parallel envs:

```bash
# train.sh
SCENARIOS="A B C"    # which scenarios "A" or "A B" or "A B C"
N_ENVS=8             # parallel environments
TIMESTEPS=30000000   # total training timesteps
GPU=1                # CUDA_VISIBLE_DEVICES
```

```bash
./train.sh
```

Or call `train_meta.py` directly:

```bash
python3 train_meta.py --scenarios A B C --timesteps 20000000 --n-envs 8 --lr 1e-4 --lstm-hidden 256
```

Training output goes to `training_logs/<timestamp>_<scenarios>_int-<uav>_tar-<target>_<guidance>_rew-<reward>_<timesteps>m/`.

Example: `Mar06_2100_A_B_C_int-gaudet_tar-f16_pro_nav_rew-gaudet_20m`

<details>
<summary>Hyperparameters</summary>

| Parameter | Default | Flag |
|:----------|:--------|:-----|
| Total timesteps | 20M | `--timesteps` |
| Parallel envs | 8 | `--n-envs` |
| Learning rate | 1e-4 | `--lr` |
| LSTM hidden size | 256 | `--lstm-hidden` |
| Batch size | 512 | - |
| N steps (rollout) | 2048 | `--n-steps` |
| N epochs | 5 | - |
| Gamma | 0.99 | - |
| GAE lambda | 0.92 | - |
| Entropy coef | 0.005 | `--ent-coef` |
| Target KL | 0.04 | `--target-kl` |
| VF coef | 1.0 | - |
| Max grad norm | 0.5 | - |
| Save frequency | 250k | `--save-freq` |
| Device | cuda | `--device` |
| Holdout scenarios | none | `--holdout` |
| Custom run dir | auto | `--run-dir` |

To change batch size, n_epochs, gamma, or GAE lambda, edit the constants in `train_meta.py` directly.

</details>

### 3. Monitor

```bash
tensorboard --logdir training_logs/
```

### 4. Evaluate

Edit `evaluate.sh` to point to your training run:

```bash
# evaluate.sh
RUN_DIR="training_logs/Mar03_0115_META_A_30M"
SCENARIOS="A"
EPISODES=50
GIFS=10
GPU=0
```

```bash
./evaluate.sh
```

Or call `evaluate_meta.py` directly:

```bash
python3 evaluate_meta.py --run training_logs/<run_dir> --scenarios all --episodes 50 --gifs 5
```

This prints a results table with hit rate, miss distance, and reward stats per scenario, and generates three GIF types per episode (trajectory overview, vehicle close-up, engagement view).

<details>
<summary>Eval parameters</summary>

| Parameter | Default | Flag |
|:----------|:--------|:-----|
| Episodes per scenario | 50 | `--episodes` |
| GIFs per scenario | 10 | `--gifs` |
| Scenarios | all | `--scenarios` |
| Custom output dir | auto | `--eval-dir` |
| Holdout tags | none | `--holdout` |
| Specific model | best_model.zip | `--model` |

</details>

Evaluation outputs to `evaluate_logs/<timestamp>_<run>_eval_<scenarios>_int-<uav>_tar-<target>_<guidance>_rew-<reward>_<episodes>ep_<gifs>gifs/`:
- `evaluation_results.txt` - combined results for all evaluated scenarios (hit rate, mean reward, miss distance)
- `meta_eval_results.json` - machine-readable results
- Per-scenario GIF folders: `<scenario>_int-<uav>_tar-<target>_<behavior>_gifs/`
  - `*_trajectory_overview.gif` - 3D bird's-eye trajectory with flight trails
  - `*_vehicle_closeup.gif` - chase camera with telemetry HUD, ADI, and attitude sphere
  - `*_engagement_view.gif` - scaled engagement view with UAV body and LOS line

Scenarios not in the training set are auto-tagged as holdout in the results.

### 5. Render config catalogs

```bash
# 3D vehicle comparison image
python3 simulation/config/vehicles/render_vehicles.py
# -> simulation/config/vehicles/vehicle_catalog.png

# animated 3D target behavior paths (2x2 rotating GIF)
python3 simulation/config/behaviors/render_behaviors.py
# -> simulation/config/behaviors/behavior_target.gif
```

### Naming Convention

Training and eval folders encode the full config:

```
<date>_<scenarios>_int-<uav>_tar-<target>_<guidance>_rew-<reward>_<timesteps>m
```

Example: `Mar06_2100_A_B_C_int-gaudet_tar-f16_pro_nav_rew-gaudet_20m`

Eval GIF subfolders use: `<scenario>_int-<uav>_tar-<target>_<behavior>_gifs`

## Project Structure

```
├── train_meta.py              training entry point (RecurrentPPO, multi-scenario)
├── evaluate_meta.py           evaluation + metrics + GIF generation
├── train.sh / evaluate.sh     shell wrappers (nohup + logging)
├── demo_intercept.py          standalone intercept animation demo
├── scenarios/                 scenario YAMLs (A–E)
├── simulation/
│   ├── config/
│   │   ├── vehicles/          vehicle configs + render_vehicles.py
│   │   ├── rewards/           reward configs (gaudet, gaudet_tight)
│   │   └── behaviors/         target behavior configs + render_behaviors.py
│   ├── core/                  scenario_loader, config_loader, guidance_laws, navigation,
│   │                          autopilot, control, geospatial, spatial, units, scale, utils
│   ├── environments/          gym environments (uav_guidance_env, meta_uav_guidance_env)
│   │   └── config/            env config dataclasses
│   └── models/                FDM wrappers around JSBSim (fdm_object, uavs, aircraft)
├── data_classes/              dataclasses for vehicle/guidance configs
├── jsbsim_data/               JSBSim aircraft XML, engines, scripts
├── demo/                      eval demo GIFs and config renders for README
├── tests/                     unit tests
├── Dockerfile                 python:3.11-slim + JSBSim + rendering deps
├── docker-compose.yml         GPU container config (meta-rl, mounted at /wizard)
├── requirements.txt           pinned dependencies
└── CITATION.cff               citation metadata
```

## Docker

```bash
docker compose up -d          # start (NVIDIA GPU, repo mounted at /wizard)
docker exec -it meta-rl bash  # exec in
docker compose down            # stop
```

## Requirements

From `requirements.txt`:

- Python 3.11+
- PyTorch 2.10.0
- Stable-Baselines3 2.7.1
- sb3-contrib 2.7.1 (RecurrentPPO)
- JSBSim 1.2.4
- gymnasium 1.2.3
- PyVista ≥0.43 (3D rendering)
- NumPy 2.4.2
- pandas 3.0.0
- matplotlib 3.10.8
- TensorBoard 2.17.1
- PyYAML ≥5.4
- pymap3d 3.2.0
- CUDA GPU recommended for training

## Reference

Gaudet, B. & Furfaro, R. (2023). [Adaptive Guidance and Integrated Navigation with Reinforcement Meta-Learning](https://arxiv.org/pdf/2109.03880).

## Citation

```bibtex
@software{moore2026metarl,
  author = {Moore, Isabel},
  title = {Meta-RL 6-DOF UAV Guidance},
  year = {2026},
  url = {https://github.com/isabelmoore/Meta-RL-6DOF-Guidance}
}
```
