# Meta-RL 6-DOF UAV Guidance

Deep reinforcement learning for 6-DOF UAV guidance-to-intercept against maneuvering targets. A single LSTM policy (RecurrentPPO) learns to control UAV fin commands across multiple engagement scenarios using meta-RL (RL²).

<p align="center">
  <img src="demo/vehicle_closeup.gif" width="420" />
  <img src="demo/engagement_view.gif" width="420" />
</p>

The left GIF shows the UAV from a chase camera with real-time telemetry, ADI, and a gimbal attitude sphere. The right shows the full engagement — UAV body with flight trails chasing a maneuvering F-16 target.

## How It Works

[JSBSim](https://jsbsim.sourceforge.net/) provides 6-DOF flight dynamics. The RL agent directly commands fin deflections — there is no guidance law or autopilot in the loop.

The policy is a RecurrentPPO with a 256-unit LSTM. Instead of training one policy per scenario, we use meta-RL (RL²) — a single policy trains across all scenarios simultaneously. Each episode the env randomly picks a scenario, so the LSTM hidden state learns to adapt on-the-fly to different engagement geometries, target behaviors, and speed regimes. The hidden state resets every episode, so the policy has to re-identify the scenario from observations alone.

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

Terminal rewards:
- **Hit** (r < r_hit): +500
- **Fly-by** (closing velocity < -50 m/s within 500m): +500 · exp(-miss / 300)
- **Timeout**: +500 · exp(-min_range / 300)
- **Constraint violation** (speed, pitch, roll, yaw, load factor, altitude): -25

### Curriculum

Adaptive hit radius written to a shared file by the training callback. Starts at 500m, shrinks to 50m as the agent improves. Crossing the curriculum radius mid-episode gives a one-time milestone bonus (+150) without terminating.

## Scenarios

All scenarios use an AIM-7 UAV intercepting a maneuvering F-16 target:

| ID | Label | Range | Angles | Description |
|----|-------|-------|--------|-------------|
| A  | paper_ICs | 3–5 km | ±15° | Baseline engagement |
| B  | extended_range | 8–12 km | ±30° | Longer-range intercepts |
| C  | wide_angle | 5–10 km | ±45° | Off-boresight engagements |

Here's a Scenario A engagement from above — the UAV (red) launches from the right and intercepts the F-16 target (blue) maneuvering on the left:

<p align="center"><img src="demo/trajectory_overview.gif" width="600" /></p>

## Quick Start

### 1. Start the container

The container runs `sleep infinity` so it stays up, letting you exec into the container and run the long training jobs even if you disconnect from your virtual machine.

```bash
docker compose up -d
docker exec -it meta-rl bash
```

### 2. Train

Edit `train.sh` to set your desired scenarios, GPU, timesteps, and number of parallel envs:

```bash
# train.sh — edit these:
SCENARIOS="A B C"    # which scenarios to train on (or "all")
N_ENVS=8             # parallel environments
TIMESTEPS=30000000   # total training timesteps
GPU=1                # which GPU (CUDA_VISIBLE_DEVICES)
```

Then run it:

```bash
./train.sh
```

Or call `train_meta.py` directly with any of these flags:

```bash
python3 train_meta.py --scenarios A B C --timesteps 20000000 --n-envs 8 --lr 1e-4 --lstm-hidden 256
```

The default hyperparameters are hardcoded at the top of `train_meta.py`:

| Parameter | Default | Flag |
|-----------|---------|------|
| Total timesteps | 20M | `--timesteps` |
| Parallel envs | 8 | `--n-envs` |
| Learning rate | 1e-4 | `--lr` |
| LSTM hidden size | 256 | `--lstm-hidden` |
| Batch size | 512 | — |
| N steps (rollout) | 2048 | `--n-steps` |
| N epochs | 5 | — |
| Gamma | 0.99 | — |
| GAE lambda | 0.92 | — |
| Entropy coef | 0.005 | `--ent-coef` |
| Target KL | 0.04 | `--target-kl` |
| Save frequency | 250k | `--save-freq` |

To change batch size, n_epochs, gamma, or GAE lambda, edit the constants in `train_meta.py` directly.

### 3. Evaluate

Edit `evaluate.sh` to point to your training run directory:

```bash
# evaluate.sh — edit these:
RUN_DIR="training_logs/Mar03_0115_META_A_30M"   # path to your trained model
SCENARIOS="A"                                     # which scenarios to evaluate
EPISODES=50                                       # episodes per scenario
GIFS=10                                           # demo GIFs to generate
GPU=0                                             # which GPU
```

Then run it:

```bash
./evaluate.sh
```

Or call `evaluate_meta.py` directly:

```bash
python3 evaluate_meta.py --run training_logs/<run_dir> --scenarios all --episodes 50 --gifs 5
```

This prints a results table with hit rate, miss distance, and reward stats per scenario, and generates three GIF types per episode (trajectory overview, vehicle close-up, engagement view).

### 4. Scenario Configuration

Each scenario is a YAML file in `scenarios/`. To create a new scenario or tweak an existing one, edit the YAML directly. Key fields you'd want to change:

```yaml
# scenarios/A.yaml
initial_conditions:
  range_min: 3000.0        # starting range to target (meters)
  range_max: 5000.0
  UAV_speed_min: 800.0     # UAV launch speed (m/s)
  UAV_speed_max: 1000.0
  target_speed_min: 200.0  # target speed (m/s)
  target_speed_max: 300.0
  elevation_min: -15.0     # look angle limits (degrees)
  elevation_max: 15.0
  azimuth_min: -15.0
  azimuth_max: 15.0

reward_params:
  hit_bonus: 500.0          # terminal reward for hitting
  hit_radius: 50.0          # final hit radius (meters)
  hit_radius_start: 500.0   # curriculum start radius
  curriculum_steps: 4000000 # timesteps to shrink from start to end

target_maneuver:
  maneuver_interval_min: 3.0   # seconds between maneuvers
  maneuver_interval_max: 8.0
  heading_change_max: 30.0     # max heading change per maneuver (degrees)
  alt_change_max: 1500.0       # max altitude change per maneuver (meters)
```

The scenario YAML gets loaded by `scenario_loader.py` into a config namespace, which is passed to the Gym environment.

### 5. Vehicle Configuration

Vehicle physics are fully defined in YAML — the JSBSim XML is auto-generated at runtime from these values. Two UAV configs are included:

- `simulation/config/aim7.yaml` — AIM-7 Sparrow (231 kg, 3.66 m, Mach 1+)
- `simulation/config/gaudet.yaml` — Gaudet & Furfaro (2023) paper vehicle (455 kg, 6.25 m)

To switch vehicles, change `UAV_config_file` in the scenario YAML. The vehicle YAML has four key sections:

```yaml
# simulation/config/aim7.yaml
mass:
  weight_lbs: 510.0       # total weight
  ixx: 1.77               # roll inertia (slug·ft²)
  iyy: 190.7              # pitch inertia
  izz: 190.7              # yaw inertia

geometry:
  length_in: 144.0        # body length (inches)
  diameter_ft: 0.667      # body diameter (feet)
  wingspan_ft: 3.33       # fin span
  wingarea_ft2: 0.349     # reference area (body cross-section)
  cg_x_in: 72.0           # CG location from nose

aero:                      # nondimensional aerodynamic coefficients
  cn_alpha_body_wing: 6.0  # normal force slope (body + wing)
  cn_crossflow: 34.4       # crossflow drag factor
  cm_alpha: -20.4          # pitch static stability
  cm_elevator: 38.2        # elevator control power
  # ... 19 coefficients total (see YAML for full list)

fcs:                       # flight control system limits
  elevator_rate_max_deg_s: 40.0   # max fin rate (deg/s)
  elevator_pos_max_deg: 10.0      # max fin deflection (deg)
  rudder_rate_max_deg_s: 40.0
  rudder_pos_max_deg: 10.0
```

**Note:** The observation normalization constants (closing speed ÷ 2000, body rates ÷ 10, accel ÷ 450, speed ÷ 1000) in `uav_guidance_env.py` are tuned for Mach 1+ interceptors. If you create a vehicle with a very different speed/maneuverability regime, you may need to adjust these.

### 6. TensorBoard

```bash
tensorboard --logdir training_logs/
```

## Docker

The container runs with GPU access and sleeps on startup — you exec in and run commands.

```bash
docker compose up -d          # start
docker exec -it meta-rl bash  # exec in
docker compose down            # stop
```

Repo is mounted at `/wizard` inside the container.

## Project Structure

```
├── train_meta.py            # Training entry point
├── evaluate_meta.py         # Evaluation + plots + GIF generation
├── train.sh / evaluate.sh   # Shell wrappers (nohup + logging)
├── scenarios/               # Scenario YAML configs (A, B, C)
├── simulation/
│   ├── config/              # Vehicle YAML configs (aim7, f16)
│   ├── core/                # Config/scenario loaders, navigation utils
│   ├── environments/        # Gym environments
│   └── models/              # FDM wrappers around JSBSim
├── data_classes/            # Dataclasses for configs
├── jsbsim_data/             # JSBSim aircraft XML, engines, scripts
└── tests/
```

## Workflow Tips

**Find your latest run:**

```bash
ls -t training_logs/ | head -5
```

**Monitor a running job:**

```bash
tail -f training_logs/<run_dir>/train.log
tensorboard --logdir training_logs/ --port 6006
```

## Requirements

- Python 3.11+
- PyTorch, Stable-Baselines3, sb3-contrib (RecurrentPPO)
- JSBSim
- CUDA GPU recommended for training
