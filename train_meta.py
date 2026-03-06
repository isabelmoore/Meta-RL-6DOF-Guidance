# Copyright (c) 2026 Isabel Moore. All rights reserved.
"""
Usage:
    python train_meta.py --scenarios A B C
    python train_meta.py --scenarios all
    python train_meta.py --scenarios A B --holdout C --timesteps 500000 --n-envs 2
"""
import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict
from datetime import datetime

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO

from simulation.environments.meta_uav_guidance_env import MetaUAVGuidanceEnv
from simulation.core.scenario_loader import find_scenario, list_scenarios

TOTAL_TIMESTEPS = 20_000_000
N_ENVS = 8
LR = 1e-4
GAMMA = 0.99
N_STEPS = 2048
BATCH_SIZE = 512
N_EPOCHS = 5
GAE_LAMBDA = 0.92
ENT_COEF = 0.005
TARGET_KL = 0.04
VF_COEF = 1.0
MAX_GRAD_NORM = 0.5
LSTM_HIDDEN = 256
SAVE_FREQ = 250_000


class MetaCallback(BaseCallback):
    def __init__(self, model_dir, scenario_labels, total_timesteps,
                 save_freq=SAVE_FREQ, curriculum_file=None,
                 radius_start=3000.0, radius_end=10.0):
        """Training callback for meta-RL with per-scenario logging and adaptive curriculum.

        Args:
            model_dir: Directory to save model checkpoints.
            scenario_labels: List of human-readable scenario labels for logging.
            total_timesteps: Total training timesteps (used for scheduling).
            save_freq: Checkpoint save frequency in timesteps.
            curriculum_file: Path to file for sharing adaptive hit radius with envs.
            radius_start: Initial curriculum hit radius in meters.
            radius_end: Minimum curriculum hit radius in meters.
        """
        super().__init__()
        self.model_dir = model_dir
        self.scenario_labels = scenario_labels
        self.total_timesteps = total_timesteps
        self.save_freq = save_freq
        self.last_save_step = 0
        self.best_mean_reward = -float("inf")
        self.w = 100

        self.recent_rewards = []
        self.recent_hits = []
        self.recent_curriculum = []
        self.recent_miss = []
        self.recent_min_range = []
        self.recent_lens = []
        self.recent_reasons = []

        self.scenario_hits = defaultdict(list)
        self.scenario_curriculum = defaultdict(list)
        self.scenario_rewards = defaultdict(list)
        self.scenario_miss = defaultdict(list)
        self.scenario_reasons = defaultdict(list)

        self.curriculum_file = curriculum_file
        self.adaptive_radius = radius_start
        self.radius_start = radius_start
        self.radius_end = radius_end
        self._last_adapt_ep = 0
        if curriculum_file:
            self._write_radius(radius_start)

    def _write_radius(self, radius):
        """Write the current curriculum radius to the shared file.

        Args:
            radius: Hit radius in meters to write.
        """
        with open(self.curriculum_file, 'w') as f:
            f.write(f"{radius:.2f}")
        self.adaptive_radius = radius

    def _on_step(self):
        """Collect episode stats, adapt curriculum radius, and trigger checkpoints."""
        for info in self.locals.get("infos", []):
            reason = info.get("termination_reason", "")
            label = info.get("scenario_label", "")

            if reason:
                is_hit = reason == "hit"
                crossed = info.get("crossed_curriculum", False)
                self.recent_hits.append(is_hit)
                self.recent_curriculum.append(crossed)
                self.recent_reasons.append(reason)
                if label:
                    self.scenario_hits[label].append(is_hit)
                    self.scenario_curriculum[label].append(crossed)
                    self.scenario_reasons[label].append(reason)

                if "range" in info:
                    self.recent_miss.append(info["range"])
                    if label:
                        self.scenario_miss[label].append(info["range"])

                if "min_range" in info:
                    self.recent_min_range.append(info["min_range"])

            if "episode" in info:
                self.recent_rewards.append(info["episode"]["r"])
                self.recent_lens.append(info["episode"]["l"])
                if label:
                    self.scenario_rewards[label].append(info["episode"]["r"])

        n_eps = len(self.recent_min_range)
        if self.curriculum_file and n_eps >= self.w and n_eps - self._last_adapt_ep >= self.w:
            self._last_adapt_ep = n_eps
            mean_closest = np.mean(self.recent_min_range[-self.w:])
            old_r = self.adaptive_radius
            if mean_closest < old_r * 0.5:
                new_r = max(self.radius_end, old_r * 0.90)
            elif mean_closest > old_r * 2.0:
                new_r = min(self.radius_start, old_r * 1.05)
            else:
                new_r = old_r
            if new_r != old_r:
                self._write_radius(new_r)

        if self.num_timesteps - self.last_save_step >= self.save_freq:
            self._save_checkpoint()
            self.last_save_step = self.num_timesteps

        return True

    def _on_rollout_end(self):
        """Log aggregate and per-scenario metrics at the end of each rollout."""
        self._log_aggregate()
        self._log_per_scenario()

    def _log_aggregate(self):
        """Log windowed aggregate metrics (reward, hit rate, miss distance) to TensorBoard."""
        if not self.recent_rewards:
            return
        n = len(self.recent_rewards)
        window = [r for r in self.recent_rewards[-self.w:] if np.isfinite(r)]
        mean_rew = np.mean(window) if window else 0.0
        self.logger.record("meta/mean_reward", mean_rew)
        self.logger.record("meta/n_episodes", n)

        if self.recent_hits:
            window = self.recent_hits[-self.w:]
            self.logger.record("meta/hit_rate", sum(window) / len(window))
        if self.recent_curriculum:
            window = self.recent_curriculum[-self.w:]
            self.logger.record("meta/curriculum_cross_rate", sum(window) / len(window))
        if self.recent_miss:
            rm = self.recent_miss[-self.w:]
            self.logger.record("meta/mean_miss_dist", np.mean(rm))
            self.logger.record("meta/min_miss_dist", np.min(rm))
        if self.recent_min_range:
            self.logger.record("meta/mean_closest_approach",
                               np.mean(self.recent_min_range[-self.w:]))
        if self.recent_lens:
            self.logger.record("meta/mean_ep_length",
                               np.mean(self.recent_lens[-self.w:]))
        if self.recent_reasons:
            rr = self.recent_reasons[-self.w:]
            for r in set(rr):
                self.logger.record(f"meta/term_{r}_pct", rr.count(r) / len(rr))

        if self.curriculum_file:
            self.logger.record("meta/curriculum_radius", self.adaptive_radius)

    def _log_per_scenario(self):
        """Log per-scenario hit rate, reward, and miss distance to TensorBoard."""
        for label in self.scenario_labels:
            prefix = f"scenario/{label}"
            hits = self.scenario_hits.get(label, [])
            if hits:
                self.logger.record(f"{prefix}/hit_rate",
                                   sum(hits[-50:]) / len(hits[-50:]))
            crossed = self.scenario_curriculum.get(label, [])
            if crossed:
                self.logger.record(f"{prefix}/curriculum_cross_rate",
                                   sum(crossed[-50:]) / len(crossed[-50:]))
            rewards = self.scenario_rewards.get(label, [])
            if rewards:
                self.logger.record(f"{prefix}/mean_reward",
                                   np.mean(rewards[-50:]))
            miss = self.scenario_miss.get(label, [])
            if miss:
                self.logger.record(f"{prefix}/mean_miss_dist",
                                   np.mean(miss[-50:]))

    def _save_checkpoint(self):
        """Save model checkpoint and print training progress summary."""
        is_best = False
        window_rew = [r for r in self.recent_rewards[-self.w:] if np.isfinite(r)]
        if window_rew:
            mr = np.mean(window_rew)
            if mr > self.best_mean_reward:
                self.best_mean_reward = mr
                is_best = True

        tag = f"{self.num_timesteps}_best" if is_best else str(self.num_timesteps)
        self.model.save(f"{self.model_dir}/{tag}")

        hit_str = ""
        if len(self.recent_hits) >= self.w:
            hit_str = f"hit={sum(self.recent_hits[-self.w:])/self.w:.0%}"
        cross_str = ""
        if len(self.recent_curriculum) >= self.w:
            cross_str = f"cross={sum(self.recent_curriculum[-self.w:])/self.w:.0%}"
        miss_str = ""
        if len(self.recent_miss) >= self.w:
            miss_str = f"miss={np.mean(self.recent_miss[-self.w:]):.0f}"
        hr_str = f"r={self.adaptive_radius:.0f}" if self.curriculum_file else ""

        scenario_parts = []
        for label in self.scenario_labels:
            hits = self.scenario_hits.get(label, [])
            if len(hits) >= 10:
                recent = hits[-50:]
                sr = sum(recent) / len(recent)
                short = label.split("_")[0]
                scenario_parts.append(f"{short}={sr:.0%}")
        scenario_str = " ".join(scenario_parts)

        print(
            f"[META] {self.num_timesteps:>10d}  "
            f"rew={np.mean(window_rew) if window_rew else 0:.1f}  "
            f"{miss_str}  {hit_str}  {cross_str}  {hr_str}  {scenario_str}  "
            f"{'BEST' if is_best else ''}",
            flush=True,
        )


def main():
    """Parse CLI args, build meta-RL env and RecurrentPPO model, and run training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--holdout", nargs="*", default=[])
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--lstm-hidden", type=int, default=LSTM_HIDDEN)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--target-kl", type=float, default=TARGET_KL)
    parser.add_argument("--ent-coef", type=float, default=ENT_COEF)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-freq", type=int, default=SAVE_FREQ)
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    if args.scenarios == ["all"]:
        scenario_names = list_scenarios()
    else:
        scenario_names = args.scenarios

    holdout_set = set(args.holdout)
    train_scenarios = [s for s in scenario_names if s not in holdout_set]
    if not train_scenarios:
        print("Error: no training scenarios left after holdout exclusion")
        sys.exit(1)

    scenario_paths = [find_scenario(s) for s in train_scenarios]
    scenario_labels = []
    scenario_configs = []
    for path in scenario_paths:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        scenario_labels.append(data.get("label", data.get("name", "")))
        scenario_configs.append(data)

    # extract shared config info for title (from first scenario)
    first = scenario_configs[0]
    guidance_tag = first.get("guidance_type", "PN")
    autopilot_tag = first.get("autopilot_type", "UAVPIDAutopilot").replace("PIDAutopilot", "")
    reward_tag = os.path.splitext(os.path.basename(first.get("reward_config", first.get("reward_type", "gaudet"))))[0]
    uav_tag = os.path.splitext(os.path.basename(first.get("UAV_config_file", "uav")))[0]
    target_tag = os.path.splitext(os.path.basename(first.get("target_config_file", "target")))[0]

    if args.run_dir:
        run_dir = args.run_dir
    else:
        timestamp = datetime.now().strftime("%b%d_%H%M")
        scen_tag = "_".join(train_scenarios)
        run_name = f"{timestamp}_{scen_tag}_int-{uav_tag}_tar-{target_tag}_{guidance_tag}_rew-{reward_tag}_{args.timesteps // 1_000_000}m"
        run_dir = f"training_logs/{run_name}"
    log_dir = f"{run_dir}/logs"
    model_dir = f"{run_dir}/models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    curriculum_file = f"{run_dir}/curriculum_radius.txt"

    meta_config = {
        "train_scenarios": train_scenarios,
        "holdout_scenarios": list(holdout_set),
        "scenario_paths": scenario_paths,
        "scenario_labels": scenario_labels,
        "uav_config": uav_tag,
        "target_config": target_tag,
        "guidance_type": guidance_tag,
        "autopilot_type": first.get("autopilot_type", "UAVPIDAutopilot"),
        "reward_config": first.get("reward_config", "inline"),
        "reward_type": first.get("reward_type", "gaudet"),
        "total_timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "lstm_hidden": args.lstm_hidden,
        "lr": args.lr,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "n_steps": args.n_steps,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "ent_coef": args.ent_coef,
        "vf_coef": VF_COEF,
        "target_kl": args.target_kl,
        "norm_reward": True,
        "device": args.device,
        "timestamp": timestamp,
    }
    with open(f"{run_dir}/meta_config.json", "w") as f:
        json.dump(meta_config, f, indent=2)

    print(f"{'='*60}")
    print(f"Meta-RL Training")
    print(f"{'='*60}")
    print(f"Run dir:    {run_dir}")
    print(f"Scenarios:  {train_scenarios}")
    print(f"Holdout:    {list(holdout_set) if holdout_set else 'none'}")
    print(f"Labels:     {scenario_labels}")
    print(f"Guidance:   {guidance_tag}")
    print(f"Autopilot:  {autopilot_tag}")
    print(f"Reward:     {reward_tag}")
    print(f"Timesteps:  {args.timesteps:,}")
    print(f"N_envs:     {args.n_envs}")
    print(f"LSTM:       {args.lstm_hidden}")
    print(f"LR:         {args.lr}")
    print(f"N_steps:    {args.n_steps}")
    print(f"Device:     {args.device}")
    print(f"{'='*60}")

    def make_env():
        def _init():
            env = MetaUAVGuidanceEnv(
                scenario_paths=scenario_paths,
                curriculum_file=curriculum_file,
                n_envs=args.n_envs,
                total_timesteps=args.timesteps,
            )
            return Monitor(env)
        return _init

    vec_env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])

    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=GAMMA,
    )

    policy_kwargs = dict(
        lstm_hidden_size=args.lstm_hidden,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    def linear_schedule(initial_lr):
        def func(progress_remaining):
            return progress_remaining * initial_lr
        return func

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=linear_schedule(args.lr),
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        n_steps=args.n_steps,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        ent_coef=args.ent_coef,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=args.target_kl,
        policy_kwargs=policy_kwargs,
        device=args.device,
    )

    callback = MetaCallback(
        model_dir=model_dir,
        scenario_labels=scenario_labels,
        total_timesteps=args.timesteps,
        save_freq=args.save_freq,
        curriculum_file=curriculum_file,
        radius_start=500.0,
        radius_end=10.0,
    )

    print("Starting training...")
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(f"{model_dir}/final")
    vec_env.save(f"{model_dir}/vecnormalize.pkl")
    print(f"Done. Final model: {model_dir}/final.zip")


if __name__ == "__main__":
    main()
