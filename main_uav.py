# Copyright (c) 2026 Isabel Moore. All rights reserved.
from jsb_gym.envs.UAVGuidanceEnv import UAVGuidanceEnv
from jsb_gym.envs.config import uav_guidance_conf

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import RecurrentPPO
from datetime import datetime
import os
import sys
import numpy as np


ALGORITHM = "RecurrentPPO"
POLICY = "MlpLstmPolicy"
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
LR = 3e-4
GAMMA = 0.995
N_STEPS = 256
ENT_COEF = 0.01
TARGET_KL = 0.01


def make_run_dir():
    timestamp = datetime.now().strftime("%b%d_%H%M")
    name = f"{timestamp}_{TOTAL_TIMESTEPS // 1000}k_lr{LR}_g{GAMMA}_ns{N_STEPS}_ent{ENT_COEF}_kl{TARGET_KL}_{N_ENVS}env"
    run_dir = f"runs/UAVGuidance_{ALGORITHM}/{name}"
    log_dir = f"{run_dir}/logs"
    model_dir = f"{run_dir}/models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return run_dir, log_dir, model_dir


class UAVGuidanceCallback(BaseCallback):
    def __init__(self, model_dir, save_freq=50_000):
        super().__init__()
        self.model_dir = model_dir
        self.best_mean_reward = -float("inf")
        self.recent_rewards = []
        self.recent_hits = []
        self.recent_miss_distances = []
        self.recent_ep_lengths = []
        self.recent_reasons = []
        self.save_freq = save_freq
        self.last_save_step = 0
        self.window = 100  # rolling window size

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "termination_reason" in info and info["termination_reason"] != "":
                reason = info["termination_reason"]
                is_hit = reason == "hit"
                self.recent_hits.append(is_hit)
                self.recent_reasons.append(reason)

                if "range" in info:
                    self.recent_miss_distances.append(info["range"])

            if "episode" in info:
                self.recent_rewards.append(info["episode"]["r"])
                self.recent_ep_lengths.append(info["episode"]["l"])

        # Log rolling stats every `window` episodes
        if len(self.recent_rewards) > 0 and len(self.recent_rewards) % self.window == 0:
            recent_rew = self.recent_rewards[-self.window:]
            mean_rew = sum(recent_rew) / len(recent_rew)
            self.logger.record("UAV/mean_reward_100", mean_rew)

            if len(self.recent_hits) >= self.window:
                hit_rate = sum(self.recent_hits[-self.window:]) / self.window
                self.logger.record("UAV/hit_rate_100", hit_rate)

            if len(self.recent_miss_distances) >= self.window:
                recent_miss = self.recent_miss_distances[-self.window:]
                self.logger.record("UAV/mean_miss_dist_100", np.mean(recent_miss))
                self.logger.record("UAV/min_miss_dist_100", np.min(recent_miss))
                self.logger.record("UAV/median_miss_dist_100", np.median(recent_miss))

            if len(self.recent_ep_lengths) >= self.window:
                recent_len = self.recent_ep_lengths[-self.window:]
                self.logger.record("UAV/mean_ep_length_100", np.mean(recent_len))

            # Termination reason breakdown
            if len(self.recent_reasons) >= self.window:
                recent_reasons = self.recent_reasons[-self.window:]
                for r in set(recent_reasons):
                    self.logger.record(f"UAV/term_{r}_pct", recent_reasons.count(r) / len(recent_reasons))

        # Save checkpoint every save_freq steps
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            step = self.num_timesteps
            is_best = False

            if len(self.recent_rewards) >= self.window:
                mean_reward = sum(self.recent_rewards[-self.window:]) / self.window
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    is_best = True

            if is_best:
                self.model.save(f"{self.model_dir}/{step}_best")
                print(f"New best model! mean_reward={self.best_mean_reward:.2f} -> {step}_best.zip")
            else:
                self.model.save(f"{self.model_dir}/{step}")
                print(f"Checkpoint saved: {step}.zip")

            self.last_save_step = self.num_timesteps

        return True


def main():
    run_dir, log_dir, model_dir = make_run_dir()
    print(f"Run directory: {run_dir}")
    print(f"Algorithm: {ALGORITHM} | Policy: {POLICY}")
    print(f"Envs: {N_ENVS} | Steps: {N_STEPS} | LR: {LR} | Gamma: {GAMMA} | Ent: {ENT_COEF} | KL: {TARGET_KL}")
    # Log file (tee'd — stdout stays visible)
    log_path = f"{run_dir}/training.log"
    print(f"Logging to: {log_path}")

    policy_kwargs = dict(
        lstm_hidden_size=128,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    vec_env = make_vec_env(
        UAVGuidanceEnv,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"conf": uav_guidance_conf},
    )

    model = RecurrentPPO(
        POLICY,
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=LR,
        gamma=GAMMA,
        n_steps=N_STEPS,
        ent_coef=ENT_COEF,
        target_kl=TARGET_KL,
        policy_kwargs=policy_kwargs,
    )

    print(f"Starting Training: {run_dir}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=UAVGuidanceCallback(model_dir))

    model.save(f"{model_dir}/final")


if __name__ == "__main__":
    main()
