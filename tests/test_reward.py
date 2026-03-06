# Copyright (c) 2026 Isabel Moore. All rights reserved.
import gymnasium as gym
import numpy as np
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import base_env_conf

def test_reward():
    print("Initializing environment...")
    env = BVRBase(base_env_conf)
    obs, info = env.reset()
    
    print("Running 10 steps...")
    for i in range(10):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: Reward = {reward:.4f}, Done = {done}")
        if done:
            break
            
    print("Test complete.")

if __name__ == "__main__":
    test_reward()
