# Copyright (c) 2026 Isabel Moore. All rights reserved.
import numpy as np
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import base_env_conf

def debug_stability():
    with open("debug_stability.txt", "w") as f:
        env = BVRBase(base_env_conf)
        env.reset()
        f.write("Step | p (rad/s) | Mach | Reward\n")
        f.write("-" * 30 + "\n")
        
        for i in range(10):
            obs, reward, done, truncated, info = env.step(np.zeros(3))
            p = env.blue_agent.simObj.fdm['velocities/p-rad_sec']
            mach = env.blue_agent.simObj.get_mach()
            f.write(f"{i:4} | {p:9.4f} | {mach:6.4f} | {reward:8.4f}\n")

if __name__ == "__main__":
    debug_stability()
