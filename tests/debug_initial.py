# Copyright (c) 2026 Isabel Moore. All rights reserved.
import numpy as np
import pymap3d as pm
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import base_env_conf

def debug_initial():
    with open("debug_out.txt", "w") as f:
        f.write("Initializing environment...\n")
        env = BVRBase(base_env_conf)
        env.reset()
        
        p = env.blue_agent.simObj.fdm['velocities/p-rad_sec']
        q = env.blue_agent.simObj.fdm['velocities/q-rad_sec']
        r = env.blue_agent.simObj.fdm['velocities/r-rad_sec']
        
        f.write(f"Initial Rates: p={p:.4f}, q={q:.4f}, r={r:.4f}\n")
        
        f.write("Running 1 step with zero action...\n")
        env.step(np.zeros(3))
        
        p = env.blue_agent.simObj.fdm['velocities/p-rad_sec']
        f.write(f"Rates after 1 step: p={p:.4f}\n")
        
        # Check Mach and velocity
        mach = env.blue_agent.simObj.get_mach()
        f.write(f"Mach: {mach:.4f}\n")

if __name__ == "__main__":
    debug_initial()
