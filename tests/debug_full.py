# Copyright (c) 2026 Isabel Moore. All rights reserved.
import numpy as np
import pymap3d as pm
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import base_env_conf

def debug_full():
    env = BVRBase(base_env_conf)
    env.reset()
    
    with open("debug_full.txt", "w") as f:
        f.write("Step | Shaping | Ctrl | Roll | OmegaNorm | p | Reward\n")
        f.write("-" * 65 + "\n")
        
        for i in range(5):
            action = np.zeros(3)
            obs, reward, done, truncated, info = env.step(action)
            
            # Extract internal state for manual verification
            p = env.blue_agent.simObj.fdm['velocities/p-rad_sec']
            lat0, lon0, h0 = env.blue_agent.simObj.get_lat_gc_deg(), env.blue_agent.simObj.get_long_gc_deg(), env.blue_agent.simObj.get_altitude()
            lat1, lon1, h1 = env.red_agent.simObj.get_lat_gc_deg(), env.red_agent.simObj.get_long_gc_deg(), env.red_agent.simObj.get_altitude()
            e, n, u = pm.geodetic2enu(lat1, lon1, h1, lat0, lon0, h0, ell=None, deg=True)
            r_tm = np.array([e, n, u])
            ve_m, vn_m, vu_m = env.blue_agent.simObj.get_v_east(), env.blue_agent.simObj.get_v_north(), -env.blue_agent.simObj.get_v_down()
            ve_t, vn_t, vu_t = env.red_agent.simObj.get_v_east(), env.red_agent.simObj.get_v_north(), -env.red_agent.simObj.get_v_down()
            v_tm = np.array([ve_t - ve_m, vn_t - vn_m, vu_t - vu_m])
            r_tm_norm_sq = np.dot(r_tm, r_tm)
            omega = np.cross(r_tm, v_tm) / r_tm_norm_sq
            omega_norm = np.linalg.norm(omega)
            
            # Reprod rewards in script
            s_r = 1.0 * np.exp(-(omega_norm**2) / (0.02**2))
            c_r = -0.01 * np.sum(np.abs(env.last_raw_action))
            r_r = -0.05 * np.abs(p)
            
            f.write(f"{i:4} | {s_r:7.4f} | {c_r:4.2f} | {r_r:4.2f} | {omega_norm:9.6f} | {p:7.2f} | {reward:8.4f}\n")

if __name__ == "__main__":
    debug_full()
