# Copyright (c) 2026 Isabel Moore. All rights reserved.
import numpy as np
import pymap3d as pm
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import base_env_conf

def debug_reward():
    env = BVRBase(base_env_conf)
    env.reset()
    
    print("Step | Shaping | Ctrl | Roll | Total | Mach | Roll Rate (rad/s)")
    print("-" * 65)
    
    for i in range(5):
        # Sample action (small or zero to see baseline)
        action = np.zeros(3) # No command
        obs, reward, done, truncated, info = env.step(action)
        
        # Manually calculate components to see what they are
        p = env.blue_agent.simObj.fdm['velocities/p-rad_sec']
        ve_m = env.blue_agent.simObj.get_v_east()
        vn_m = env.blue_agent.simObj.get_v_north()
        vu_m = -env.blue_agent.simObj.get_v_down()
        v_m = np.array([ve_m, vn_m, vu_m])
        
        # Shaping part (LOS rate)
        lat0, lon0, h0 = env.blue_agent.simObj.get_lat_gc_deg(), env.blue_agent.simObj.get_long_gc_deg(), env.blue_agent.simObj.get_altitude()
        lat1, lon1, h1 = env.red_agent.simObj.get_lat_gc_deg(), env.red_agent.simObj.get_long_gc_deg(), env.red_agent.simObj.get_altitude()
        e, n, u = pm.geodetic2enu(lat1, lon1, h1, lat0, lon0, h0, ell=None, deg=True)
        r_tm = np.array([e, n, u])
        r_tm_norm_sq = np.dot(r_tm, r_tm)
        ve_t, vn_t, vu_t = env.red_agent.simObj.get_v_east(), env.red_agent.simObj.get_v_north(), -env.red_agent.simObj.get_v_down()
        v_tm = np.array([ve_t - ve_m, vn_t - vn_m, vu_t - vu_m])
        omega = np.cross(r_tm, v_tm) / r_tm_norm_sq if r_tm_norm_sq > 1e-6 else np.zeros(3)
        omega_norm = np.linalg.norm(omega)
        r_shaping = 1.0 * np.exp(-(omega_norm**2) / (0.02**2))
        
        r_roll = -0.05 * np.abs(p)
        r_ctrl = -0.01 * np.sum(np.abs(action))
        
        print(f"{i:4} | {r_shaping:7.4f} | {r_ctrl:4.2f} | {r_roll:5.1f} | {reward:6.1f} | {env.blue_agent.simObj.get_mach():4.2f} | {p:7.2f}")

if __name__ == "__main__":
    debug_reward()
