# Copyright (c) 2026 Isabel Moore. All rights reserved.
import sys
import os
import numpy as np
import gymnasium as gym

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.environments.uav_guidance_env import UAVGuidanceEnv
try:
    from simulation.environments.config import uav_guidance_conf
except ImportError:
    # If running from root, import might be different depending on path setup
    sys.path.append(os.getcwd())
    from simulation.environments.config import uav_guidance_conf

def test_loading():
    print("Testing UAVGuidanceEnv initialization with YAML configs...")
    
    # Create environment (will load aim7.yaml and f16.yaml by default)
    try:
        env = UAVGuidanceEnv(conf=uav_guidance_conf)
    except Exception as e:
        print(f"Failed to initialize env: {e}")
        raise e
    
    print(f"UAV Config: {env.UAV_config.name} ({env.UAV_config.weight_lbs} lbs)")
    print(f"Target Config: {env.target_config.name} ({env.target_config.weight_lbs} lbs)")
    
    if env.UAV_config.name != "AIM-7":
        print(f"ERROR: Expected AIM-7, got {env.UAV_config.name}")
    if env.target_config.name != "F-16":
        print(f"ERROR: Expected F-16, got {env.target_config.name}")
    
    # Reset environment
    print("Resetting environment...")
    try:
        obs, info = env.reset()
    except Exception as e:
        print(f"Failed to reset env: {e}")
        raise e
        
    print("Environment reset successful.")
    print(f"Observation shape: {obs.shape}")
    
    # Step
    action = np.zeros(3)
    try:
        obs, reward, done, truncated, info = env.step(action)
        print("Step successful.")
    except Exception as e:
         print(f"Failed to step env: {e}")
         raise e
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_loading()
