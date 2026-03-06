# Copyright (c) 2026 Isabel Moore. All rights reserved.
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import base_env_conf

env = BVRBase(base_env_conf)
obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Sample a random action
    obs, reward, done, trunk, info = env.step(action)
    env.log_tacview()
print(info)
print("View data in Tacview from data_output/tacview/")

