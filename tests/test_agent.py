# Copyright (c) 2026 Isabel Moore. All rights reserved.

from jsb_gym.agents.config import blue_agent
from jsb_gym.agents.agents import RLBVRAgent



# reset initial position of agent
blue_agent.lat = 59.0
blue_agent.long = 18.0
blue_agent.alt = 3000.0
blue_agent.vel = 250.0
blue_agent.heading = 0.0

agent = RLBVRAgent(blue_agent, None)
print(f"Agent initialized with ammo count: {len(agent.ammo)}")
exit()
