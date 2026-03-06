# Copyright (c) 2026 Isabel Moore. All rights reserved.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.core.config_loader import ConfigLoader
from simulation.models.aircraft import Aircraft
from simulation.models.uavs import UAV

AAM_config = ConfigLoader.load_config('simulation/config/aim7.yaml')
f16_config = ConfigLoader.load_config('simulation/config/f16.yaml')

AAM_config.data_output_xml = None
f16_config.data_output_xml = None
AAM_config.fg_sleep_time = None
f16_config.fg_sleep_time = None
AAM_config.UAV_performance.effective_radius = 1
AAM_config.UAV_simulation.Sim_time_step = 1
f16_config.aircraft_simulation.Sim_time_step = 1


#self.blue_agent = RLBVRAgent(blue_agent, self)
AIM = UAV(AAM_config)
F16 = Aircraft(f16_config)


F16.reset(lat=59, long=18, alt=3000, vel=250, heading=180)

AIM.reset(lat=58.4, long=18, alt=10000, vel=250, heading=0)
AIM.set_target(F16)


for _ in range(1000):
    action = [150, 4000, 0.5]  # heading, altitude, throttle
    F16.step(action)
    AIM.step()

    if AIM.PN.distance_to_target < 600:
        print("Close to target!, distance:", AIM.PN.distance_to_target)
        # input("Press Enter to continue a step...")
    AIM.print_status()

# run from BVRGYM directory using: 
#  python -m tests.test_UAV