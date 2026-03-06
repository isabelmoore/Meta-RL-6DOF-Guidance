# Copyright (c) 2026 Isabel Moore. All rights reserved.

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.core.config_loader import ConfigLoader
from simulation.models.aircraft import Aircraft
from simulation.models.uavs import UAV
import matplotlib.pyplot as plt

# Load initial configs
AAM_config = ConfigLoader.load_config('simulation/config/vehicles/aim7.yaml')
f16_config = ConfigLoader.load_config('simulation/config/vehicles/f16.yaml')

# Setup Headless/No-FlightGear config
AAM_config.data_output_xml = None
f16_config.data_output_xml = None
AAM_config.fg_sleep_time = None
f16_config.fg_sleep_time = None

# Tighten tolerances for baseline
AAM_config.UAV_performance.effective_radius = 5.0 # meters
AAM_config.UAV_simulation.Sim_time_step = 1
f16_config.aircraft_simulation.Sim_time_step = 1

# Initialize objects
AIM = UAV(AAM_config)
F16 = Aircraft(f16_config)

# Reset positions
# Target F16 starts at 3000m, moving South
F16.reset(lat=59, long=18, alt=3000, vel=250, heading=180)

# UAV starts at 5000m, heading North towards target
# approx 0.6 degrees latitude difference is ~66km
AIM.reset(lat=58.5, long=18, alt=5000, vel=250, heading=0)
AIM.set_target(F16)

min_distance = float('inf')
time_steps = 0
intercepted = False

print(f"{'Time':>6} | {'Dist (m)':>10} | {'Msl Alt':>8} | {'Tgt Alt':>8} | {'Msl Mach':>8}")
print("-" * 55)

distances = []

for i in range(50000): # max steps
    # Weaving maneuver for target
    # Every 100 seconds (approx 100 steps if Sim_time_step=1 and frequency is high)
    # Actually, Sim_time_step=1 means 1 JSBSim execution per loop.
    # Default frequency is usually 120Hz in JSBSim scripts.
    
    t = AIM.get_sim_time_sec()
    
    # Target maneuver: Weave heading every 10 seconds
    target_heading = 180 + 30 * np.sin(0.1 * t)
    target_altitude = 3000 + 500 * np.cos(0.05 * t)
    
    action = [target_heading, target_altitude, 0.5]
    F16.step(action)
    AIM.step()
    
    dist = AIM.PN.distance_to_target
    distances.append(dist)
    if dist < min_distance:
        min_distance = dist
    
    if i % 100 == 0:
        print(f"{t:6.1f} | {dist:10.1f} | {AIM.get_altitude():8.1f} | {F16.get_altitude():8.1f} | {AIM.get_mach():8.3f}")

    if AIM.is_target_hit():
        print(f"\nINTERCEPT ACHIEVED!")
        intercepted = True
        break
    
    if not AIM.is_alive():
        print(f"\nUAV Dead (Fuel/Alt/Velocity).")
        break

    time_steps = i

print("-" * 55)
print(f"Final Results:")
print(f"Intercept: {intercepted}")
print(f"Min Distance: {min_distance:.2f} meters")
print(f"Time to Intercept: {AIM.get_sim_time_sec():.2f} seconds")

# Optional: Plotting could be done if running in a window, but we are headless.
# We'll just rely on the print output for now.
