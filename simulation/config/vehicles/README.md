# Vehicle Configs

Each YAML defines a vehicle's mass, geometry, aerodynamics, and
flight control system. JSBSim XML is auto-generated at runtime
from these values.

Scenarios point to them with UAV_config_file and target_config_file.


## Files

aim7.yaml
  AIM-7 Sparrow. 231 kg, 3.66m long, Mach 1+. Triangular
  tail fins and mid-body wings. The main interceptor.

f16.yaml
  F-16 Fighting Falcon. Used as the target aircraft in most
  scenarios. Full-authority autopilot, subsonic.

gaudet.yaml
  Vehicle from Gaudet & Furfaro (2023). 455 kg, 6.25m long.
  Same fin style as AIM-7. Used to reproduce the paper results.

rs28_sarmat.yaml
  RS-28 Sarmat ICBM. 35.5m long, 3m diameter, 208 tonnes.
  Small tail fins, low aero authority, slow FCS. Used as the
  ballistic target in scenario E.

## YAML sections

Each vehicle YAML has four main sections:

mass
  weight, moments of inertia, CG location

geometry
  length, diameter, wingspan, fin configuration (tail_fins,
  tail_tip_ratio, tail_sweep, wings)

aero
  aerodynamic coefficients (CD0, CLa, CMa, CMq, etc.)

fcs
  flight control system gains and rate limits


## Making a new vehicle

Copy an existing file, change the numbers, and point your
scenario at it:

  UAV_config_file: "simulation/config/vehicles/my_vehicle.yaml"
