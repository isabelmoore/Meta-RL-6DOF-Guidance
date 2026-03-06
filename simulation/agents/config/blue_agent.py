from data_classes.agent_dataclass import Agent_parameters
from simulation.core.config_loader import ConfigLoader

agent_parameters = Agent_parameters(
    ammo=4,
    lat=58.0,
    long=18.0,
    alt=7000.0,
    vel=330.0,
    heading=0.0
)

aircraft_simObj_conf = ConfigLoader.load_config('simulation/config/f16.yaml')

UAV_simObj_conf = ConfigLoader.load_config('simulation/config/aim7.yaml')

aircraft_name = aircraft_simObj_conf.name

agent_name = "RL"

team = "Blue"