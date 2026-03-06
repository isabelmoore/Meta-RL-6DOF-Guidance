# Copyright (c) 2026 Isabel Moore. All rights reserved.
from dataclasses import dataclass

@dataclass
class Agent_parameters:
    ammo: int
    lat: float
    long: float 
    alt: float 
    vel: float 
    heading: float 


@dataclass
class BT_parameters:
    ammo: int
    lat: float 
    long: float 
    alt: float 
    vel: float 
    heading: float 
