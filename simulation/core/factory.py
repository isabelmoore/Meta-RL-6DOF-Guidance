# Copyright (c) 2026 Isabel Moore. All rights reserved.

from simulation.core.autopilot import UAVPIDAutopilot, AircraftPIDAutopilot
from simulation.core.guidance_laws import PN, APN, ZEM, PurePursuit

class ComponentFactory:
    _autopilots = {
        "UAVPIDAutopilot": UAVPIDAutopilot,
        "AircraftPIDAutopilot": AircraftPIDAutopilot,
    }

    _guidance_laws = {
        "PN": PN,
        "pro_nav": PN,
        "APN": APN,
        "augmented_pro_nav": APN,
        "ZEM": ZEM,
        "zero_effort_miss": ZEM,
        "pure_pursuit": PurePursuit,
    }
    
    @classmethod
    def register_autopilot(cls, name, class_ref):
        cls._autopilots[name] = class_ref
        
    @classmethod
    def register_guidance(cls, name, class_ref):
        cls._guidance_laws[name] = class_ref
        
    @classmethod
    def get_autopilot(cls, name):
        if name not in cls._autopilots:
            options = ', '.join(sorted(cls._autopilots.keys()))
            raise ValueError(f"Unknown autopilot_type: '{name}'. Options: {options}")
        return cls._autopilots[name]

    @classmethod
    def get_guidance(cls, name):
        if name not in cls._guidance_laws:
            options = ', '.join(sorted(cls._guidance_laws.keys()))
            raise ValueError(f"Unknown guidance_type: '{name}'. Options: {options}")
        return cls._guidance_laws[name]
