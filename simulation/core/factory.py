
from simulation.core.autopilot import UAVPIDAutopilot, AircraftPIDAutopilot
from simulation.core.guidance_laws import PN

class ComponentFactory:
    _autopilots = {
        "UAVPIDAutopilot": UAVPIDAutopilot,
        "AircraftPIDAutopilot": AircraftPIDAutopilot,
    }
    
    _guidance_laws = {
        "PN": PN,
    }
    
    @classmethod
    def register_autopilot(cls, name, class_ref):
        cls._autopilots[name] = class_ref
        
    @classmethod
    def register_guidance(cls, name, class_ref):
        cls._guidance_laws[name] = class_ref
        
    @classmethod
    def get_autopilot(cls, name):
        return cls._autopilots.get(name, UAVPIDAutopilot)
        
    @classmethod
    def get_guidance(cls, name):
        return cls._guidance_laws.get(name, PN)
