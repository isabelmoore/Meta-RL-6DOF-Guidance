import yaml
import sys
import os
from dataclasses import fields

# Add parent directory to path to allow imports if running as script
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from data_classes import uav_dataclass as MDC
from data_classes import aircraft_dataclass as ADC
from data_classes.uav_dataclass import UAVVisualization

class ConfigLoader:
    @staticmethod
    def load_config(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        config_type = data.get('type', 'UAV')
        
        if config_type == 'UAV':
            return UAVConfig(data)
        elif config_type == 'aircraft':
            return AircraftConfig(data)
        else:
            raise ValueError(f"Unknown config type: {config_type}")

class UAVConfig:
    def __init__(self, data):
        self.name = data.get('name')
        self.type = data.get('type')
        self.fdm_xml = data.get('fdm_xml')
        self.weight_lbs = data.get('weight_lbs')
        self.length_in = data.get('length_in')
        self.data_output_xml = data.get('data_output_xml', None)
        self.fg_sleep_time = data.get('fg_sleep_time', None)
        self.autopilot_type = data.get('autopilot_type', 'UAVPIDAutopilot')
        self.guidance_type = data.get('guidance_type', 'pro_nav')

        # Limits
        self.UAV_limits = MDC.UAVLimits(**data.get('limits', {}))
        
        # PID Gains
        pid_data = data.get('pid_gains', {})
        self.UAV_PID_Gains = MDC.UAVPIDGains(
            Roll=MDC.PIDGains(**pid_data.get('roll', {})),
            Pitch=MDC.PIDGains(**pid_data.get('pitch', {})),
            Heading=MDC.PIDGains(**pid_data.get('heading', {}))
        )

        # Navigation
        nav_data = data.get('navigation', {})
        self.UAV_navigation = MDC.UAVNavigation(
             N=nav_data.get('N', 2.0),
             dt=nav_data.get('dt', 0.1),
             cp=nav_data.get('cp', 360.0),
             acceleration_stage_in_sec=nav_data.get('acceleration_stage_in_sec', 2.0),
             dive_at=nav_data.get('dive_at', 30000.0),
             tan_ref=nav_data.get('tan_ref', 2000.0),
             theta_min_cruise=nav_data.get('theta_min_cruise', -30.0),
             theta_max_cruise=nav_data.get('theta_max_cruise', 30.0),
             theta_min=nav_data.get('theta_min', -70.0),
             theta_max=nav_data.get('theta_max', 70.0),
             alt_cruise=nav_data.get('alt_cruise', 15000.0)
        )

        # Simulation
        sim_data = data.get('simulation', {})
        self.UAV_simulation = MDC.UAVSimulation(
            Sim_time_step=sim_data.get('sim_time_step', 6),
            Control_time_step=sim_data.get('control_time_step', 10)
        )

        # Performance
        perf_data = data.get('performance', {})
        self.UAV_performance = MDC.UAVPerformance(
            target_lost_below_mach=perf_data.get('target_lost_below_mach', 1.0),
            target_lost_below_alt=perf_data.get('target_lost_below_alt', 0.0),
            lost_count=perf_data.get('lost_count', 3),
            effective_radius=perf_data.get('effective_radius', 300.0)
        )

        # Visualization (optional, defaults to AIM-7 dimensions)
        viz_data = data.get('visualization', {})
        self.visualization = UAVVisualization(**viz_data)

class AircraftConfig:
    def __init__(self, data):
        self.name = data.get('name')
        self.type = data.get('type')
        self.fdm_xml = data.get('fdm_xml')
        self.weight_lbs = data.get('weight_lbs')
        self.length_in = data.get('length_in')
        self.data_output_xml = data.get('data_output_xml', None)
        self.fg_sleep_time = data.get('fg_sleep_time', None)
        self.autopilot_type = data.get('autopilot_type', 'AircraftPIDAutopilot')
        self.guidance_type = data.get('guidance_type', None)

        # Limits
        self.aircraft_limits = ADC.AircraftLimits(**data.get('limits', {}))

        # PID Gains
        pid_data = data.get('pid_gains', {})
        self.aircraft_PID_Gains = ADC.AircraftPIDGains(
            Roll=ADC.PIDGains(**pid_data.get('roll', {})),
            Roll_sec=ADC.PIDGains(**pid_data.get('roll_sec', {})),
            Pitch=ADC.PIDGains(**pid_data.get('pitch', {})),
            Rudder_theta=ADC.PIDGains(**pid_data.get('rudder_theta', {})),
            Rudder_psi=ADC.PIDGains(**pid_data.get('rudder_psi', {})),
            Elevator_psi=ADC.PIDGains(**pid_data.get('elevator_psi', {}))
        )

        # Navigation
        nav_data = data.get('navigation', {})
        # Note: Aircraft dataclass uses capitalized fields
        self.aircraft_navigation = ADC.AircraftNavigation(
             Head_act_space_min=nav_data.get('head_act_space_min'),
             Head_act_space_max=nav_data.get('head_act_space_max'),
             Tan_ref=nav_data.get('tan_ref'),
             Mach_min=nav_data.get('mach_min'),
             Mach_max=nav_data.get('mach_max'),
             V_down_min=nav_data.get('v_down_min'),
             V_down_max=nav_data.get('v_down_max'),
             Roll_max=nav_data.get('roll_max'),
             Pitch_max=nav_data.get('pitch_max'),
             Pitch_min=nav_data.get('pitch_min'),
             Alt_act_space_min=nav_data.get('alt_act_space_min'),
             Alt_act_space_max=nav_data.get('alt_act_space_max'),
             Climb_theta_max=nav_data.get('climb_theta_max'),
             Dive_theta_max=nav_data.get('dive_theta_max'),
             Theta_act_space_min=nav_data.get('theta_act_space_min'),
             Theta_act_space_max=nav_data.get('theta_act_space_max')
        )
        
        # Simulation
        sim_data = data.get('simulation', {})
        self.aircraft_simulation = ADC.AircraftSimulation(
            Sim_time_step=sim_data.get('sim_time_step', 6),
            Control_time_step=sim_data.get('control_time_step', 10)
        )
