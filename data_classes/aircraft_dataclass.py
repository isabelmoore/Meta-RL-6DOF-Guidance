from dataclasses import dataclass

@dataclass
class AircraftLimits:
    phi_min: float
    phi_max: float
    theta_min: float
    theta_max: float
    psi_min: float
    psi_max: float
    alt_min: float
    alt_max: float
    thr_min: float
    thr_max: float


@dataclass
class PIDGains:
    P: float
    I: float
    D: float
    Deriv: float
    Integ: float
    Integ_max: float
    Integ_min: float

@dataclass
class AircraftPIDGains:
    Roll: PIDGains
    Roll_sec: PIDGains
    Pitch: PIDGains
    Rudder_theta: PIDGains
    Rudder_psi: PIDGains
    Elevator_psi: PIDGains


@dataclass
class AircraftNavigation:
    Head_act_space_min: float
    Head_act_space_max: float
    Tan_ref: float
    Mach_min: float
    Mach_max: float
    V_down_min: float
    V_down_max: float  
    Roll_max: float
    Pitch_max: float
    Pitch_min: float
    Alt_act_space_min: float
    Alt_act_space_max: float
    Climb_theta_max: float
    Dive_theta_max: float
    Theta_act_space_min: float
    Theta_act_space_max: float

@dataclass
class AircraftSimulation:
    Sim_time_step: float
    Control_time_step: float