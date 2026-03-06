from dataclasses import dataclass

@dataclass
class UAVLimits:
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
class UAVPIDGains:
    Roll: PIDGains
    Pitch: PIDGains
    Heading: PIDGains


@dataclass
class UAVNavigation:
    N: float
    dt: float
    cp: float
    acceleration_stage_in_sec: float
    dive_at: float  
    tan_ref: float
    theta_min_cruise: float
    theta_max_cruise: float
    theta_min: float
    theta_max: float
    alt_cruise: float
    
@dataclass
class UAVSimulation:
    Sim_time_step: float
    Control_time_step: float

@dataclass
class UAVPerformance:
    target_lost_below_mach: float
    target_lost_below_alt: float
    lost_count: float
    effective_radius: float

@dataclass
class UAVVisualization:
    # Body geometry (meters)
    length: float = 6.25
    body_radius: float = 0.155
    cg_offset: float = 3.13
    nose_length: float = 0.94
    # Tail fin geometry
    fin_root_offset: float = 5.5
    fin_span: float = 0.63
    fin_sweep: float = 0.18
    fin_tip_chord_ratio: float = 0.55
    fin_thickness: float = 0.008
    # Wing geometry
    wing_cg_offset: float = 2.0
    wing_chord: float = 0.7
    wing_span: float = 0.40
    wing_sweep: float = 0.35
    wing_tip_chord_ratio: float = 0.3
    # GIF rendering
    n_frames: int = 120
    body_color: str = '#a8c4d8'
    fin_color: str = '#90b0c4'