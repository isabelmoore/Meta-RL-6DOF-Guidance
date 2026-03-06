# Copyright (c) 2026 Isabel Moore. All rights reserved.
import jsbsim
import numpy as np
from simulation.core.units import m2f, f2m, lbs2kg

class FDMObject(object):
    """Wrapper around a JSBSim FGFDMExec instance for a single vehicle."""

    def __init__(self, conf):
        """Initialize JSBSim FDM from a vehicle config.

        Args:
            conf: Vehicle configuration namespace with fdm_xml, data_output_xml,
                  and mass/geometry/aero/fcs dicts for XML generation.
        """
        if hasattr(conf, 'aero') and conf.aero:
            from simulation.core.xml_generator import generate_aircraft_xml
            aircraft_dir = f"jsbsim_data/aircraft/{getattr(conf, 'fdm_aircraft', 'AIM')}"
            generate_aircraft_xml(conf, output_dir=aircraft_dir)

        self.fdm = jsbsim.FGFDMExec('jsbsim_data', None)
        script_path = conf.fdm_xml
        if script_path.startswith('jsbsim_data/'):
            script_path = script_path[len('jsbsim_data/'):]
        self.fdm.load_script(script_path)
        if conf.data_output_xml is not None:
            self.fdm.set_output_directive(conf.data_output_xml)

        fcs = getattr(conf, 'fcs', {})
        import math
        self._ail_pos_max = math.radians(fcs.get('aileron_pos_max_deg', 0.1))
        self._elev_pos_max = math.radians(fcs.get('elevator_pos_max_deg', 10.0))
        self._rud_pos_max = math.radians(fcs.get('rudder_pos_max_deg', 10.0))
    
    def set_target(self, target):
        """
        Set the target object for this FDM object.
        This is used for relative state calculations and potentially for logic 
        that depends on the other agent (e.g. evasion).
        """
        self.target = target

    def reset(self, lat, long, alt, vel, heading):
        """Reset FDM to initial conditions and run one frame.

        Args:
            lat: Geocentric latitude in degrees.
            long: Geocentric longitude in degrees.
            alt: Altitude above sea level in meters.
            vel: Forward speed in m/s.
            heading: True heading in degrees.
        """
        # input  lat long in deg
        self.fdm.set_property_value("ic/lat-gc-deg", lat)
        self.fdm.set_property_value("ic/long-gc-deg", long)
        # input  altitude in meters 
        self.fdm.set_property_value("ic/h-sl-ft", m2f(alt))
        # input vel in m/s      
        self.fdm['ic/u-fps'] = m2f(vel)
        # input  heading in deg    
        self.fdm['ic/psi-true-rad'] = np.radians(heading)

        self.fdm.set_property_value('propulsion/set-running', -1)

        self.fdm.reset_to_initial_conditions(0)
        
        self.fdm.run()
        
         
    def get_lat_gc_deg(self):
        """Return geocentric latitude in degrees."""
        return self.fdm['position/lat-gc-deg']

    def get_long_gc_deg(self):
        """Return geocentric longitude in degrees."""
        return self.fdm['position/long-gc-deg']

    def get_sim_time_sec(self):
        """Return simulation time in seconds."""
        return self.fdm['simulation/sim-time-sec']

    def get_mach(self):
        """Return current Mach number."""
        return self.fdm['velocities/mach']


    def get_phi(self, in_deg = True):
        """Return roll angle (degrees by default, radians if in_deg=False)."""
        if in_deg:
            return  self.fdm['attitude/phi-deg']
        return self.fdm['attitude/phi-rad']

    def get_theta(self, in_deg = True):
        """Return pitch angle (degrees by default, radians if in_deg=False)."""
        if in_deg:
            return self.fdm['attitude/theta-deg']
        return  self.fdm['attitude/theta-rad']

    def get_psi(self, in_deg = True):
        """Return yaw/heading angle (degrees by default, radians if in_deg=False)."""
        if in_deg:
            return  self.fdm['attitude/psi-deg']
        return  self.fdm['attitude/psi-rad']

    def get_altitude(self):
        """Return altitude above sea level in meters."""
        return self.fdm['position/h-sl-meters']


    def get_true_airspeed(self):
        """Return true airspeed in m/s."""
        return f2m(self.fdm['velocities/vt-fps'])


    def get_v_north(self):
        """Return northward velocity component in m/s."""
        return f2m(self.fdm['velocities/v-north-fps'])

    def get_v_east(self):
        """Return eastward velocity component in m/s."""
        return f2m(self.fdm['velocities/v-east-fps'])

    def get_v_down(self):
        """Return downward velocity component in m/s."""
        return f2m(self.fdm['velocities/v-down-fps'])


    def get_u(self):
        """Return body-frame forward velocity in m/s."""
        return f2m(self.fdm['u-fps'])

    def get_total_fuel(self):
        """Return total fuel mass in kg."""
        return lbs2kg(self.fdm['propulsion/total-fuel-lbs'])

    def set_retract_gear(self):
        """Retract landing gear (set gear position to 0)."""
        self.fdm['gear/gear-pos-norm'] = 0


    def set_aileron(self, cmd):
        """Set aileron command in [-1, 1]."""
        self.fdm['fcs/aileron-cmd-norm'] = cmd

    def set_elevator(self, cmd):
        """Set elevator command in [-1, 1]."""
        self.fdm['fcs/elevator-cmd-norm'] = cmd

    def set_rudder(self, cmd):
        """Set rudder command in [-1, 1]."""
        self.fdm['fcs/rudder-cmd-norm'] = cmd

    def set_throttle(self, cmd):
        """Set throttle command in [0, 1]."""
        self.fdm['fcs/throttle-cmd-norm'] = cmd


    # --- Body angular rates ---
    def get_p_rad_sec(self):
        """Return roll rate in rad/s."""
        return self.fdm['velocities/p-rad_sec']

    def get_q_rad_sec(self):
        """Return pitch rate in rad/s."""
        return self.fdm['velocities/q-rad_sec']

    def get_r_rad_sec(self):
        """Return yaw rate in rad/s."""
        return self.fdm['velocities/r-rad_sec']

    # --- Body-frame accelerations ---
    def get_body_accel_mps2(self):
        """Return body-frame acceleration [ax, ay, az] in m/s^2."""
        ax = f2m(self.fdm['accelerations/udot-ft_sec2'])
        ay = f2m(self.fdm['accelerations/vdot-ft_sec2'])
        az = f2m(self.fdm['accelerations/wdot-ft_sec2'])
        return np.array([ax, ay, az])

    def get_n_pilot(self):
        """Return pilot load factor (Nz) in g's."""
        return self.fdm['accelerations/Nz']

    # --- Quaternion from Euler angles ---
    def get_quaternion(self):
        """Return attitude quaternion [w, x, y, z] from Euler angles (ZYX convention)."""
        phi = self.get_phi(in_deg=False)
        theta = self.get_theta(in_deg=False)
        psi = self.get_psi(in_deg=False)
        # ZYX convention: q = Rz(psi) * Ry(theta) * Rx(phi)
        cy, sy = np.cos(psi / 2), np.sin(psi / 2)
        cp, sp = np.cos(theta / 2), np.sin(theta / 2)
        cr, sr = np.cos(phi / 2), np.sin(phi / 2)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z])

    # --- Current fin positions (read FCS integrator outputs, normalized to [-1, 1]) ---
    def get_aileron_pos(self):
        """Return aileron position normalized to [-1, 1]."""
        return self.fdm['fcs/left-aileron-pos-rad'] / self._ail_pos_max

    def get_elevator_pos(self):
        """Return elevator position normalized to [-1, 1]."""
        return self.fdm['fcs/elevator-pos-rad'] / self._elev_pos_max

    def get_rudder_pos(self):
        """Return rudder position normalized to [-1, 1]."""
        return self.fdm['fcs/rudder-pos-rad'] / self._rud_pos_max

    def print_status(self):
        """Print current time, position, altitude, Mach, and heading."""
        print(f"Time: {self.get_sim_time_sec():.1f} s, Lat: {self.get_lat_gc_deg():.4f} deg, Long: {self.get_long_gc_deg():.4f} deg, Alt: {self.get_altitude():.1f} m, Mach: {self.get_mach():.3f}, Heading: {self.get_psi():.1f} deg")
