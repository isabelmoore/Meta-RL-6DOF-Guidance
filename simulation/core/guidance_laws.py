# Proportional Navigation (PN) guidance law.
#
# PN steers the UAV by commanding acceleration perpendicular to the
# line-of-sight (LOS) vector, proportional to the LOS rotation rate.
# The constant N (navigation ratio, typically 2-5) controls how
# aggressively the UAV leads the target.
#
# Core equation:  a_cmd = N * (V_rel x Omega)
#   where Omega = (R x V_rel) / |R|^2  is the LOS rotation rate vector
#         R     = target position in UAV-centered ENU frame
#         V_rel = target velocity - UAV velocity
#
# The guidance output is converted to heading and altitude commands
# for the PID autopilot.

import numpy as np
import pymap3d as pm
from simulation.core.geospatial import angle_between

class PN:
    def __init__(self, conf):
        self.N = conf.UAV_navigation.N      # navigation ratio (dimensionless)
        self.dt = conf.UAV_navigation.dt     # integration timestep for velocity prediction
        self.distance_to_target = None

    def get_target_ENU(self, UAV):
        """Get target position in ENU frame centered on the UAV."""
        lat0 = UAV.get_lat_gc_deg()
        lon0 = UAV.get_long_gc_deg()
        h0 = UAV.get_altitude()

        try:
            lat = UAV.target.simObj.get_lat_gc_deg()
            lon = UAV.target.simObj.get_long_gc_deg()
            h = UAV.target.simObj.get_altitude()
        except AttributeError:
            lat = UAV.target.get_lat_gc_deg()
            lon = UAV.target.get_long_gc_deg()
            h = UAV.target.get_altitude()

        east, north , up = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)

        return np.array([east, north, up])

    def get_target_v_ENU(self, UAV):
        """Get target velocity in ENU frame."""
        try:
            v_east =UAV.target.simObj.get_v_east()
            v_north = UAV.target.simObj.get_v_north()
            v_up = -UAV.target.simObj.get_v_down()

        except AttributeError:
            v_east =UAV.target.get_v_east()
            v_north = UAV.target.get_v_north()
            v_up = -UAV.target.get_v_down()
        return np.array([v_east, v_north, v_up])

    def get_v_ENU(self, UAV):
        """Get UAV velocity in ENU frame."""
        v_east = UAV.get_v_east()
        v_north = UAV.get_v_north()
        v_up = -UAV.get_v_down()

        return np.array([v_east, v_north, v_up])

    def get_heading_rel_direction(self, v1,v2):
        """Return +1 or -1 indicating turn direction (right-hand rule on Z-axis cross product)."""
        if np.cross(v1, v2)[2] < 0:
            return 1
        return -1

    def get_target_altitude(self, UAV):
        """Get target altitude in meters."""
        try:
            alt = UAV.target.simObj.get_altitude()
        except AttributeError:
            alt = UAV.target.get_altitude()
        return alt

    def get_guidance(self, UAV):
        """Compute PN guidance commands (heading, altitude) for the autopilot.

        Returns:
            heading_cmd: desired heading in degrees [0, 360)
            altitude_cmd: desired altitude in meters (target alt + lead correction)
        """
        # R = target position relative to UAV in ENU
        taget_ENU = self.get_target_ENU(UAV)

        target_v_ENU = self.get_target_v_ENU(UAV)

        v_ENU = self.get_v_ENU(UAV)

        # V_rel = closing velocity vector (positive toward target)
        v_rel_ENU = target_v_ENU - v_ENU

        # LOS rotation rate: Omega = (R x V_rel) / |R|^2
        # This is the angular velocity of the LOS vector
        rotation_vector = (np.cross(taget_ENU, v_rel_ENU)) / (taget_ENU @ taget_ENU)

        # PN acceleration command: a_cmd = N * (V_rel x Omega)
        # This produces acceleration perpendicular to both V_rel and Omega,
        # steering the UAV to maintain a collision course
        acc_cmd_ENU = self.N * np.cross(v_rel_ENU , rotation_vector)

        # Predict velocity after applying PN correction for one timestep
        v_ENU_PN = v_ENU + acc_cmd_ENU*self.dt

        # Extract heading change from the PN-corrected velocity
        # Project both current and corrected velocity onto the horizontal plane
        v1 = v_ENU.copy()
        v2 = v_ENU_PN.copy()

        v1[2] = 0.0
        v2[2] = 0.0

        heading_PN = angle_between(v1, v2, in_deg= True)

        hrd = self.get_heading_rel_direction(v1, v2)

        heading_cmd = (UAV.get_psi() +  hrd * heading_PN) %360

        # Altitude command: aim where the target will be at intercept time
        # time_to_impact = range / closing speed
        v_rel_ENU_norm = np.linalg.norm(v_rel_ENU)
        taget_ENU_norm = np.linalg.norm(taget_ENU)
        time_to_impact = taget_ENU_norm/v_rel_ENU_norm
        altitude_cmd = self.get_target_altitude(UAV) + target_v_ENU[2]*time_to_impact
        self.distance_to_target = taget_ENU_norm
        return heading_cmd, altitude_cmd
