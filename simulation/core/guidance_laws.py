# Copyright (c) 2026 Isabel Moore. All rights reserved.
import numpy as np    
import pymap3d as pm
from simulation.core.geospatial import angle_between

class PN:
    def __init__(self, conf):
        self.N = conf.UAV_navigation.N
        self.dt = conf.UAV_navigation.dt
        self.distance_to_target = None
        
    def get_target_ENU(self, UAV):
        # The local coordinate origin
        lat0 = UAV.get_lat_gc_deg() # deg
        lon0 = UAV.get_long_gc_deg()  # deg
        h0 = UAV.get_altitude()     # meters

        # The point of interest
        try:
            lat = UAV.target.simObj.get_lat_gc_deg() # deg
            lon = UAV.target.simObj.get_long_gc_deg()  # deg
            h = UAV.target.simObj.get_altitude()     # meters
        except AttributeError:
            lat = UAV.target.get_lat_gc_deg() # deg
            lon = UAV.target.get_long_gc_deg()  # deg
            h = UAV.target.get_altitude()     # meters

        east, north , up = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
        
        return np.array([east, north, up])
    
    def get_target_v_ENU(self, UAV):
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
        v_east = UAV.get_v_east()
        v_north = UAV.get_v_north()
        v_up = -UAV.get_v_down()

        return np.array([v_east, v_north, v_up])

    def get_heading_rel_direction(self, v1,v2):
        if np.cross(v1, v2)[2] < 0:
            return 1
        return -1

    def get_target_altitude(self, UAV):
        try:
            alt = UAV.target.simObj.get_altitude()
        except AttributeError:
            alt = UAV.target.get_altitude()
        return alt

    def get_guidance(self, UAV):
        taget_ENU = self.get_target_ENU(UAV)
        
        target_v_ENU = self.get_target_v_ENU(UAV)

        v_ENU = self.get_v_ENU(UAV)
        
        v_rel_ENU = target_v_ENU - v_ENU

        rotation_vector = (np.cross(taget_ENU, v_rel_ENU)) / (taget_ENU @ taget_ENU)
        
        acc_cmd_ENU = self.N * np.cross(v_rel_ENU , rotation_vector)

        v_ENU_PN = v_ENU + acc_cmd_ENU*self.dt
        
        # get heading
        v1 = v_ENU.copy()
        v2 = v_ENU_PN.copy()
        
        v1[2] = 0.0   
        v2[2] = 0.0

        heading_PN = angle_between(v1, v2, in_deg= True)

        hrd = self.get_heading_rel_direction(v1, v2)
        
        heading_cmd = (UAV.get_psi() +  hrd * heading_PN) %360
        
        v_rel_ENU_norm = np.linalg.norm(v_rel_ENU)
        taget_ENU_norm = np.linalg.norm(taget_ENU)
        time_to_impact = taget_ENU_norm/v_rel_ENU_norm
        altitude_cmd = self.get_target_altitude(UAV) + target_v_ENU[2]*time_to_impact
        self.distance_to_target = taget_ENU_norm
        return heading_cmd, altitude_cmd