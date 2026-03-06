# Copyright (c) 2026 Isabel Moore. All rights reserved.

import numpy as np
geodesic = pyproj.Geod(ellps='WGS84')

def get_random_position_in_circle(lat0 , long0, d = [40e3, 100e3], b = [0, 360]):
    # input in [km] 
    # get a random lat long for aircraft/UAV position
    # lat0, long0 [deg]
    # d [km] distance 
    # b [deg] bearing 
    origin = geopy.Point(lat0, long0)        
    d = np.sqrt(np.random.uniform(d[0]**2, d[1]**2))
    b = np.random.uniform(b[0], b[1])
    destination = geodesic(meters=d).destination(origin, b)
    lat, long = destination.latitude, destination.longitude      
    return lat, long, d, b

def db2latlong(sef, lat0, long0, d, b):
    #from distance and bearing to lat and long 
    # lat0, long0 [deg]
    # d [km] distance 
    # b [deg] bearing 
    origin = geopy.Point(lat0, long0)        
    destination = geodesic(meters=d).destination(origin, b)
    lat, long = destination.latitude, destination.longitude      
    return lat, long

def get_bearing(self, lat_tgt, long_tgt, lat, long):
    fwd_azimuth, back_azimuth, distance = self.geodesic.inv(long_tgt, lat_tgt, long, lat)
    if fwd_azimuth < 0:
        fwd_azimuth += 360
    return fwd_azimuth


def get_relative_unit_position_NED(self, lat0, lon0, h0, lat, lon, h):
    # The local coordinate origin
    #lat0 = TAU.get_lat_gc_deg() # deg
    #lon0 = TAU.get_long_gc_deg()  # deg
    #h0 = TAU.get_altitude()     # meters

    # The point of interest
    #lat = Tgt.get_lat_gc_deg() # deg
    #lon = Tgt.get_long_gc_deg()  # deg
    #h = Tgt.get_altitude()     # meters

    east, north , up = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
    down = -up
    self.d_tgt_east = east
    self.d_tgt_north = north
    self.d_tgt_down = down
    try:
        self.position_tgt_NED_norm = round(np.linalg.norm(np.array([east, north, down])))            
    except ValueError:          
        print('position_tgt_NED_norm value error')
        print(east, north, down)
        print(lat, lon, h, lat0, lon0, h0, self.position_tgt_NED_norm_old)
        self.position_tgt_NED_norm = self.position_tgt_NED_norm_old
        print('Sim, time: ', self.get_sim_time_sec())
        
    self.position_tgt_NED_norm_old = self.position_tgt_NED_norm
    
    return east, north, down

def x_rotation(self, vector,theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
    return np.dot(R,vector)

def y_rotation(self, vector,theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R,vector)

def z_rotation(self, vector,theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.dot(R,vector)
