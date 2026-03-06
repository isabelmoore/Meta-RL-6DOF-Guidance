# Copyright (c) 2026 Isabel Moore. All rights reserved.
import gymnasium as gym
from gymnasium import spaces

import numpy as np

from simulation.agents.config import blue_agent, red_agent
from simulation.agents.agents import RLBVRAgent, BTBVRAgent

from simulation.core.geospatial import dinstance_between_agents, bearing_between_agents, relative_bearing_between_agents, to_360
import pymap3d as pm
from simulation.core.loggers import TacviewLogger

from simulation.core.scale import scale_between_inv, scale_between

from simulation.bts.bts import BVRBT

class BVRBase(gym.Env):
    def __init__(self, conf):
        # Environment config file 
        super().__init__()
        self.conf = conf
        self.obs_shape = conf.observation_shape
        self.act_shape = conf.action_shape

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=self.obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.act_shape, dtype=np.float32)        

        self.state = None
        self.done = False
        self.tacview_logger = None
        self.observation = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.blue_agent = RLBVRAgent(blue_agent, self)
        
        self.red_agent = BTBVRAgent(red_agent, self)
        
        self.red_agent.load_BT(BVRBT)
        
        self.all_agents = [self.blue_agent, self.red_agent]

        self.blue_agent.set_target(self.red_agent)

        self.red_agent.set_target(self.blue_agent)
        


        self.update_state()
        self.last_raw_action = np.zeros(self.act_shape)

        return self.state, {}

    def log_tacview(self):

        if self.conf.tacview_output_dir is not None:
            if self.tacview_logger is None:
                self.tacview_logger = TacviewLogger(self)
            elif self.done:
                self.tacview_logger.save_logs()
            else:
                self.tacview_logger.log_flight_data()
              
    def update_state(self):
        self.update_observation()
        obs_nn = self.from_obs2nn(self.blue_agent)
        
        if self.state is None:
            self.state = np.tile(obs_nn, (self.obs_shape[0], 1))
        else:
            self.state = np.roll(self.state, shift=-1, axis=0)
            self.state[-1,:] = obs_nn

    def update_observation(self):
        
        self.observation['bearing'] = to_360(bearing_between_agents(self.blue_agent, self.blue_agent.target))
        self.observation['heading'] = self.blue_agent.simObj.get_psi()
        self.observation['mach'] = self.blue_agent.simObj.get_mach()
        self.observation['altitude'] = self.blue_agent.simObj.get_altitude()

        self.observation['d'] = dinstance_between_agents(self.blue_agent, self.blue_agent.target)

        self.observation['enemy_bearing'] = to_360(bearing_between_agents(self.red_agent, self.red_agent.target)) 

        self.observation['enemy_heading'] = self.red_agent.simObj.get_psi()
        self.observation['enemy_mach'] = self.red_agent.simObj.get_mach()
        self.observation['enemy_altitude'] = self.red_agent.simObj.get_altitude()

        self.observation['own_UAV_active'] = 0
        self.observation['enemy_UAV_active'] = 0
        

    def step(self, action):
        # Store raw action for reward calculation (copy to avoid mutation)
        self.last_raw_action = action.copy()
        # apply action to agent
        action = self.from_nn2agent(action, self.blue_agent)
        
        for i in range(self.conf.step_length):
            # If step_length is 10, this should result aplllying action for 10 sim seconds, unless you changed the sim step time in FDM config
            self.blue_agent.apply_action(action)
            self.blue_agent.last_action = action
            self.red_agent.apply_action()
            # get new observation
            self.update_state()
            
            self.done = self.is_done()
            # calculate reward
            # check done
            
            self.reward = self.get_reward(self.done)
            if self.done:
                break
        return self.state, self.reward, self.done, self.max_episode_time_passed(), {'done': self.done, 'trunk': self.max_episode_time_passed()}

    def get_red_agent_actions(self):
        pass

    def from_obs2nn(self, agent):
        
        bearing_sin = np.sin(np.radians(self.observation['bearing']))
        bearing_cos = np.cos(np.radians(self.observation['bearing']))
        heading_sin = np.sin(np.radians(self.observation['heading']))
        heading_cos = np.cos(np.radians(self.observation['heading']))
        
        mach = scale_between(self.observation['mach'], a_min = 0.1, a_max = 1.5)
        altitude = scale_between(self.observation['altitude'], a_min = agent.simObj.conf.aircraft_limits.alt_min,
                               a_max = agent.simObj.conf.aircraft_limits.alt_max )
        d = scale_between(self.observation['d'], a_min = 0.0, a_max = 120e3)
        
        enemy_bearing_sin= np.sin(np.radians(self.observation['enemy_bearing']))
        enemy_bearing_cos= np.cos(np.radians(self.observation['enemy_bearing']))
        enemy_heading_sin = np.sin(np.radians(self.observation['enemy_heading']))
        enemy_heading_cos = np.cos(np.radians(self.observation['enemy_heading']))
        
        enemy_mach = scale_between(self.observation['enemy_mach'], a_min = 0.1, a_max = 1.5)
        enemy_altitude = scale_between(self.observation['enemy_altitude'], a_min = agent.simObj.conf.aircraft_limits.alt_min,
                               a_max = agent.simObj.conf.aircraft_limits.alt_max )
        
        return np.array([bearing_sin, bearing_cos, heading_sin, heading_cos, mach, altitude, d, enemy_bearing_sin, enemy_bearing_cos, 
                         enemy_heading_sin, enemy_heading_cos, enemy_mach, enemy_altitude, 
                         self.observation['own_UAV_active'], self.observation['enemy_UAV_active']])
        
        
        

    
    def from_nn2agent(self, action, agent):
        # heading 
        action[0] = scale_between_inv(action[0],
                                      a_min= agent.simObj.conf.aircraft_limits.psi_min,
                                        a_max= agent.simObj.conf.aircraft_limits.psi_max)        
        # altitude 
        action[1] = scale_between_inv(action[1],
                                      a_min= agent.simObj.conf.aircraft_limits.alt_min,
                                        a_max= agent.simObj.conf.aircraft_limits.alt_max)
        # throttle full thrust without or with afterburner 
        action[2] = 0.49 if action[2] <= 0.0 else 0.69
        return action


    def get_reward(self, is_done):
        """
        LOS rate shaping + closing velocity reward (paper-style).
        Incentivizes: reducing LOS rotation rate AND closing distance to target.
        Penalizes: excessive control effort.
        """
        distance = dinstance_between_agents(self.blue_agent, self.blue_agent.target)

        # --- LOS rate shaping ---
        lat0 = self.blue_agent.simObj.get_lat_gc_deg()
        lon0 = self.blue_agent.simObj.get_long_gc_deg()
        h0 = self.blue_agent.simObj.get_altitude()
        lat = self.blue_agent.target.simObj.get_lat_gc_deg()
        lon = self.blue_agent.target.simObj.get_long_gc_deg()
        h = self.blue_agent.target.simObj.get_altitude()
        e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0, deg=True)
        los = np.array([e, n, u])
        los_norm = np.linalg.norm(los)
        if los_norm > 1e-6:
            los_unit = los / los_norm
        else:
            los_unit = np.array([0.0, 0.0, 0.0])

        # LOS rate via finite difference
        if not hasattr(self, '_prev_los_unit_bvr'):
            self._prev_los_unit_bvr = los_unit.copy()
        omega = los_unit - self._prev_los_unit_bvr
        self._prev_los_unit_bvr = los_unit.copy()

        sigma = 0.02
        omega_sq = np.dot(omega, omega)
        r_shaping = np.exp(-omega_sq / (sigma ** 2))

        # --- Closing velocity reward ---
        if not hasattr(self, '_prev_distance_bvr'):
            self._prev_distance_bvr = distance
        closing_speed = self._prev_distance_bvr - distance  # positive = getting closer
        self._prev_distance_bvr = distance
        # Normalize: BVR distances ~120km, aircraft ~330 m/s
        r_closing = 0.1 * max(0.0, closing_speed / 500.0)

        # Control effort penalty
        r_ctrl = -0.01 * np.sum(np.abs(self.last_raw_action))

        reward = r_shaping + r_closing + r_ctrl

        # Terminal rewards
        if is_done:
            if self.red_agent.healthPoints <= 0:
                reward += 10.0    # Kill bonus
            elif self.blue_agent.healthPoints <= 0:
                reward -= 10.0    # Death penalty
            elif self.max_episode_time_passed():
                reward -= 5.0     # Timeout penalty

        return reward

    def is_done(self):
        for agent in self.all_agents:
            if agent.healthPoints <= 0.0:
                return True
            
        
        if self.max_episode_time_passed():
            return True 
        return False
    
    def max_episode_time_passed(self):
        if self.blue_agent.simObj.get_sim_time_sec() >= self.conf.max_episode_time:
            return True 
        return False
               

