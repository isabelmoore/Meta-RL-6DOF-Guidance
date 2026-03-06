# Copyright (c) 2026 Isabel Moore. All rights reserved.
import numpy as np
from simulation.core.control import PID
from simulation.core.navigation import roll_circle_clip, delta_heading

class AircraftPIDAutopilot:
	"""
	Aircraft PID controller class
	"""

	def __init__(self, aircraft):
		self.aircraft = aircraft
		self.aircraft_PID_Gains = self.aircraft.conf.aircraft_PID_Gains
		self.aircraft_navigation = self.aircraft.conf.aircraft_navigation
		self.aircraft_limits = self.aircraft.conf.aircraft_limits
		self.reset_controllers()
	
	def reset_controllers(self):
		
		self.roll_PID = PID(self.aircraft_PID_Gains.Roll)
		self.roll_sec_PID = PID(self.aircraft_PID_Gains.Roll_sec)
		self.pitch_PID = PID(self.aircraft_PID_Gains.Pitch)
		self.rudder_theta_PID = PID(self.aircraft_PID_Gains.Rudder_theta)
		self.rudder_psi_PID = PID(self.aircraft_PID_Gains.Rudder_psi)
		self.elevator_psi_PID = PID(self.aircraft_PID_Gains.Elevator_psi)

		self.rudder_cmd = 0.0
		self.elevator_cmd = 0.0
		self.aileron_cmd = 0.0

		self.alt_act_space = self.aircraft_navigation.Alt_act_space_min
		self.head_act_space = self.aircraft_navigation.Head_act_space_min
		self.theta_act_space = self.aircraft_navigation.Theta_act_space_min


	def set_roll_PID(self, roll_ref, secondary_pid=False):
        # make sure within range
		roll_ref = np.clip(roll_ref, self.aircraft_limits.phi_min, self.aircraft_limits.phi_max)
        
        # save for recording and other stuff
        
        # get reference value for control input 
		diff = roll_circle_clip(roll_ref- self.aircraft.get_phi())
        
		if secondary_pid:
			cmd = -self.roll_sec_PID.update(current_value=diff)
		else:
			cmd = -self.roll_PID.update(current_value=diff)

		self.aileron_cmd = np.clip(a = cmd, a_min = -1, a_max= 1)


	def get_control_input(self, diff_head, diff_alt):
        
		
		if abs(diff_head) >= self.head_act_space and abs(diff_alt) <= self.alt_act_space:
            
			# create larger altitude action space
			self.alt_act_space = self.aircraft_navigation.Alt_act_space_max
            # decide which direction to roll 
			roll_rot_dir = 1 if diff_head >= 0 else -1

			self.set_roll_PID(roll_ref= roll_rot_dir * (self.aircraft_navigation.Roll_max))
            # if roll is close to the limit 
			if self.aircraft_navigation.Roll_max - abs(self.aircraft.get_phi()) < 30:
                
				self.elevator_cmd = -0.3
			else:
				self.elevator_cmd = -0.9

			self.head_act_space = self.aircraft_navigation.Head_act_space_min

		else:
            # decrease margine adjust heading a bit 
			self.alt_act_space = self.aircraft_navigation.Alt_act_space_min
			theta_ref = np.degrees(np.arctan2(diff_alt, self.aircraft_navigation.Tan_ref))
			theta_ref = np.clip(a= theta_ref, a_min = self.aircraft_navigation.Dive_theta_max , a_max = self.aircraft_navigation.Climb_theta_max)

			if abs(diff_alt) > 1.5e3:

				self.theta_act_space = self.aircraft_navigation.Theta_act_space_max
                #self.conf.ctrl['theta_act_space'] = self.conf.ctrl['theta_act_space_max']
				self.set_roll_PID(roll_ref= 0.0)        

			elif abs(diff_alt) < 1.5e3 and abs(self.aircraft.get_theta()) > self.theta_act_space:
                
				self.theta_act_space = self.aircraft_navigation.Theta_act_space_min
				self.set_roll_PID(roll_ref= 0.0)
			else:
				self.theta_act_space = self.aircraft_navigation.Theta_act_space_max
				
				roll_rot_dir = 1 if diff_head >= 0 else -1
                
				if abs(diff_head) < 10:
                    
					self.set_roll_PID(roll_ref= roll_rot_dir * abs(diff_head), secondary_pid=True)
				else:
					self.set_roll_PID(roll_ref= roll_rot_dir * abs(diff_head)*3)
            
			self.set_pitch_PID(theta_ref)
			self.head_act_space = self.aircraft_navigation.Head_act_space_max

		return self.aileron_cmd, self.elevator_cmd, self.rudder_cmd			

        
	def set_pitch_PID(self, theta_ref):
		theta_ref = np.clip(theta_ref, self.aircraft_limits.theta_min, self.aircraft_limits.theta_max)
		diff = theta_ref - self.aircraft.get_theta()
		cmd = self.pitch_PID.update(current_value= diff)
		cmd = np.clip(a = cmd, a_min = -1, a_max= 1)
		self.elevator_cmd = cmd


class UAVPIDAutopilot:
	"""
	UAV PID controller class
	"""

	def __init__(self, UAV):
		self.UAV = UAV
		self.UAV_PID_Gains = self.UAV.conf.UAV_PID_Gains
		self.UAV_limits = self.UAV.conf.UAV_limits
		self.UAV_navigation = self.UAV.conf.UAV_navigation
		self.reset_controllers()
	
	def reset_controllers(self):
		
		self.roll_PID = PID(self.UAV_PID_Gains.Roll)
		self.pitch_PID = PID(self.UAV_PID_Gains.Pitch)
		self.heading_PID = PID(self.UAV_PID_Gains.Heading)

		self.rudder_cmd = 0.0
		self.elevator_cmd = 0.0
		self.aileron_cmd = 0.0

	def get_control_input(self, heading_cmd, altitude_cmd):
		# control output TODO
		if not self.acceleration_stage_done():
			heading_cmd = self.UAV.get_psi()
			altitude_cmd = self.UAV.get_altitude()

		self.aileron_cmd = self.set_roll_PID(roll_ref= 0.0)

		self.elevator_cmd = self.set_altitude_PID(ref= altitude_cmd)

		self.rudder_cmd = self.set_heading_PID(head_ref= heading_cmd)

		self.throttle_cmd = self.set_throttle()

		return self.aileron_cmd, self.elevator_cmd, self.rudder_cmd	, self.throttle_cmd


	def set_roll_PID(self, roll_ref):
        # make sure within range
		roll_ref = np.clip(roll_ref, self.UAV_limits.phi_min, self.UAV_limits.phi_max)
        
        # get reference value for control input 
		diff = roll_circle_clip(roll_ref- self.UAV.get_phi())
		cmd = -self.roll_PID.update(current_value=diff)
	
		return np.clip(a = cmd, a_min = -1, a_max= 1)

	def set_pitch_PID(self, theta_ref):
		diff = theta_ref - self.UAV.get_theta()
		pitch_cmd = self.pitch_PID.update(current_value= diff)
		pitch_cmd =  np.clip(pitch_cmd, -1, 1)
		return -pitch_cmd

	def set_heading_PID(self, head_ref):
		diff = delta_heading(head_ref, self.UAV.get_psi() )
		cmd = self.heading_PID.update(current_value= diff)
		return np.clip(cmd, -1, 1 )

	def set_throttle(self,cmd = 0.7):
		return cmd

	def set_altitude_PID(self, ref):
		# angle between different altitudes 
		diff_atl = ref - self.UAV.get_altitude()

		theta_ref = np.degrees(np.arctan2(diff_atl, self.UAV_navigation.tan_ref))
		
		theta_ref = np.clip(a = theta_ref, a_min = self.UAV_navigation.theta_min, a_max = self.UAV_navigation.theta_max)
		
		return self.set_pitch_PID(theta_ref)


	def acceleration_stage_done(self):
		if self.UAV.get_sim_time_sec() < self.UAV_navigation.acceleration_stage_in_sec:
			# still accelerating
			return False
		else:
			return True







