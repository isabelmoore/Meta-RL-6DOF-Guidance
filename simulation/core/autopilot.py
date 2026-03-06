# PID autopilot controllers for aircraft and UAV vehicles.
#
# AircraftPIDAutopilot: Controls F-16-style aircraft with coordinated turn logic.
#   - Heading changes above head_act_space threshold trigger bank-and-pull maneuvers
#   - Below that threshold, uses proportional roll + pitch-to-altitude
#   - Adaptive action space margins (alt_act_space, head_act_space, theta_act_space)
#     prevent oscillation by widening/narrowing deadbands based on error magnitude
#
# UAVPIDAutopilot: Simpler controller for UAV/missile vehicles.
#   - Wings-level roll hold, pitch-to-altitude, heading via rudder
#   - Enforces an acceleration stage at launch (holds heading/altitude)
#   - Throttle is fixed at 0.7 (full thrust)
#
# Both autopilots receive guidance commands (heading_cmd, altitude_cmd) from
# the PN guidance law and output fin commands [aileron, elevator, rudder] in [-1, 1].

import numpy as np
from simulation.core.control import PID
from simulation.core.navigation import roll_circle_clip, delta_heading

class AircraftPIDAutopilot:
	"""PID autopilot for maneuvering aircraft (F-16 target).

	Uses coordinated turn logic: large heading errors trigger a bank-and-pull
	maneuver, while small errors use proportional roll + pitch-to-altitude.
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

		# Adaptive deadband margins - start at minimum to be responsive,
		# widen when actively maneuvering to prevent oscillation
		self.alt_act_space = self.aircraft_navigation.Alt_act_space_min
		self.head_act_space = self.aircraft_navigation.Head_act_space_min
		self.theta_act_space = self.aircraft_navigation.Theta_act_space_min


	def set_roll_PID(self, roll_ref, secondary_pid=False):
		roll_ref = np.clip(roll_ref, self.aircraft_limits.phi_min, self.aircraft_limits.phi_max)
		diff = roll_circle_clip(roll_ref- self.aircraft.get_phi())

		if secondary_pid:
			cmd = -self.roll_sec_PID.update(current_value=diff)
		else:
			cmd = -self.roll_PID.update(current_value=diff)

		self.aileron_cmd = np.clip(a = cmd, a_min = -1, a_max= 1)


	def get_control_input(self, diff_head, diff_alt):
		"""Compute fin commands given heading and altitude errors.

		Two-mode logic:
		  Mode 1 (large heading error, small altitude error):
		    Bank to max roll toward the target heading, then pull elevator
		    to turn. Widens altitude margin to avoid fighting altitude
		    during the turn.

		  Mode 2 (small heading error or large altitude error):
		    Compute pitch reference from altitude error via arctan,
		    apply roll correction proportional to heading error.
		    Uses secondary PID for fine heading adjustments (<10 deg).
		"""

		if abs(diff_head) >= self.head_act_space and abs(diff_alt) <= self.alt_act_space:
			# MODE 1: Bank-and-pull turn
			# Widen altitude margin so we don't fight altitude during the turn
			self.alt_act_space = self.aircraft_navigation.Alt_act_space_max
			roll_rot_dir = 1 if diff_head >= 0 else -1

			self.set_roll_PID(roll_ref= roll_rot_dir * (self.aircraft_navigation.Roll_max))
			# Modulate elevator pull based on how close roll is to the target bank angle
			if self.aircraft_navigation.Roll_max - abs(self.aircraft.get_phi()) < 30:
				# Near target bank - gentle pull to sustain turn
				self.elevator_cmd = -0.3
			else:
				# Still rolling - stronger pull to accelerate the turn
				self.elevator_cmd = -0.9

			self.head_act_space = self.aircraft_navigation.Head_act_space_min

		else:
			# MODE 2: Pitch-to-altitude with proportional heading correction
			self.alt_act_space = self.aircraft_navigation.Alt_act_space_min
			# Convert altitude error to pitch reference via arctan
			# tan_ref controls how aggressively pitch responds to altitude error
			theta_ref = np.degrees(np.arctan2(diff_alt, self.aircraft_navigation.Tan_ref))
			theta_ref = np.clip(a= theta_ref, a_min = self.aircraft_navigation.Dive_theta_max , a_max = self.aircraft_navigation.Climb_theta_max)

			if abs(diff_alt) > 1.5e3:
				# Large altitude error: prioritize pitch, wings level
				self.theta_act_space = self.aircraft_navigation.Theta_act_space_max
				self.set_roll_PID(roll_ref= 0.0)

			elif abs(diff_alt) < 1.5e3 and abs(self.aircraft.get_theta()) > self.theta_act_space:
				# Moderate altitude error but pitch is overshooting: level off
				self.theta_act_space = self.aircraft_navigation.Theta_act_space_min
				self.set_roll_PID(roll_ref= 0.0)
			else:
				# Small altitude error: correct heading with proportional roll
				self.theta_act_space = self.aircraft_navigation.Theta_act_space_max

				roll_rot_dir = 1 if diff_head >= 0 else -1

				if abs(diff_head) < 10:
					# Fine heading adjustment - use secondary PID (lower gains)
					self.set_roll_PID(roll_ref= roll_rot_dir * abs(diff_head), secondary_pid=True)
				else:
					# Coarse heading adjustment - amplify roll command x3
					self.set_roll_PID(roll_ref= roll_rot_dir * abs(diff_head)*3)

			self.set_pitch_PID(theta_ref)
			self.head_act_space = self.aircraft_navigation.Head_act_space_max

		return self.aileron_cmd, self.elevator_cmd, self.rudder_cmd


	def set_pitch_PID(self, theta_ref):
		"""Compute elevator command from pitch angle error."""
		theta_ref = np.clip(theta_ref, self.aircraft_limits.theta_min, self.aircraft_limits.theta_max)
		diff = theta_ref - self.aircraft.get_theta()
		cmd = self.pitch_PID.update(current_value= diff)
		cmd = np.clip(a = cmd, a_min = -1, a_max= 1)
		self.elevator_cmd = cmd


class UAVPIDAutopilot:
	"""PID autopilot for UAV vehicles (AIM-7).

	Simpler than the aircraft autopilot: wings-level roll hold,
	pitch-to-altitude via arctan, heading correction via rudder PID.
	Enforces an initial acceleration stage where heading/altitude are frozen.
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
		"""Compute fin commands from guidance heading and altitude commands.

		During the acceleration stage (first N seconds after launch),
		heading and altitude commands are overridden to hold current values,
		allowing the motor to build speed before maneuvering.
		"""
		if not self.acceleration_stage_done():
			heading_cmd = self.UAV.get_psi()
			altitude_cmd = self.UAV.get_altitude()

		self.aileron_cmd = self.set_roll_PID(roll_ref= 0.0)

		self.elevator_cmd = self.set_altitude_PID(ref= altitude_cmd)

		self.rudder_cmd = self.set_heading_PID(head_ref= heading_cmd)

		self.throttle_cmd = self.set_throttle()

		return self.aileron_cmd, self.elevator_cmd, self.rudder_cmd	, self.throttle_cmd


	def set_roll_PID(self, roll_ref):
		"""Compute aileron command to hold wings level (roll_ref=0)."""
		roll_ref = np.clip(roll_ref, self.UAV_limits.phi_min, self.UAV_limits.phi_max)
		diff = roll_circle_clip(roll_ref- self.UAV.get_phi())
		cmd = -self.roll_PID.update(current_value=diff)

		return np.clip(a = cmd, a_min = -1, a_max= 1)

	def set_pitch_PID(self, theta_ref):
		"""Compute elevator command from pitch angle error."""
		diff = theta_ref - self.UAV.get_theta()
		pitch_cmd = self.pitch_PID.update(current_value= diff)
		pitch_cmd =  np.clip(pitch_cmd, -1, 1)
		return -pitch_cmd

	def set_heading_PID(self, head_ref):
		"""Compute rudder command from heading error."""
		diff = delta_heading(head_ref, self.UAV.get_psi() )
		cmd = self.heading_PID.update(current_value= diff)
		return np.clip(cmd, -1, 1 )

	def set_throttle(self,cmd = 0.7):
		return cmd

	def set_altitude_PID(self, ref):
		"""Convert altitude error to a pitch reference, then compute elevator.

		Uses arctan(alt_error / tan_ref) to map altitude error to a smooth
		pitch angle reference, clamped to [theta_min, theta_max].
		"""
		diff_atl = ref - self.UAV.get_altitude()

		theta_ref = np.degrees(np.arctan2(diff_atl, self.UAV_navigation.tan_ref))

		theta_ref = np.clip(a = theta_ref, a_min = self.UAV_navigation.theta_min, a_max = self.UAV_navigation.theta_max)

		return self.set_pitch_PID(theta_ref)


	def acceleration_stage_done(self):
		"""Check if the initial motor acceleration phase is complete."""
		if self.UAV.get_sim_time_sec() < self.UAV_navigation.acceleration_stage_in_sec:
			return False
		else:
			return True
