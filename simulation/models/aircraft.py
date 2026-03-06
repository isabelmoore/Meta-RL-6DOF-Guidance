from simulation.core.autopilot import AircraftPIDAutopilot
from simulation.models.fdm_object import FDMObject
from simulation.core.navigation import delta_heading
import time

class Aircraft(FDMObject):
    def __init__(self, conf, autopilot_cls=None):
        super().__init__(conf)
        self.conf = conf
        
        AutopilotClass = autopilot_cls if autopilot_cls else AircraftPIDAutopilot
        self.aircraft_control = AutopilotClass(self)
        
        self.aircraft_simulation_config = self.conf.aircraft_simulation
        

    
    def set_target(self, target):
        ''' 
        Set target aircraft object 
        Input: simObject (agent/UAV)
        '''
        self.target = target

    def step(self, action):
        self.set_retract_gear()
        for _ in range(self.aircraft_simulation_config.Sim_time_step):
            
            self._step(action)


    def _step(self, action):

        # heading betwen 0-360 deg
        # altitude in meters
        # throttle between 0-1 
        action_heading = action[0]
        action_altitude = action[1]
        action_throttle = action[2]

        diff_head = delta_heading(action_heading, self.get_psi() )
        diff_alt = action_altitude - self.get_altitude()

        for _ in range(self.aircraft_simulation_config.Control_time_step):
            # Calculate errors for PID
            diff_head = delta_heading(action_heading, self.get_psi())
            diff_alt = action_altitude - self.get_altitude()
            
            aileron_cmd, elevator_cmd, rudder_cmd = self.aircraft_control.get_control_input(diff_head, diff_alt)
            self.command_aircraft(aileron_cmd, elevator_cmd, rudder_cmd, action_throttle)    
            self.fdm.run()
            if self.conf.fg_sleep_time is not None:
                time.sleep(self.conf.fg_sleep_time)


    def get_autopilot_control(self, heading_cmd, alt_cmd):
        """
        Unified interface for autopilot control.
        Calculates errors and calls the aircraft PID.
        Returns 4 values (ail, elev, rud, thr).
        """
        diff_head = delta_heading(heading_cmd, self.get_psi())
        diff_alt = alt_cmd - self.get_altitude()
        
        ail, elev, rud = self.aircraft_control.get_control_input(diff_head, diff_alt)
        # Default throttle for autopilot control (can be improved)
        thr = 0.5 
        
        return ail, elev, rud, thr

    def command_flight(self, aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd):
        """Unified interface for commanding the flight dynamics."""
        self.command_aircraft(aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd)

    def command_aircraft(self, aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd):

        self.set_aileron(aileron_cmd)
        self.set_elevator(elevator_cmd)
        self.set_rudder(rudder_cmd)
        # todo velocity control. for now set 0.49 throttle at the imput 
        self.set_throttle(throttle_cmd)

