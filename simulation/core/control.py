
class PID:
	"""
	Discrete PID control
	"""

	def __init__(self, PID_gains):

		self.Kp=PID_gains.P
		self.Ki=PID_gains.I
		self.Kd=PID_gains.D
		self.Derivator=PID_gains.Deriv
		self.Integrator=PID_gains.Integ
		self.Integrator_max=PID_gains.Integ_max
		self.Integrator_min=PID_gains.Integ_min

		self.set_point=0.0
		self.error=0.0

	def update(self,current_value):
		"""
		Calculate PID output value for given reference input and feedback
		"""

		self.error = self.set_point - current_value

		self.P_value = self.Kp * self.error
		self.D_value = self.Kd * ( self.error - self.Derivator)
		self.Derivator = self.error

		self.Integrator = self.Integrator + self.error

		if self.Integrator > self.Integrator_max:
			self.Integrator = self.Integrator_max
		elif self.Integrator < self.Integrator_min:
			self.Integrator = self.Integrator_min

		self.I_value = self.Integrator * self.Ki

		PID = self.P_value + self.I_value + self.D_value

		return PID

	def setPoint(self,set_point):
		"""
		Initilize the setpoint of PID
		"""
		self.set_point = set_point
		self.Integrator=0
		self.Derivator=0

	def setIntegrator(self, Integrator):
		self.Integrator = Integrator

	def setDerivator(self, Derivator):
		self.Derivator = Derivator

	def setKp(self,P):
		self.Kp=P

	def setKi(self,I):
		self.Ki=I

	def setKd(self,D):
		self.Kd=D

	def getPoint(self):
		return self.set_point

	def getError(self):
		return self.error

	def getIntegrator(self):
		return self.Integrator

	def getDerivator(self):
		return self.Derivator




from typing import Tuple

class PID_discrete:
    def __init__(self,
                 Kp: float = 1.0,
                 Ki: float = 0.0,
                 Kd: float = 0.0,
                 setpoint: float = 0.0,
                 output_limits: Tuple[float, float] = (None, None),
                 sample_time: float = None,
                 tau: float = 0.02,  # derivative filter time constant (seconds)
                 beta: float = 1.0   # setpoint weighting for proportional term
                 ) -> None:
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.setpoint = float(setpoint)
        self.min_output, self.max_output = output_limits if output_limits is not None else (None, None)
        self.sample_time = sample_time  # not enforced, user may pass variable dt
        self.tau = float(tau) if tau is not None else 0.0
        self.beta = float(beta)

        # internal states
        self._integral = 0.0
        self._prev_measurement = None
        self._prev_filtered_deriv = 0.0
        self._last_output = 0.0

    def reset(self):
        """Reset internal state (integral, derivative history, last output)."""
        self._integral = 0.0
        self._prev_measurement = None
        self._prev_filtered_deriv = 0.0
        self._last_output = 0.0

    def _clamp(self, value: float) -> float:
        if (self.max_output is not None) and (value > self.max_output):
            return self.max_output
        if (self.min_output is not None) and (value < self.min_output):
            return self.min_output
        return value

    def update(self, measurement: float, dt: float) -> float:
        """
        Compute PID output given a measurement and time step dt (seconds).

        Args:
            measurement: current process variable (PV)
            dt: time step in seconds (must be > 0)

        Returns:
            control output (float) within output_limits if set.
        """
        if dt <= 0:
            raise ValueError("dt must be > 0")

        error = self.setpoint - measurement

        # Proportional term with setpoint weighting (beta)
        P = self.Kp * (self.beta * self.setpoint - measurement)

        # Derivative term using measurement derivative (reduces derivative kick from setpoint changes)
        if self._prev_measurement is None:
            deriv_meas = 0.0
        else:
            deriv_meas = (measurement - self._prev_measurement) / dt

        # first-order filter for derivative: filtered = (tau*prev + dt*deriv) / (tau + dt)
        if self.tau > 0.0:
            filtered = (self.tau * self._prev_filtered_deriv + dt * deriv_meas) / (self.tau + dt)
        else:
            filtered = deriv_meas

        D = -self.Kd * filtered  # negative sign because derivative on measurement

        # Tentative integral update (we'll do simple anti-windup after checking saturation)
        self._integral += error * dt
        I = self.Ki * self._integral

        # Unclamped output
        unclamped = P + I + D

        # Clamp the output
        output = self._clamp(unclamped)

        # Simple anti-windup: if we are saturated and the integral is driving us further into saturation,
        # then undo the last integration step (i.e. don't integrate while saturated unless integration helps)
        saturated_high = (self.max_output is not None) and (unclamped > self.max_output)
        saturated_low = (self.min_output is not None) and (unclamped < self.min_output)
        if (saturated_high and error > 0) or (saturated_low and error < 0):
            # revert the integration step we just applied
            self._integral -= error * dt
            I = self.Ki * self._integral
            unclamped = P + I + D
            output = self._clamp(unclamped)

        # update stored states for next call
        self._prev_measurement = measurement
        self._prev_filtered_deriv = filtered
        self._last_output = output

        return output

    def tunings(self):
        return self.Kp, self.Ki, self.Kd

    def set_output_limits(self, limits: Tuple[float, float]):
        self.min_output, self.max_output = limits

    def set_setpoint(self, sp: float):
        self.setpoint = float(sp)