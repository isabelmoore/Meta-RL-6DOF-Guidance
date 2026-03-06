# Copyright (c) 2026 Isabel Moore. All rights reserved.
import numpy as np
import pymap3d as pm
from simulation.core.geospatial import angle_between


class GuidanceBase:
    """Shared helpers for all guidance laws."""

    def __init__(self, conf):
        self.N = conf.UAV_navigation.N
        self.dt = conf.UAV_navigation.dt
        self.distance_to_target = None

    def get_target_ENU(self, UAV):
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
        east, north, up = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
        return np.array([east, north, up])

    def get_target_v_ENU(self, UAV):
        try:
            v_east = UAV.target.simObj.get_v_east()
            v_north = UAV.target.simObj.get_v_north()
            v_up = -UAV.target.simObj.get_v_down()
        except AttributeError:
            v_east = UAV.target.get_v_east()
            v_north = UAV.target.get_v_north()
            v_up = -UAV.target.get_v_down()
        return np.array([v_east, v_north, v_up])

    def get_v_ENU(self, UAV):
        v_east = UAV.get_v_east()
        v_north = UAV.get_v_north()
        v_up = -UAV.get_v_down()
        return np.array([v_east, v_north, v_up])

    def get_heading_rel_direction(self, v1, v2):
        if np.cross(v1, v2)[2] < 0:
            return 1
        return -1

    def get_target_altitude(self, UAV):
        try:
            alt = UAV.target.simObj.get_altitude()
        except AttributeError:
            alt = UAV.target.get_altitude()
        return alt

    def _acc_to_heading_alt(self, UAV, acc_cmd_ENU, v_ENU, target_v_ENU, taget_ENU):
        """Convert an acceleration command in ENU to heading + altitude commands."""
        v_ENU_cmd = v_ENU + acc_cmd_ENU * self.dt

        v1 = v_ENU.copy()
        v2 = v_ENU_cmd.copy()
        v1[2] = 0.0
        v2[2] = 0.0

        heading_angle = angle_between(v1, v2, in_deg=True)
        hrd = self.get_heading_rel_direction(v1, v2)
        heading_cmd = (UAV.get_psi() + hrd * heading_angle) % 360

        v_rel_ENU = target_v_ENU - v_ENU
        v_rel_norm = np.linalg.norm(v_rel_ENU)
        r_norm = np.linalg.norm(taget_ENU)
        self.distance_to_target = r_norm
        tgo = r_norm / max(v_rel_norm, 1.0)
        altitude_cmd = self.get_target_altitude(UAV) + target_v_ENU[2] * tgo

        return heading_cmd, altitude_cmd


class PN(GuidanceBase):
    """Proportional Navigation (PN).

    Classic PN: a_cmd = N * (V_rel x omega), where omega is the LOS rotation rate.
    """

    def get_guidance(self, UAV):
        taget_ENU = self.get_target_ENU(UAV)
        target_v_ENU = self.get_target_v_ENU(UAV)
        v_ENU = self.get_v_ENU(UAV)
        v_rel_ENU = target_v_ENU - v_ENU

        rotation_vector = np.cross(taget_ENU, v_rel_ENU) / (taget_ENU @ taget_ENU)
        acc_cmd_ENU = self.N * np.cross(v_rel_ENU, rotation_vector)

        return self._acc_to_heading_alt(UAV, acc_cmd_ENU, v_ENU, target_v_ENU, taget_ENU)


class APN(GuidanceBase):
    """Augmented Proportional Navigation (APN).

    Adds a target acceleration compensation term:
        a_cmd = N * (V_rel x omega) + (N/2) * a_target_normal
    Improves performance against maneuvering targets.
    """

    def __init__(self, conf):
        super().__init__(conf)
        self._prev_target_v = None

    def get_guidance(self, UAV):
        taget_ENU = self.get_target_ENU(UAV)
        target_v_ENU = self.get_target_v_ENU(UAV)
        v_ENU = self.get_v_ENU(UAV)
        v_rel_ENU = target_v_ENU - v_ENU

        # Estimate target acceleration from velocity difference
        if self._prev_target_v is not None:
            a_target = (target_v_ENU - self._prev_target_v) / self.dt
        else:
            a_target = np.zeros(3)
        self._prev_target_v = target_v_ENU.copy()

        # Normal component of target acceleration (perpendicular to LOS)
        r_norm = np.linalg.norm(taget_ENU)
        if r_norm > 1.0:
            r_hat = taget_ENU / r_norm
            a_target_normal = a_target - np.dot(a_target, r_hat) * r_hat
        else:
            a_target_normal = np.zeros(3)

        # PN term
        rotation_vector = np.cross(taget_ENU, v_rel_ENU) / (taget_ENU @ taget_ENU)
        acc_pn = self.N * np.cross(v_rel_ENU, rotation_vector)

        # APN: add half-N times target normal acceleration
        acc_cmd_ENU = acc_pn + (self.N / 2.0) * a_target_normal

        return self._acc_to_heading_alt(UAV, acc_cmd_ENU, v_ENU, target_v_ENU, taget_ENU)


class ZEM(GuidanceBase):
    """Zero Effort Miss (ZEM) Guidance.

    Computes the predicted miss vector assuming no further control,
    then applies acceleration to cancel it:
        a_cmd = N * ZEM / t_go^2
    Optimal for constant-velocity targets.
    """

    def get_guidance(self, UAV):
        taget_ENU = self.get_target_ENU(UAV)
        target_v_ENU = self.get_target_v_ENU(UAV)
        v_ENU = self.get_v_ENU(UAV)
        v_rel_ENU = target_v_ENU - v_ENU

        v_rel_norm = np.linalg.norm(v_rel_ENU)
        r_norm = np.linalg.norm(taget_ENU)
        self.distance_to_target = r_norm
        tgo = max(r_norm / max(v_rel_norm, 1.0), 0.01)

        # Zero effort miss: where would we miss if we did nothing?
        zem = taget_ENU + v_rel_ENU * tgo

        # Acceleration to cancel the miss
        acc_cmd_ENU = self.N * zem / (tgo * tgo)

        return self._acc_to_heading_alt(UAV, acc_cmd_ENU, v_ENU, target_v_ENU, taget_ENU)


class PurePursuit(GuidanceBase):
    """Pure Pursuit Guidance.

    Always steer directly toward the target's current position.
    Simple but suboptimal — leads to tail-chase trajectories.
    """

    def get_guidance(self, UAV):
        taget_ENU = self.get_target_ENU(UAV)
        target_v_ENU = self.get_target_v_ENU(UAV)
        v_ENU = self.get_v_ENU(UAV)

        r_norm = np.linalg.norm(taget_ENU)
        self.distance_to_target = r_norm

        # Desired velocity: fly directly at target at current speed
        speed = np.linalg.norm(v_ENU)
        if r_norm > 1.0:
            v_desired = (taget_ENU / r_norm) * speed
        else:
            v_desired = v_ENU

        # Acceleration to turn toward target
        acc_cmd_ENU = (v_desired - v_ENU) / self.dt

        return self._acc_to_heading_alt(UAV, acc_cmd_ENU, v_ENU, target_v_ENU, taget_ENU)


# Registry for easy lookup by name
GUIDANCE_LAWS = {
    "pro_nav": PN,
    "PN": PN,
    "APN": APN,
    "augmented_pro_nav": APN,
    "ZEM": ZEM,
    "zero_effort_miss": ZEM,
    "pure_pursuit": PurePursuit,
}
