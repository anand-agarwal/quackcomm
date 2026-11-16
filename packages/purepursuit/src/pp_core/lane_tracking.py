"""Kalman tracking of the lane pose with motion prediction and delay compensation.

State x = [d, phi, curvature] in the dt-core LanePose convention:
d > 0 robot left of the lane centre, phi > 0 robot yawed left of the lane
direction, curvature > 0 road turning left.

Motion model for a robot moving at forward speed v and yaw rate omega:

    d'         = v * sin(phi)
    phi'       = omega - curvature * v * cos(phi) / (1 - curvature * d)
    curvature' = 0   (random walk)

(Frenet equations relative to the lane-centre circle; 1 - curvature * d is
the ratio of the robot's turn radius to the centre's, which matters in a
0.18 m right turn.)

Camera frames are processed late. Each measurement is applied at its image
time, and the controller asks for the state *now*, which is predicted forward
from the last image time with the velocity history. That removes the pipeline
latency from the control loop instead of steering on a stale picture.
"""

import bisect
import math

import numpy as np


class VelocityHistory(object):
    """(time, v, omega) samples; piecewise-constant integration between them."""

    def __init__(self, horizon_s=3.0):
        self.horizon_s = float(horizon_s)
        self.times = []
        self.values = []

    def add(self, t, v, omega):
        t = float(t)
        if self.times and t < self.times[-1]:
            return
        self.times.append(t)
        self.values.append((float(v), float(omega)))
        cutoff = t - self.horizon_s
        while len(self.times) > 2 and self.times[1] < cutoff:
            self.times.pop(0)
            self.values.pop(0)

    def at(self, t):
        if not self.times:
            return 0.0, 0.0
        index = bisect.bisect_right(self.times, t) - 1
        return self.values[max(0, index)]

    def segments(self, t0, t1, max_step=0.05):
        """Yield (dt, v, omega) pieces covering [t0, t1]."""
        if t1 <= t0:
            return
        t = t0
        while t < t1 - 1e-9:
            index = bisect.bisect_right(self.times, t)
            next_change = self.times[index] if index < len(self.times) else t1
            t_next = min(t1, t + max_step, max(next_change, t + 1e-6))
            v, omega = self.at(t)
            yield t_next - t, v, omega
            t = t_next


def _predict(x, P, v, omega, dt, q):
    d, phi, kappa = x
    g = max(0.2, 1.0 - kappa * d)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    x_new = np.array([
        d + v * sin_phi * dt,
        phi + (omega - kappa * v * cos_phi / g) * dt,
        kappa,
    ])
    F = np.array([
        [1.0, v * cos_phi * dt, 0.0],
        [-kappa * kappa * v * cos_phi / (g * g) * dt, 1.0 + kappa * v * sin_phi / g * dt,
         -v * cos_phi / (g * g) * dt],
        [0.0, 0.0, 1.0],
    ])
    Q = np.diag(q) * dt
    return x_new, F @ P @ F.T + Q


class LaneTracker(object):
    def __init__(self, process_noise=(0.002, 0.05, 2.0), measurement_noise=(0.01, 0.05, 0.6),
                 reset_after_s=1.0, max_kappa=10.0):
        self.q = np.array(process_noise, dtype=np.float64) ** 2
        self.r = np.array(measurement_noise, dtype=np.float64) ** 2
        self.reset_after_s = float(reset_after_s)
        self.max_kappa = float(max_kappa)
        self.x = None
        self.P = None
        self.t = None

    @property
    def initialized(self):
        return self.x is not None

    def reset(self):
        self.x = self.P = self.t = None

    def _propagate(self, x, P, t0, t1, history):
        for dt, v, omega in history.segments(t0, t1):
            x, P = _predict(x, P, v, omega, dt, self.q)
        x[2] = float(np.clip(x[2], -self.max_kappa, self.max_kappa))
        return x, P

    def update(self, t, d, phi, curvature, confidence, history):
        """Apply a measurement taken at image time t."""
        z = np.array([d, phi, curvature], dtype=np.float64)
        confidence = max(0.05, min(1.0, float(confidence)))
        R = np.diag(self.r / confidence)
        if self.x is None or self.t is None or t - self.t > self.reset_after_s:
            self.x, self.P, self.t = z.copy(), R.copy(), float(t)
            self.x[2] = float(np.clip(self.x[2], -self.max_kappa, self.max_kappa))
            return
        if t <= self.t:
            return  # out-of-order frame
        x, P = self._propagate(self.x.copy(), self.P.copy(), self.t, t, history)
        innovation = z - x
        S = P + R
        K = P @ np.linalg.inv(S)
        self.x = x + K @ innovation
        self.P = (np.eye(3) - K) @ P
        self.x[2] = float(np.clip(self.x[2], -self.max_kappa, self.max_kappa))
        self.t = float(t)

    @property
    def last_update(self):
        return self.t

    def state_at(self, t, history, max_age_s=None):
        """Predicted (d, phi, curvature) at time t, or None if stale/uninitialized.

        max_age_s (default reset_after_s) is how far past the last measurement
        to predict. While the lane is lost the controller asks for longer, to
        keep driving the last seen turn on the motion model.
        """
        limit = self.reset_after_s if max_age_s is None else float(max_age_s)
        if self.x is None or t - self.t > limit:
            return None
        x, _ = self._propagate(self.x.copy(), self.P.copy(), self.t, max(t, self.t), history)
        return float(x[0]), float(x[1]), float(x[2])
