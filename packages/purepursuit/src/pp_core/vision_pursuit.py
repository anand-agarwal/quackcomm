"""Vision lane following without ROS: tracker, arc Pure Pursuit, lane-loss recovery.

purepursuit_controller_node feeds it lane estimates and measured velocity and
asks it for a command each control step; the closed-loop tests drive the same
object through simulated turns. Times are in seconds (any common clock).
"""

import threading

from pp_core.lane_tracking import LaneTracker, VelocityHistory
from pp_core.pure_pursuit import adaptive_lookahead, command_from_lane_state, turn_lookahead
from pp_core.recovery import LaneLossRecovery


class VisionPursuit(object):
    def __init__(self, limits, lookahead=None, tracker=None, recovery=None,
                 vision_timeout_s=0.5, vision_min_confidence=0.2,
                 turn_radius_factor=1.2, turn_lookahead_min_m=0.12, coast_speed_scale=1.0,
                 velocity_timeout_s=0.5):
        self.limits = dict(limits)
        self.lookahead = dict(lookahead or {})
        self.tracker = tracker or LaneTracker()
        self.recovery = recovery or LaneLossRecovery()
        self.vision_timeout_s = float(vision_timeout_s)
        self.vision_min_confidence = float(vision_min_confidence)
        self.turn_radius_factor = float(turn_radius_factor)
        self.turn_lookahead_min_m = float(turn_lookahead_min_m)
        self.coast_speed_scale = float(coast_speed_scale)
        self.velocity_timeout_s = float(velocity_timeout_s)
        self.velocity = VelocityHistory()
        self.last_velocity_time = None
        self.last_vision_time = None
        self.last_confidence = 0.0
        self.last_state = None
        self.last_v = 0.0
        self.last_omega = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ inputs
    def on_velocity(self, stamp, now, v, omega):
        """Executed (v, omega), e.g. from kinematics_node."""
        with self._lock:
            self.velocity.add(stamp, v, omega)
            self.last_velocity_time = now

    def _velocity_source(self, now):
        """Measured velocity if it arrives, else our own last command."""
        if self.last_velocity_time is None or now - self.last_velocity_time > self.velocity_timeout_s:
            self.velocity.add(now, self.last_v, self.last_omega)
        return self.velocity

    def on_estimate(self, stamp, now, valid, d, phi, curvature, confidence):
        """A lane estimate for an image taken at `stamp`. Returns True if used."""
        if not valid or confidence < self.vision_min_confidence:
            return False
        with self._lock:
            self.tracker.update(stamp, d, phi, curvature, confidence, self._velocity_source(now))
            self.last_vision_time = now
            self.last_confidence = float(confidence)
        return True

    # ------------------------------------------------------------------ control
    def _state(self, now, max_age_s=None):
        with self._lock:
            return self.tracker.state_at(now, self._velocity_source(now), max_age_s)

    def command_from_state(self, state):
        d, phi, curvature = state
        lookahead = turn_lookahead(
            adaptive_lookahead(self.last_v, **self.lookahead), curvature,
            self.turn_radius_factor, self.turn_lookahead_min_m,
        )
        return command_from_lane_state(d, phi, curvature, lookahead, self.limits)

    def vision_command(self, now):
        """(command, confidence) from a fresh estimate, else (None, 0)."""
        if self.last_vision_time is None or now - self.last_vision_time > self.vision_timeout_s:
            return None, 0.0
        state = self._state(now)
        if state is None:
            return None, 0.0
        self.last_state = state
        return self.command_from_state(state), self.last_confidence

    def tracking(self, dt, command):
        """Record that `command` (from vision_command) is being driven."""
        if self.last_state is not None:
            self.recovery.on_tracking(dt, self.last_state[2], command["omega"], command["lateral_m"])

    def recover(self, now, dt):
        """No lane: (v, omega, command_or_None, status) from the recovery state machine."""
        status, manual = self.recovery.on_lost(now, dt, self.tracker.initialized)
        if manual is None:
            # Coast or turn memory: follow the tracker's prediction, which
            # carries the last curvature, so a turn is continued.
            predicted = self._state(now, self.recovery.coast_limit_s + self.vision_timeout_s)
            if predicted is not None:
                command = self.command_from_state(predicted)
                scale = self.coast_speed_scale
                return command["v"] * scale, command["omega"] * scale, command, status
            manual = (0.0, 0.0)
        return manual[0], manual[1], None, status

    def commanded(self, v, omega):
        """The command actually published (velocity fallback for the tracker)."""
        self.last_v, self.last_omega = float(v), float(omega)
