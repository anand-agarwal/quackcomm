"""What the controller does while the lane is not seen.

States (also published as ControllerStatus.status):

- tracking     a fresh lane estimate: follow it.
- coast        lane lost on a straight: keep following the tracker's
               prediction of the lane for coast_s.
- turn_memory  lane lost inside a turn: keep following the prediction, which
               carries the turn's curvature, until the rest of the turn is
               driven (a Duckietown turn is a quarter circle), at most
               max_turn_memory_s.
- search       still lost: rotate toward the side the lane was last seen,
               sweeping back and forth with a growing amplitude (optionally
               creeping forward), until an estimate returns.
- lost         nothing within search_timeout_s: stop.
- waiting      no lane seen since start: stop (never search blind).

Pure Python, no ROS, so it can be unit tested. Time is in seconds.
"""

import math

TRACKING = "tracking"
COAST = "coast"
TURN_MEMORY = "turn_memory"
SEARCH = "search"
LOST = "lost"
WAITING = "waiting"


class LaneLossRecovery(object):
    def __init__(self, coast_s=0.6, turn_kappa=1.5, turn_angle_rad=math.pi / 2,
                 max_turn_memory_s=3.0, search_enabled=True, search_omega=2.0,
                 search_forward_speed=0.0, search_angle_rad=0.6, search_max_angle_rad=1.5,
                 search_timeout_s=6.0):
        self.coast_s = float(coast_s)
        self.turn_kappa = float(turn_kappa)
        self.turn_angle_rad = float(turn_angle_rad)
        self.max_turn_memory_s = float(max_turn_memory_s)
        self.search_enabled = bool(search_enabled)
        self.search_omega = abs(float(search_omega))
        self.search_forward_speed = max(0.0, float(search_forward_speed))
        self.search_angle_rad = abs(float(search_angle_rad))
        self.search_max_angle_rad = abs(float(search_max_angle_rad))
        self.search_timeout_s = float(search_timeout_s)
        self.state = WAITING
        self.ever_tracked = False
        self.last_kappa = 0.0
        self.last_omega = 0.0
        self.last_side = 1.0
        self.turned_rad = 0.0
        self.lost_since = None
        self.coast_limit_s = 0.0
        self.search_since = None
        self.search_dir = 1.0
        self.search_angle = 0.0
        self.search_amplitude = 0.0

    # ------------------------------------------------------------------
    def on_tracking(self, dt, kappa, omega, lateral_target):
        """Call each control step while a fresh lane estimate is followed."""
        self.state = TRACKING
        self.ever_tracked = True
        self.lost_since = None
        self.search_since = None
        kappa, omega = float(kappa), float(omega)
        in_turn = abs(kappa) >= self.turn_kappa
        if in_turn and kappa * omega > 0.0:
            self.turned_rad += abs(omega) * max(0.0, float(dt))
        elif abs(kappa) < 0.5 * self.turn_kappa:
            self.turned_rad = 0.0
        self.last_kappa, self.last_omega = kappa, omega
        if abs(lateral_target) > 0.01:
            self.last_side = math.copysign(1.0, lateral_target)
        elif abs(kappa) > 0.3:
            self.last_side = math.copysign(1.0, kappa)

    def in_turn(self):
        return abs(self.last_kappa) >= self.turn_kappa and abs(self.last_omega) > 0.05

    def remaining_turn_s(self):
        """Time to finish the current turn at the last turn rate."""
        remaining = max(0.0, self.turn_angle_rad - self.turned_rad)
        return remaining / max(abs(self.last_omega), 1e-3)

    # ------------------------------------------------------------------
    def on_lost(self, now, dt, can_predict):
        """Call each control step without a fresh lane estimate.

        Returns (state, command): command is None for coast/turn_memory (the
        caller follows the tracker's prediction), else (v, omega).
        """
        if not self.ever_tracked:
            self.state = WAITING
            return WAITING, (0.0, 0.0)
        if self.lost_since is None:
            self.lost_since = now
            if self.in_turn():
                self.coast_limit_s = min(self.max_turn_memory_s,
                                         max(self.coast_s, self.remaining_turn_s()))
            else:
                self.coast_limit_s = self.coast_s
        age = now - self.lost_since
        if can_predict and age <= self.coast_limit_s and self.state in (TRACKING, COAST, TURN_MEMORY):
            self.state = TURN_MEMORY if self.in_turn() else COAST
            return self.state, None
        if not self.search_enabled:
            self.state = LOST
            return LOST, (0.0, 0.0)
        if self.state != SEARCH and self.state != LOST:
            self.state = SEARCH
            self.search_since = now
            self.search_dir = self.last_side
            self.search_angle = 0.0
            self.search_amplitude = self.search_angle_rad
        if self.state == LOST or now - self.search_since > self.search_timeout_s:
            self.state = LOST
            return LOST, (0.0, 0.0)
        # Sweep: toward the last side to +amplitude, then across to the other
        # side, widening each time, up to search_max_angle_rad.
        # The angle counts the command issued this step; the turn-around applies
        # from the next step.
        direction = self.search_dir
        self.search_angle += direction * self.search_omega * max(0.0, float(dt))
        if direction * self.search_angle >= self.search_amplitude:
            self.search_dir = -direction
            self.search_amplitude = min(self.search_max_angle_rad,
                                        self.search_amplitude + self.search_angle_rad)
        return SEARCH, (self.search_forward_speed, direction * self.search_omega)
