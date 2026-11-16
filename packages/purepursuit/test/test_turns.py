"""Closed loop through Duckietown turns: VisionPursuit (the controller node's
vision path) driving a simulated robot on a straight -> quarter turn -> straight
track, with delayed, noisy, low-rate lane estimates.

The perception model is what the arc lane detector reports (see
test_lane_vision.TurnPerceptionTests): the turn's circle once the turn is
about 0.12 m ahead, otherwise the straight. Its curvature is under-read by
15% (rendered right turns read -4.5..-5.1 for -5.56).
"""

import math
import unittest

import numpy as np

from pp_core.lane_tracking import LaneTracker
from pp_core.recovery import LaneLossRecovery
from pp_core.vision_pursuit import VisionPursuit

# config/controller.yaml
LIMITS = {"max_speed": 0.18, "min_speed": 0.10, "max_omega": 1.8,
          "max_lateral_accel": 0.18, "curvature_slowdown_gain": 0.65}
LOOKAHEAD = {"base_m": 0.20, "gain_s": 0.6, "min_m": 0.18, "max_m": 0.35}
CONTROL_DT = 0.05


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Track(object):
    """Straight of `before` m, a quarter turn of signed curvature kappa, straight of `after` m."""

    def __init__(self, kappa, before=0.6, after=0.8, ds=0.002):
        self.kappa = kappa
        self.arc_start = before
        self.arc_end = before + (math.pi / 2) / abs(kappa)
        self.length = self.arc_end + after
        self.s = np.arange(0.0, self.length, ds)
        self.k = np.where((self.s >= self.arc_start) & (self.s < self.arc_end), kappa, 0.0)
        self.heading = np.concatenate([[0.0], np.cumsum(self.k[:-1] * ds)])
        steps = np.c_[np.cos(self.heading), np.sin(self.heading)] * ds
        self.xy = np.concatenate([[[0.0, 0.0]], np.cumsum(steps[:-1], axis=0)])
        i = int(self.arc_start / ds)
        normal = np.array([-math.sin(self.heading[i]), math.cos(self.heading[i])])
        self.center = self.xy[i] + normal / kappa
        self.radius = 1.0 / abs(kappa)

    def nearest(self, p, hint=0):
        lo, hi = max(0, hint - 300), min(len(self.s), hint + 300)
        j = lo + int(np.argmin(np.hypot(*(self.xy[lo:hi] - p).T)))
        return j

    def frenet(self, p, theta, j):
        n = np.array([-math.sin(self.heading[j]), math.cos(self.heading[j])])
        return float((p - self.xy[j]) @ n), wrap(theta - self.heading[j])

    def perceived(self, p, theta, j, preview=0.12):
        """(d, phi, kappa) the arc detector reports from this pose."""
        s_here = self.s[j]
        in_view = self.arc_start - preview <= s_here < self.arc_end - 0.05
        if not in_view:
            d, phi = self.frenet(p, theta, j)
            return d, phi, 0.0
        u = p - self.center
        sign = math.copysign(1.0, self.kappa)
        d = sign * (self.radius - math.hypot(*u))
        tangent = math.atan2(u[1], u[0]) + sign * math.pi / 2
        return d, wrap(theta - tangent), self.kappa


def drive(kappa, seconds=25.0, dropout=None, seed=0, latency=0.25, rate_hz=5.0):
    """Run the loop; returns per-step records (t, s, d_true, status, v, omega)."""
    rng = np.random.default_rng(seed)
    track = Track(kappa)
    pursuit = VisionPursuit(
        LIMITS, LOOKAHEAD, LaneTracker(), LaneLossRecovery(), vision_timeout_s=0.5,
    )
    pose = np.array([0.0, 0.0]), 0.0
    p, theta = pose
    j = 0
    pending = []  # (deliver_at, stamp, measurement)
    next_capture = 0.0
    records = []
    t = 0.0
    while t < seconds:
        j = track.nearest(p, j)
        # Camera: capture now, deliver after the pipeline latency.
        if t >= next_capture:
            next_capture += 1.0 / rate_hz
            lost = dropout is not None and dropout[0] <= track.s[j] and t < dropout[1](t)
            if not lost:
                d, phi, k = track.perceived(p, theta, j)
                measurement = (d + rng.normal(0, 0.006), phi + rng.normal(0, 0.03),
                               0.85 * k + rng.normal(0, 0.4 if k else 0.1))
                pending.append((t + latency, t, measurement))
        while pending and pending[0][0] <= t:
            _, stamp, (d, phi, k) = pending.pop(0)
            pursuit.on_estimate(stamp, t, True, d, phi, k, 0.8)
        # Controller step, as purepursuit_controller_node._control.
        command, _ = pursuit.vision_command(t)
        if command is not None:
            pursuit.tracking(CONTROL_DT, command)
            v, omega, status = command["v"], command["omega"], "tracking"
        else:
            v, omega, _, status = pursuit.recover(t, CONTROL_DT)
        pursuit.commanded(v, omega)
        pursuit.on_velocity(t, t, v, omega)  # kinematics_node feedback
        # Robot moves (unicycle, exact).
        theta += omega * CONTROL_DT
        p = p + v * CONTROL_DT * np.array([math.cos(theta), math.sin(theta)])
        d_true, _ = track.frenet(p, theta, track.nearest(p, j))
        records.append((t, track.s[j], d_true, status, v, omega))
        t += CONTROL_DT
        if track.s[j] > track.length - 0.15:
            break
    return track, records


class ClosedLoopTurnTests(unittest.TestCase):
    def assert_follows(self, kappa, max_offset_m):
        for seed in range(3):
            track, records = drive(kappa, seed=seed)
            offsets = [abs(r[2]) for r in records]
            self.assertGreater(records[-1][1], track.arc_end + 0.5, "did not get through the turn")
            # Robot half width ~0.065 m in a 0.23 m lane: 0.05 m keeps the wheels inside.
            self.assertLess(max(offsets), max_offset_m, "seed %d" % seed)

    def test_right_turn(self):
        self.assert_follows(-1.0 / 0.18, 0.05)

    def test_left_turn(self):
        self.assert_follows(1.0 / 0.41, 0.05)

    def test_lane_lost_mid_turn_is_finished_from_memory(self):
        kappa = -1.0 / 0.18
        track = Track(kappa)
        mid = track.arc_start + 0.25 * (track.arc_end - track.arc_start)
        window = {}

        def until(t):  # blind for 1.2 s from the moment the robot reaches `mid`
            window.setdefault("start", t)
            return window["start"] + 1.2

        _, records = drive(kappa, dropout=(mid, until))
        statuses = {r[3] for r in records}
        self.assertIn("turn_memory", statuses)
        self.assertNotIn("search", statuses)
        self.assertGreater(records[-1][1], track.arc_end + 0.5)
        self.assertLess(max(abs(r[2]) for r in records), 0.06)

    def test_lane_gone_for_good_searches_then_stops(self):
        kappa = 1.0 / 0.41

        def forever(_t):
            return float("inf")

        _, records = drive(kappa, seconds=12.0, dropout=(0.3, forever))
        statuses = [r[3] for r in records]
        self.assertIn("coast", statuses)
        self.assertIn("search", statuses)
        self.assertEqual(statuses[-1], "lost")
        search = [r for r in records if r[3] == "search"]
        self.assertTrue(all(r[4] == 0.0 and abs(r[5]) > 0 for r in search))  # rotates in place
        self.assertEqual((records[-1][4], records[-1][5]), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
