import math
import unittest

from pp_core.recovery import (COAST, LOST, SEARCH, TRACKING, TURN_MEMORY, WAITING,
                              LaneLossRecovery)

DT = 0.05


def track(recovery, seconds, kappa=0.0, omega=0.0, lateral=0.0):
    for _ in range(int(round(seconds / DT))):
        recovery.on_tracking(DT, kappa, omega, lateral)


def lose(recovery, start, seconds, can_predict=True):
    """Run on_lost from `start` for `seconds`; return [(t, state, command)]."""
    out = []
    for i in range(int(round(seconds / DT))):
        t = start + i * DT
        state, command = recovery.on_lost(t, DT, can_predict)
        out.append((t, state, command))
    return out


class RecoveryTests(unittest.TestCase):
    def test_never_searches_before_a_lane_was_seen(self):
        recovery = LaneLossRecovery()
        states = {state for _, state, _ in lose(recovery, 0.0, 10.0)}
        self.assertEqual(states, {WAITING})

    def test_straight_coasts_then_searches_toward_the_last_side(self):
        recovery = LaneLossRecovery(coast_s=0.6)
        track(recovery, 1.0, lateral=-0.02)  # lane centre was to the right
        steps = lose(recovery, 10.0, 1.0)
        coast = [s for t, s, _ in steps if t - 10.0 <= 0.6]
        self.assertEqual(set(coast), {COAST})
        first_search = next(c for _, s, c in steps if s == SEARCH)
        self.assertEqual(first_search[0], 0.0)
        self.assertLess(first_search[1], 0.0)  # rotates right first

    def test_turn_memory_lasts_for_the_rest_of_the_turn(self):
        # Right turn at omega -0.55 rad/s; a quarter turn takes ~2.9 s.
        recovery = LaneLossRecovery(coast_s=0.6, max_turn_memory_s=3.0)
        track(recovery, 1.0, kappa=-5.5, omega=-0.55, lateral=-0.1)
        self.assertAlmostEqual(recovery.turned_rad, 0.55, delta=0.03)
        steps = lose(recovery, 10.0, 3.0)
        memory = [t - 10.0 for t, s, _ in steps if s == TURN_MEMORY]
        expected = (math.pi / 2 - 0.55) / 0.55  # ~1.86 s left of the turn
        self.assertAlmostEqual(max(memory), expected, delta=2 * DT)
        self.assertEqual(steps[-1][1], SEARCH)

    def test_turn_memory_is_capped(self):
        recovery = LaneLossRecovery(max_turn_memory_s=1.0)
        track(recovery, 0.1, kappa=-5.5, omega=-0.2)
        self.assertLessEqual(recovery.coast_limit_s if recovery.lost_since else 1.0, 1.0)
        steps = lose(recovery, 5.0, 2.0)
        memory = [t - 5.0 for t, s, _ in steps if s == TURN_MEMORY]
        self.assertLessEqual(max(memory), 1.0 + 1e-9)

    def test_search_sweeps_both_ways_with_growing_amplitude(self):
        recovery = LaneLossRecovery(coast_s=0.0, search_omega=2.0, search_angle_rad=0.6,
                                    search_max_angle_rad=1.5, search_timeout_s=10.0)
        track(recovery, 0.5, lateral=0.03)  # left
        steps = lose(recovery, 0.0, 5.0, can_predict=False)
        angle, extremes, direction = 0.0, [], None
        for _, state, command in steps:
            self.assertEqual(state, SEARCH)
            if direction is not None and math.copysign(1, command[1]) != direction:
                extremes.append(angle)
            direction = math.copysign(1, command[1])
            angle += command[1] * DT
        self.assertGreater(extremes[0], 0.55)          # left first, ~0.6 rad
        self.assertLess(extremes[1], -1.1)             # then right, ~1.2 rad
        self.assertLessEqual(max(abs(e) for e in extremes), 1.5 + 2 * 2.0 * DT)

    def test_search_gives_up_and_stops(self):
        recovery = LaneLossRecovery(coast_s=0.2, search_timeout_s=2.0)
        track(recovery, 0.5)
        steps = lose(recovery, 0.0, 4.0)
        self.assertEqual(steps[-1][1], LOST)
        self.assertEqual(steps[-1][2], (0.0, 0.0))

    def test_lane_found_again_resets(self):
        recovery = LaneLossRecovery(coast_s=0.2, search_timeout_s=1.0)
        track(recovery, 0.5)
        lose(recovery, 0.0, 3.0)
        self.assertEqual(recovery.state, LOST)
        track(recovery, 0.1)
        self.assertEqual(recovery.state, TRACKING)
        steps = lose(recovery, 10.0, 0.2)
        self.assertEqual(steps[0][1], COAST)

    def test_search_can_be_disabled(self):
        recovery = LaneLossRecovery(coast_s=0.2, search_enabled=False)
        track(recovery, 0.5)
        steps = lose(recovery, 0.0, 1.0)
        self.assertEqual(steps[-1][1], LOST)

    def test_creep_speed_is_used_while_searching(self):
        recovery = LaneLossRecovery(coast_s=0.0, search_forward_speed=0.05)
        track(recovery, 0.5)
        _, state, command = lose(recovery, 0.0, 0.1, can_predict=False)[-1]
        self.assertEqual(state, SEARCH)
        self.assertEqual(command[0], 0.05)


if __name__ == "__main__":
    unittest.main()
