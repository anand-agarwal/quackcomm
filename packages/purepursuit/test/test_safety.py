import unittest

from pp_core.safety import StopLatch, evaluate_safety
from pp_core.topics import EMERGENCY_STOP_SUFFIX, FINAL_COMMAND_SUFFIX, WHEELS_COMMAND_SUFFIX


def params():
    return {
        "max_speed": 0.10,
        "stop_distance_m": 0.22,
        "clear_distance_m": 0.32,
        "brake_distance_m": 0.70,
        "brake_decel_m_s2": 0.40,
        "uncertain_speed_m_s": 0.04,
        "yolo_unavailable_speed_scale": 0.5,
        "tof_unavailable_speed_scale": 0.4,
        "require_fsm": True,
        "allowed_fsm_states": ["LANE_FOLLOWING"],
    }


def clear_path(**overrides):
    values = {
        "allow_motion": True,
        "startup_active": False,
        "fsm_state": "LANE_FOLLOWING",
        "desired_v": 0.10,
        "desired_omega": 0.20,
        "desired_fresh": True,
        "tof_valid": True,
        "tof_stale": False,
        "tof_range_m": 1.0,
        "yolo_fresh": True,
        "yolo_model_ok": True,
        "camera_ok": True,
        "obstacles": [],
    }
    values.update(overrides)
    return values


def corridor_object(distance, confirmed=True, projection_valid=True, in_corridor=True):
    return {
        "in_corridor": in_corridor,
        "distance_m": distance,
        "projection_valid": projection_valid,
        "confirmed": confirmed,
    }


class SafetyTests(unittest.TestCase):
    def test_clear_path_follows(self):
        decision = evaluate_safety(clear_path(), params())
        self.assertEqual(decision["state"], "CLEAR")
        self.assertAlmostEqual(decision["v"], 0.10)
        self.assertAlmostEqual(decision["omega"], 0.20)

    def test_object_outside_corridor_continues(self):
        decision = evaluate_safety(
            clear_path(obstacles=[corridor_object(0.15, in_corridor=False)]), params(),
        )
        self.assertEqual(decision["state"], "CLEAR")
        self.assertAlmostEqual(decision["v"], 0.10)

    def test_unconfirmed_detection_is_ignored(self):
        decision = evaluate_safety(
            clear_path(obstacles=[corridor_object(0.10, confirmed=False)]), params(),
        )
        self.assertEqual(decision["state"], "CLEAR")

    def test_object_in_corridor_and_tof_close_stops(self):
        decision = evaluate_safety(
            clear_path(tof_range_m=0.12, obstacles=[corridor_object(0.40)]), params(),
        )
        self.assertEqual(decision["state"], "STOP")
        self.assertEqual(decision["v"], 0.0)
        self.assertIn("ToF", decision["reason"])

    def test_object_in_corridor_far_can_move(self):
        decision = evaluate_safety(
            clear_path(obstacles=[corridor_object(1.0)]), params(),
        )
        self.assertEqual(decision["state"], "CLEAR")

    def test_object_in_corridor_brake_region_slows(self):
        decision = evaluate_safety(
            clear_path(obstacles=[corridor_object(0.40)]), params(),
        )
        self.assertEqual(decision["state"], "BRAKE")
        self.assertGreater(decision["v"], 0.0)
        self.assertLess(decision["v"], 0.10)

    def test_no_yolo_object_but_tof_close_stops(self):
        decision = evaluate_safety(clear_path(tof_range_m=0.10), params())
        self.assertEqual(decision["state"], "STOP")
        self.assertEqual(decision["omega"], 0.0)

    def test_tof_stale_and_corridor_object_stops(self):
        decision = evaluate_safety(
            clear_path(tof_valid=False, tof_stale=True, tof_range_m=None, obstacles=[corridor_object(0.50)]),
            params(),
        )
        self.assertEqual(decision["state"], "STOP")

    def test_both_sensors_unavailable_fails_safe(self):
        decision = evaluate_safety(
            clear_path(yolo_fresh=False, camera_ok=False, tof_valid=False, tof_stale=True, tof_range_m=None),
            params(),
        )
        self.assertEqual(decision["state"], "FAILSAFE")
        self.assertEqual(decision["v"], 0.0)

    def test_camera_disconnect_slows_when_tof_is_clear(self):
        decision = evaluate_safety(clear_path(camera_ok=False), params())
        self.assertEqual(decision["state"], "CLEAR")
        self.assertAlmostEqual(decision["v"], 0.05)

    def test_yolo_crash_and_clear_tof_slows(self):
        decision = evaluate_safety(clear_path(yolo_fresh=False, yolo_model_ok=False), params())
        self.assertAlmostEqual(decision["v"], 0.05)

    def test_invalid_tof_with_healthy_yolo_slows(self):
        decision = evaluate_safety(
            clear_path(tof_valid=False, tof_stale=True, tof_range_m=None), params(),
        )
        self.assertEqual(decision["state"], "CLEAR")
        self.assertAlmostEqual(decision["v"], 0.04)

    def test_multiple_obstacles_use_the_nearest_in_corridor(self):
        decision = evaluate_safety(
            clear_path(obstacles=[
                corridor_object(1.0),
                corridor_object(0.10),
                corridor_object(0.15, in_corridor=False),
            ]),
            params(),
        )
        self.assertEqual(decision["state"], "STOP")

    def test_stop_does_not_pass_the_desired_command_through(self):
        decision = evaluate_safety(clear_path(tof_range_m=0.05), params())
        self.assertGreater(decision["desired_v"], 0.0)
        self.assertEqual(decision["v"], 0.0)
        self.assertEqual(decision["omega"], 0.0)

    def test_rotation_in_place_passes_only_when_clear(self):
        # The controller's lane search rotates in place (v = 0).
        spin = dict(desired_v=0.0, desired_omega=2.0)
        clear = evaluate_safety(clear_path(**spin), params())
        self.assertEqual((clear["v"], clear["omega"]), (0.0, 2.0))
        braking = evaluate_safety(clear_path(tof_range_m=0.28, **spin), params())
        self.assertEqual(braking["omega"], 0.0)
        stopped = evaluate_safety(clear_path(tof_range_m=0.05, **spin), params())
        self.assertEqual(stopped["omega"], 0.0)
        held = evaluate_safety(clear_path(startup_active=True, **spin), params())
        self.assertEqual(held["omega"], 0.0)

    def test_motion_disabled_and_startup_and_fsm(self):
        self.assertEqual(evaluate_safety(clear_path(allow_motion=False), params())["reason"], "motion disabled")
        self.assertEqual(evaluate_safety(clear_path(startup_active=True), params())["reason"], "startup hold")
        self.assertIn("JOYSTICK", evaluate_safety(clear_path(fsm_state="NORMAL_JOYSTICK_CONTROL"), params())["reason"])
        self.assertEqual(evaluate_safety(clear_path(desired_fresh=False), params())["v"], 0.0)

    def test_clear_path_hysteresis_then_recovery(self):
        latch = StopLatch()
        stopped = evaluate_safety(clear_path(tof_range_m=0.10), params())
        held = latch.apply(0.0, stopped, 0.45)
        self.assertEqual(held["state"], "STOP")
        clear = evaluate_safety(clear_path(), params())
        still = latch.apply(0.20, clear, 0.45)
        self.assertEqual(still["state"], "STOP")
        self.assertEqual(still["reason"], "clear-path hysteresis")
        released = latch.apply(0.70, clear, 0.45)
        self.assertEqual(released["state"], "CLEAR")
        self.assertAlmostEqual(released["v"], 0.10)

    def test_final_command_is_not_the_wheel_interface(self):
        self.assertNotEqual(FINAL_COMMAND_SUFFIX, WHEELS_COMMAND_SUFFIX)
        self.assertNotIn("wheels_cmd", FINAL_COMMAND_SUFFIX)
        self.assertNotEqual(FINAL_COMMAND_SUFFIX, EMERGENCY_STOP_SUFFIX)


if __name__ == "__main__":
    unittest.main()
