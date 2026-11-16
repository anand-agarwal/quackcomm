import unittest

from pp_core.geometry import ground_to_pixel, project_pixel_to_ground
from pp_core.localize import localize_detection
from pp_core.pure_pursuit import (
    WHITE,
    YELLOW,
    centerline_from_segments,
    command_from_target,
    point_at_lookahead,
    pursuit_command,
    target_from_lane_pose,
)


LIMITS = {
    "max_speed": 0.10,
    "min_speed": 0.035,
    "max_omega": 1.8,
    "max_lateral_accel": 0.18,
    "curvature_slowdown_gain": 0.65,
}

CALIBRATION = {
    "homography": None,
    "fx": 300.0,
    "fy": 300.0,
    "cx": 320.0,
    "cy": 240.0,
    "height_m": 0.10,
    "pitch_deg": 15.0,
    "cam_x_m": 0.0,
    "max_distance_m": 2.5,
}

CORRIDOR = {
    "x_min_m": 0.05,
    "x_max_m": 1.2,
    "half_width_m": 0.12,
    "uncertain_center_fraction": 0.35,
    "tof_gate_m": 0.08,
}


class PursuitTests(unittest.TestCase):
    def test_left_target_turns_left(self):
        command = pursuit_command(0.40, 0.10, 0.10, 0.035, 1.8, 0.18, 0.65)
        self.assertGreater(command["omega"], 0.0)
        self.assertGreater(command["curvature"], 0.0)

    def test_right_target_turns_right(self):
        command = pursuit_command(0.40, -0.10, 0.10, 0.035, 1.8, 0.18, 0.65)
        self.assertLess(command["omega"], 0.0)

    def test_speed_stays_inside_limits(self):
        command = pursuit_command(0.30, 0.25, 0.10, 0.035, 1.8, 0.18, 0.65)
        self.assertLessEqual(command["v"], 0.10)
        self.assertGreaterEqual(command["v"], 0.035)
        self.assertLessEqual(abs(command["omega"]), 1.8)

    def test_white_and_yellow_segments_form_a_centreline(self):
        segments = [
            (YELLOW, [(0.2, 0.12), (0.4, 0.12), (0.6, 0.12)]),
            (WHITE, [(0.2, -0.11), (0.4, -0.11), (0.6, -0.11)]),
        ]
        centers = centerline_from_segments(segments, lane_width_m=0.23)
        target = point_at_lookahead(centers, 0.42)
        command = command_from_target(target, LIMITS)
        self.assertIsNotNone(command)
        self.assertAlmostEqual(command["lateral_m"], 0.005, delta=0.03)

    def test_far_lane_white_line_is_ignored(self):
        # The opposite lane's outer white line sits left of the yellow line.
        segments = [
            (YELLOW, [(0.2, 0.12), (0.4, 0.12), (0.6, 0.12)]),
            (WHITE, [(0.2, -0.11), (0.4, -0.11), (0.6, -0.11)]),
            (WHITE, [(0.2, 0.35), (0.4, 0.35), (0.6, 0.35)]),
        ]
        centers = centerline_from_segments(segments, lane_width_m=0.23)
        for _, center_y in centers:
            self.assertAlmostEqual(center_y, 0.005, delta=0.01)

    def test_positive_lane_pose_d_steers_right(self):
        target = target_from_lane_pose(0.10, 0.0, 0.40)
        command = command_from_target(target, LIMITS)
        self.assertLess(command["omega"], 0.0)


class LocalizationTests(unittest.TestCase):
    def test_principal_point_lands_ahead_on_the_centreline(self):
        point = project_pixel_to_ground(320, 240, 300, 300, 320, 240, 0.10, 15.0)
        self.assertIsNotNone(point)
        self.assertGreater(point[0], 0.2)
        self.assertAlmostEqual(point[1], 0.0, places=3)

    def test_projection_round_trip(self):
        point = project_pixel_to_ground(360, 300, 300, 300, 320, 240, 0.10, 15.0)
        pixel = ground_to_pixel(point[0], point[1], 300, 300, 320, 240, 0.10, 15.0)
        self.assertAlmostEqual(pixel[0], 360.0, delta=0.2)
        self.assertAlmostEqual(pixel[1], 300.0, delta=0.2)

    def test_obstacle_inside_and_outside_the_corridor(self):
        inside = localize_detection(
            {"x_min": 300, "y_min": 200, "x_max": 340, "y_max": 280, "class_name": "person",
             "class_id": 0, "confidence": 0.9, "confirmed": True},
            640, 480, CALIBRATION, CORRIDOR, None,
        )
        outside = localize_detection(
            {"x_min": 0, "y_min": 200, "x_max": 40, "y_max": 280, "class_name": "person",
             "class_id": 0, "confidence": 0.9, "confirmed": True},
            640, 480, CALIBRATION, CORRIDOR, None,
        )
        self.assertTrue(inside["projection_valid"])
        self.assertTrue(inside["in_corridor"])
        self.assertTrue(outside["projection_valid"])
        self.assertFalse(outside["in_corridor"])

    def test_failed_projection_in_the_image_centre_is_uncertain(self):
        # A pixel above the horizon does not meet the ground.
        detection = localize_detection(
            {"x_min": 300, "y_min": 10, "x_max": 340, "y_max": 40, "class_name": "car",
             "class_id": 2, "confidence": 0.9, "confirmed": True},
            640, 480, CALIBRATION, CORRIDOR,
            {"available": True, "range_m": 0.18},
        )
        self.assertFalse(detection["projection_valid"])
        self.assertTrue(detection["in_corridor"])
        self.assertTrue(detection["distance_from_tof"])
        self.assertAlmostEqual(detection["distance_m"], 0.18)

    def test_tof_does_not_replace_a_closer_side_projection(self):
        detection = localize_detection(
            {"x_min": 0, "y_min": 220, "x_max": 30, "y_max": 300, "class_name": "car",
             "class_id": 2, "confidence": 0.9, "confirmed": True},
            640, 480, CALIBRATION, CORRIDOR,
            {"available": True, "range_m": 0.90},
        )
        self.assertFalse(detection["in_corridor"])
        self.assertFalse(detection["distance_from_tof"])


if __name__ == "__main__":
    unittest.main()
