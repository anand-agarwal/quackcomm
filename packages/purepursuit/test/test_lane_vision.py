import math
import unittest

import numpy as np

from pp_core.lane_tracking import LaneTracker, VelocityHistory
from pp_core.lane_vision import ArcFit, GroundModel, arc_through, detect_lane, estimate_lane, fit_line
from pp_core.pure_pursuit import (adaptive_lookahead, command_from_target, target_on_arc,
                                  target_on_lane_curve, turn_lookahead)

# Duckietown turns (lane centre): right ~0.18 m radius, left ~0.41 m.
RIGHT_TURN = -1.0 / 0.18
LEFT_TURN = 1.0 / 0.41

# myrobot2 calibration (2026-09-30), the same values dt-core loads.
K = [326.049927752541, 0.0, 275.3636982404864, 0.0, 339.8614106918087, 230.12051504962272, 0.0, 0.0, 1.0]
D = [-0.405873168602115, 0.24723420914482114, 0.0049496592668074576, 0.005268811004744025, -0.06338349055831163]
H = [0.018885912169966473, 0.11060945850334832, 0.4243052547946663, -0.347564228292063,
     -0.03899840454618963, 0.00863919347803072, 0.049071201545771964, 3.155988817717057, 1.0]
LANE_WIDTH = 0.23
LIMITS = {"max_speed": 0.18, "min_speed": 0.10, "max_omega": 1.8,
          "max_lateral_accel": 0.18, "curvature_slowdown_gain": 0.65}


def model():
    return GroundModel(K, D, H, 640, 480)


def centerline(d, phi, kappa, straight_before_m=0.0, turn_rad=math.pi / 2, ds=0.004):
    """Lane centre as on a real mat: straight, then an arc of `turn_rad`, then straight.

    (d, phi, kappa) is the robot's pose relative to the arc's circle (the
    estimator's convention); with straight_before_m > 0 the robot is on the
    straight that far before the arc starts. Returns (s, points, headings).
    """
    s = np.arange(-0.3, 1.2, ds)
    arc_len = turn_rad / abs(kappa) if kappa else np.inf
    in_arc = np.clip(s - straight_before_m, 0.0, arc_len)
    heading = -phi + kappa * in_arc
    step = np.c_[np.cos(heading), np.sin(heading)] * ds
    n0 = np.array([math.sin(phi), math.cos(phi)])  # left normal of the lane at the robot
    start = np.argmin(np.abs(s))
    points = np.cumsum(step, axis=0)
    points = points - points[start] - d * n0
    return s, points, heading


def render(d, phi, kappa, yellow=True, white=True, dashed_yellow=False, straight_before_m=0.0):
    """Draw the lane seen from a robot at (d, phi) on a road of curvature kappa.

    Lines are offset along the centre's normal, so they stay parallel in turns
    as painted lines do.
    """
    import cv2
    m = model()
    image = np.full((480, 640, 3), (45, 45, 45), dtype=np.uint8)
    s_all, centre, heading = centerline(d, phi, kappa, straight_before_m)
    normal = np.c_[-np.sin(heading), np.cos(heading)]
    xs = s_all

    def offset(dy):
        return centre + dy * normal

    lines = []
    if yellow:
        lines.append((LANE_WIDTH / 2, 0.012, (0, 200, 230), dashed_yellow))
    if white:
        lines.append((-LANE_WIDTH / 2, 0.024, (235, 235, 235), False))
    def valid(ground, px):
        back = m.pixels_to_ground(px)
        inside = (px[:, 0] > -50) & (px[:, 0] < 690) & (px[:, 1] > -50) & (px[:, 1] < 530)
        return inside & (np.linalg.norm(back - ground, axis=1) < 0.01) & (ground[:, 0] > 0.03)

    for line_offset, half_width, bgr, dashed in lines:
        # Each stretch of line is a polygon between its two projected edges.
        piece = 0.05 if dashed else 0.1
        start = xs[0]
        while start < xs[-1]:
            sel = (xs >= start) & (xs < start + piece + 0.004)
            start += 2 * piece if dashed else piece
            if sel.sum() < 2:
                continue
            left = offset(line_offset + half_width)[sel]
            right = offset(line_offset - half_width)[sel]
            pl, pr = m.ground_to_pixels(left), m.ground_to_pixels(right)
            ok = valid(left, pl) & valid(right, pr)
            if ok.sum() < 2:
                continue
            polygon = np.r_[pl[ok], pr[ok][::-1]].round().astype(np.int32)
            cv2.fillPoly(image, [polygon], bgr)
    return image


def run_pipeline(image):
    import cv2
    m = model()
    small = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    top = m.horizon_row(0.8) // 2
    estimate, _, _, _ = detect_lane(small[top:], m, 2.0, top, lane_width_m=LANE_WIDTH)
    return estimate


class ProjectionTests(unittest.TestCase):
    def test_round_trip(self):
        m = model()
        ground = np.array([[0.2, 0.0], [0.4, 0.1], [0.6, -0.15]])
        back = m.pixels_to_ground(m.ground_to_pixels(ground))
        self.assertLess(np.abs(back - ground).max(), 1e-3)

    def test_matches_dt_core_ground_projection(self):
        # Pairs recorded on myrobot2: line_detector pixels_normalized -> ground_projection output.
        m = model()
        pairs = [((0.4937500059604645, 0.5833333134651184), None)]
        pixel = np.array([[pairs[0][0][0] * 640, pairs[0][0][1] * 480]])
        ground = m.pixels_to_ground(pixel)[0]
        self.assertGreater(ground[0], 0.2)
        self.assertLess(ground[0], 0.6)


class LaneDetectionTests(unittest.TestCase):
    def assert_estimate(self, estimate, d, phi, kappa, tol_d=0.015, tol_phi=0.05, tol_k=0.6):
        self.assertTrue(estimate.valid)
        self.assertAlmostEqual(estimate.d, d, delta=tol_d)
        self.assertAlmostEqual(estimate.phi, phi, delta=tol_phi)
        self.assertAlmostEqual(estimate.curvature, kappa, delta=tol_k)

    def test_centered_straight(self):
        estimate = run_pipeline(render(0.0, 0.0, 0.0))
        self.assert_estimate(estimate, 0.0, 0.0, 0.0)
        self.assertEqual(estimate.source, "both")

    def test_offset_and_yaw(self):
        estimate = run_pipeline(render(0.04, 0.15, 0.0))
        self.assert_estimate(estimate, 0.04, 0.15, 0.0)

    def test_right_offset_yawed_right(self):
        estimate = run_pipeline(render(-0.05, -0.12, 0.0))
        self.assert_estimate(estimate, -0.05, -0.12, 0.0)

    def test_left_curve(self):
        estimate = run_pipeline(render(0.0, 0.0, 1.8))
        self.assert_estimate(estimate, 0.0, 0.0, 1.8)

    def test_right_curve(self):
        estimate = run_pipeline(render(0.02, 0.05, -1.8))
        self.assert_estimate(estimate, 0.02, 0.05, -1.8)

    def test_yellow_only_dashed(self):
        estimate = run_pipeline(render(0.03, 0.0, 0.0, white=False, dashed_yellow=True))
        self.assert_estimate(estimate, 0.03, 0.0, 0.0, tol_d=0.02)
        self.assertEqual(estimate.source, "yellow")

    def test_white_only(self):
        estimate = run_pipeline(render(-0.03, 0.05, 0.0, yellow=False))
        self.assert_estimate(estimate, -0.03, 0.05, 0.0, tol_d=0.02)
        self.assertEqual(estimate.source, "white")

    def test_no_lines(self):
        self.assertFalse(run_pipeline(render(0.0, 0.0, 0.0, yellow=False, white=False)).valid)

    def test_opposite_lane_white_is_ignored(self):
        # Ground points: yellow centre at +0.115, our white at -0.115, the
        # opposite lane's white at +0.345. The far white must not be used.
        xs = np.linspace(0.1, 0.6, 60)
        yellow = np.c_[xs, np.full_like(xs, 0.115)]
        white = np.r_[np.c_[xs, np.full_like(xs, -0.115)], np.c_[xs, np.full_like(xs, 0.345)]]
        estimate = estimate_lane(yellow, white, LANE_WIDTH)
        self.assertEqual(estimate.source, "both")
        self.assertAlmostEqual(estimate.d, 0.0, delta=0.005)

    def test_white_at_another_angle_is_not_paired(self):
        # Fits recorded on myrobot2 parked on a straight (raw frame 05, 2026-11-16):
        # a steady yellow line, and a white fit the right width away mid-overlap
        # but ~0.67 rad off near the robot. Pairing them gave phi=+0.78 instead of
        # ~+0.15; the yellow line alone must be used.
        rng = np.random.default_rng(1)
        xs = np.linspace(0.13, 0.69, 200)
        yellow = np.c_[xs, -0.030 - 0.15 * xs + 0.05 * xs * xs + rng.normal(0, 0.003, len(xs))]
        xw = np.linspace(0.28, 0.69, 120)
        white = np.c_[xw, 0.166 - 1.84 * xw + 1.48 * xw * xw + rng.normal(0, 0.003, len(xw))]
        estimate = estimate_lane(yellow, white, LANE_WIDTH)
        self.assertEqual(estimate.source, "yellow")
        self.assertIn("white dropped: not parallel", estimate.notes)
        self.assertAlmostEqual(estimate.d, 0.030 + LANE_WIDTH / 2, delta=0.015)
        # The yellow alone: its slope is 0.15 at the robot and 0.11 mid-view.
        self.assertAlmostEqual(estimate.phi, math.atan(0.13), delta=0.03)

    def test_fit_rejects_outliers(self):
        xs = np.linspace(0.1, 0.6, 80)
        points = np.c_[xs, 0.05 + 0.1 * xs]
        noise = np.c_[np.linspace(0.1, 0.6, 15), np.full(15, -0.3)]
        fit = fit_line(np.r_[points, noise])
        # Line y = 0.05 + 0.1 x, left of the robot: d < 0, phi = -atan(0.1).
        self.assertAlmostEqual(fit.d, -0.05 / math.sqrt(1.01), delta=0.005)
        self.assertAlmostEqual(fit.phi, -math.atan(0.1), delta=0.02)
        self.assertAlmostEqual(fit.kappa, 0.0, delta=0.05)
        self.assertEqual(fit.inliers, 80)


class TrackerTests(unittest.TestCase):
    def test_prediction_follows_motion(self):
        history = VelocityHistory()
        history.add(0.0, 0.2, 0.0)
        tracker = LaneTracker()
        tracker.update(0.0, 0.0, 0.1, 0.0, 1.0, history)
        d, phi, _ = tracker.state_at(0.5, history)
        self.assertAlmostEqual(d, 0.2 * math.sin(0.1) * 0.5, delta=0.002)
        self.assertAlmostEqual(phi, 0.1, delta=1e-6)

    def test_turning_on_a_curve_keeps_heading(self):
        # Driving the road's own curvature keeps phi constant.
        history = VelocityHistory()
        history.add(0.0, 0.15, 0.15 * 2.0)
        tracker = LaneTracker()
        tracker.update(0.0, 0.0, 0.0, 2.0, 1.0, history)
        _, phi, _ = tracker.state_at(0.4, history)
        self.assertAlmostEqual(phi, 0.0, delta=1e-6)

    def test_filter_smooths_noise(self):
        history = VelocityHistory()
        history.add(0.0, 0.0, 0.0)
        tracker = LaneTracker()
        rng = np.random.default_rng(1)
        for i in range(60):
            tracker.update(i * 0.05, 0.02 + rng.normal(0, 0.01), 0.0, 0.0, 1.0, history)
        d, _, _ = tracker.state_at(3.0, history)
        self.assertAlmostEqual(d, 0.02, delta=0.006)

    def test_stale_state_is_dropped(self):
        history = VelocityHistory()
        tracker = LaneTracker(reset_after_s=1.0)
        tracker.update(0.0, 0.0, 0.0, 0.0, 1.0, history)
        self.assertIsNone(tracker.state_at(2.0, history))


class VisionControlTests(unittest.TestCase):
    def command(self, d, phi, kappa):
        target = target_on_lane_curve(d, phi, kappa, adaptive_lookahead(0.15))
        return command_from_target(target, LIMITS)

    def test_left_of_centre_steers_right(self):
        self.assertLess(self.command(0.05, 0.0, 0.0)["omega"], 0.0)

    def test_yawed_left_steers_right(self):
        self.assertLess(self.command(0.0, 0.2, 0.0)["omega"], 0.0)

    def test_left_curve_steers_left_with_feed_forward(self):
        result = self.command(0.0, 0.0, 2.0)
        self.assertGreater(result["omega"], 0.0)
        # Pure Pursuit on the curve commands about the road's own curvature.
        self.assertAlmostEqual(result["curvature"], 2.0, delta=0.3)

    def test_centered_straight_goes_straight(self):
        result = self.command(0.0, 0.0, 0.0)
        self.assertAlmostEqual(result["omega"], 0.0, delta=1e-9)
        self.assertGreaterEqual(result["v"], LIMITS["min_speed"])

    def test_steering_is_stronger_than_old_lookahead(self):
        # 3 cm offset: old fixed 0.42 m lookahead vs adaptive (~0.29 m at 0.15 m/s).
        old = command_from_target(target_on_lane_curve(0.03, 0.0, 0.0, 0.42), LIMITS)
        new = self.command(0.03, 0.0, 0.0)
        self.assertGreater(abs(new["curvature"]), 1.8 * abs(old["curvature"]))


class ArcModelTests(unittest.TestCase):
    def test_three_points_on_a_line_give_a_straight_arc(self):
        arc = arc_through((0.1, 0.06), (0.3, 0.08), (0.5, 0.10))
        self.assertAlmostEqual(arc.kappa, 0.0, delta=1e-9)
        self.assertAlmostEqual(arc.d, -0.05 / math.sqrt(1.01), delta=1e-6)
        self.assertAlmostEqual(arc.phi, -math.atan(0.1), delta=1e-6)

    def test_three_points_on_a_circle_recover_it(self):
        truth = ArcFit(0.02, 0.1, RIGHT_TURN)
        arc = arc_through(*truth.points([0.02, 0.12, 0.24]))
        for got, want in ((arc.d, 0.02), (arc.phi, 0.1), (arc.kappa, RIGHT_TURN)):
            self.assertAlmostEqual(got, want, delta=1e-6)

    def test_lines_of_a_right_turn_are_concentric(self):
        centre = ArcFit(0.0, 0.0, RIGHT_TURN)
        yellow, white = centre.shifted(LANE_WIDTH / 2), centre.shifted(-LANE_WIDTH / 2)
        self.assertAlmostEqual(1 / abs(yellow.kappa), 0.18 + LANE_WIDTH / 2, delta=1e-9)
        self.assertAlmostEqual(1 / abs(white.kappa), 0.18 - LANE_WIDTH / 2, delta=1e-9)
        back = white.shifted(LANE_WIDTH / 2)
        self.assertAlmostEqual(back.kappa, RIGHT_TURN, delta=1e-9)
        self.assertAlmostEqual(back.d, 0.0, delta=1e-9)

    def test_offset_through_the_turn_centre_is_refused(self):
        self.assertIsNone(ArcFit(0.0, 0.0, RIGHT_TURN).shifted(-0.2))

    def test_locate_gives_signed_offsets(self):
        arc = ArcFit(0.0, 0.0, LEFT_TURN)
        on = arc.points([0.1, 0.2])
        _, offset = arc.locate(np.r_[on, on + [0.0, 0.01]])
        self.assertLess(np.abs(offset[:2]).max(), 1e-9)
        self.assertTrue(np.all(offset[2:] > 0.0))


class TurnPerceptionTests(unittest.TestCase):
    """Tight turns were rejected outright by the quadratic model."""

    def check_turn(self, d, phi, kappa, tol_k):
        estimate = run_pipeline(render(d, phi, kappa))
        self.assertTrue(estimate.valid, estimate.reason)
        self.assertAlmostEqual(estimate.curvature, kappa, delta=tol_k)
        self.assertAlmostEqual(estimate.d, d, delta=0.03)
        return estimate

    def test_right_turn(self):
        self.check_turn(0.0, 0.0, RIGHT_TURN, 1.2)

    def test_right_turn_offset_and_yawed(self):
        self.check_turn(0.02, 0.1, RIGHT_TURN, 1.2)

    def test_left_turn(self):
        self.check_turn(0.0, 0.0, LEFT_TURN, 0.4)

    def test_turn_ahead_is_seen_before_it_starts(self):
        estimate = run_pipeline(render(0.0, 0.0, RIGHT_TURN, straight_before_m=0.1))
        self.assertTrue(estimate.valid, estimate.reason)
        self.assertLess(estimate.curvature, -2.5)

    def test_rejected_estimate_says_why(self):
        xs = np.linspace(0.15, 0.5, 80)
        estimate = estimate_lane(np.zeros((0, 2)), np.c_[xs, -0.2 + 1.7 * (xs - 0.15)], LANE_WIDTH)
        self.assertFalse(estimate.valid)
        self.assertIn("rad", estimate.reason)


class ArcTargetTests(unittest.TestCase):
    def test_on_the_centre_line_pursuit_commands_the_road_curvature(self):
        for kappa in (0.0, LEFT_TURN, -LEFT_TURN, RIGHT_TURN):
            lookahead = turn_lookahead(adaptive_lookahead(0.1), kappa)
            target = target_on_arc(0.0, 0.0, kappa, lookahead)
            self.assertAlmostEqual(math.hypot(*target), lookahead, delta=1e-6)
            self.assertAlmostEqual(command_from_target(target, LIMITS)["curvature"], kappa, delta=0.02)

    def test_quadratic_target_under_steers_a_right_turn(self):
        lookahead = adaptive_lookahead(0.1)
        old = command_from_target(target_on_lane_curve(0.0, 0.0, RIGHT_TURN, lookahead), LIMITS)
        self.assertGreater(old["curvature"], 0.8 * RIGHT_TURN)  # old: ~-3.6 for -5.6

    def test_lookahead_shrinks_in_tight_turns(self):
        self.assertAlmostEqual(turn_lookahead(0.26, RIGHT_TURN), 1.2 * 0.18, delta=1e-9)
        self.assertEqual(turn_lookahead(0.26, 0.0), 0.26)
        self.assertEqual(turn_lookahead(0.26, LEFT_TURN), 0.26)

    def test_offset_in_a_turn_adds_correction(self):
        lookahead = turn_lookahead(adaptive_lookahead(0.1), RIGHT_TURN)
        left_of_centre = command_from_target(target_on_arc(0.03, 0.0, RIGHT_TURN, lookahead), LIMITS)
        self.assertLess(left_of_centre["curvature"], RIGHT_TURN)  # steers harder right


if __name__ == "__main__":
    unittest.main()


class PlausibilityTests(unittest.TestCase):
    def test_line_across_the_robot_is_rejected(self):
        # A white border crossing in front of the robot at ~60 degrees.
        xs = np.linspace(0.15, 0.5, 80)
        white = np.c_[xs, -0.2 + 1.7 * (xs - 0.15)]
        estimate = estimate_lane(np.zeros((0, 2)), white, LANE_WIDTH)
        self.assertFalse(estimate.valid)


class CalibratedLocalizationTests(unittest.TestCase):
    def test_box_distance_uses_calibrated_model(self):
        from pp_core.localize import localize_detection
        m = model()
        u, v = m.ground_to_pixels([[0.40, 0.03]])[0]
        detection = {"x_min": u - 20, "x_max": u + 20, "y_min": v - 40, "y_max": v}
        corridor = {"x_min_m": 0.05, "x_max_m": 1.2, "half_width_m": 0.12}
        result = localize_detection(detection, 640, 480, {"model": m, "max_distance_m": 2.5}, corridor)
        self.assertTrue(result["projection_valid"])
        self.assertAlmostEqual(result["distance_m"], 0.40, delta=0.005)
        self.assertAlmostEqual(result["lateral_m"], 0.03, delta=0.005)
        self.assertTrue(result["in_corridor"])

    def test_ground_to_image_rejects_points_outside_the_lens_view(self):
        from pp_core.geometry import ground_to_image
        calibration = {"model": model()}
        self.assertIsNotNone(ground_to_image(0.4, 0.0, calibration))
        self.assertIsNone(ground_to_image(0.05, 0.6, calibration))  # far to the side, right under the camera
