#!/usr/bin/env python3
"""Camera-image lane follower with approximate Pure Pursuit (no BEV homography).

Detects the yellow left and white right lane markings in the forward camera
image, estimates the lane centre row-by-row, and follows a target point. Since
there is no camera/ground-plane calibration, pixel rows are mapped to an
approximate forward distance and lateral distance is scaled by observed lane
width. Tune conservatively on the real robot.
"""

import math
import os

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Float32


class PurePursuitLaneFollower:
    def __init__(self):
        self.vehicle = rospy.get_param(
            "~vehicle_name", os.environ.get("VEHICLE_NAME", "duckiebot"))
        self.image_topic = rospy.get_param(
            "~image_topic", f"/{self.vehicle}/camera_node/image/compressed")
        self.cmd_topic = rospy.get_param(
            "~cmd_topic",f"/{self.vehicle}/car_cmd_switch_node/cmd_lane_following")

        # Lane colours in HSV. Tune for lighting and camera exposure.
        self.white_lower = np.array(rospy.get_param("~white_hsv_lower", [0, 0, 155]), dtype=np.uint8)
        self.white_upper = np.array(rospy.get_param("~white_hsv_upper", [179, 90, 255]), dtype=np.uint8)
        self.yellow_lower = np.array(rospy.get_param("~yellow_hsv_lower", [15, 70, 80]), dtype=np.uint8)
        self.yellow_upper = np.array(rospy.get_param("~yellow_hsv_upper", [40, 255, 255]), dtype=np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Image scan area and lane geometry. No BEV dimensions or source points.
        self.scan_top_frac = float(rospy.get_param("~scan_top_frac", 0.42))
        self.scan_bottom_frac = float(rospy.get_param("~scan_bottom_frac", 0.88))
        self.scan_step_px = int(rospy.get_param("~scan_step_px", 5))
        self.lane_width_m = float(rospy.get_param("~lane_width_m", 0.585))
        self.near_lane_width_px = float(rospy.get_param("~near_lane_width_px", 180.0))
        self.far_lane_width_px = float(rospy.get_param("~far_lane_width_px", 35.0))
        self.min_lane_width_px = float(rospy.get_param("~min_lane_width_px", 12.0))
        self.min_valid_rows = int(rospy.get_param("~min_valid_rows", 8))
        self.min_confidence = float(rospy.get_param("~min_confidence", 0.25))
        self.max_row_jump_frac = float(rospy.get_param("~max_row_jump_frac", 0.12))

        # Controller and speed references. Low defaults are intentional until tested.
        self.lookahead_m = float(rospy.get_param("~lookahead_m", 0.42))
        self.max_speed = float(rospy.get_param("~max_speed", 0.10))
        self.min_speed = float(rospy.get_param("~min_speed", 0.035))
        self.max_omega = float(rospy.get_param("~max_omega", 1.8))
        self.max_lateral_accel = float(rospy.get_param("~max_lateral_accel", 0.18))
        self.curvature_slowdown_gain = float(rospy.get_param("~curvature_slowdown_gain", 0.65))
        self.turn_slowdown_gain = float(rospy.get_param("~turn_slowdown_gain", 0.75))
        self.max_fps = float(rospy.get_param("~max_fps", 12.0))
        self.image_timeout_s = float(rospy.get_param("~image_timeout_s", 0.5))
        self.publish_debug = bool(rospy.get_param("~publish_debug", False))

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist2DStamped, queue_size=1)
        self.conf_pub = rospy.Publisher("~confidence", Float32, queue_size=1)
        self.debug_pub = None
        if self.publish_debug:
            self.debug_pub = rospy.Publisher("~debug/compressed", CompressedImage, queue_size=1)

        self.last_process_time = rospy.Time(0)
        self.last_valid_time = rospy.Time(0)
        self.image_sub = rospy.Subscriber(
            self.image_topic, CompressedImage, self.image_callback,
            queue_size=1, buff_size=2**22)
        self.watchdog = rospy.Timer(rospy.Duration(0.1), self.watchdog_callback)
        rospy.on_shutdown(lambda: self.publish_cmd(0.0, 0.0))
        rospy.loginfo("Image-space Pure Pursuit: %s -> %s", self.image_topic, self.cmd_topic)

    def publish_cmd(self, v, omega):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = float(v)
        msg.omega = float(omega)
        self.cmd_pub.publish(msg)

    def stop(self, reason=None):
        self.publish_cmd(0.0, 0.0)
        if reason:
            rospy.logwarn_throttle(2.0, "Lane follower stopped: %s", reason)

    def watchdog_callback(self, _event):
        if self.last_valid_time == rospy.Time(0):
            self.stop("waiting for valid lane perception")
        elif (rospy.Time.now() - self.last_valid_time).to_sec() > self.image_timeout_s:
            self.stop("camera/lane estimate timed out")

    @staticmethod
    def row_candidates(mask, y, min_pixels=2):
        xs = np.flatnonzero(mask[y, :])
        if xs.size < min_pixels:
            return []
        groups = np.split(xs, np.where(np.diff(xs) > 3)[0] + 1)
        return [float(np.median(g)) for g in groups if len(g) >= min_pixels]

    def estimate_lane(self, image):
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white = cv2.morphologyEx(cv2.inRange(hsv, self.white_lower, self.white_upper), cv2.MORPH_CLOSE, self.kernel)
        yellow = cv2.morphologyEx(cv2.inRange(hsv, self.yellow_lower, self.yellow_upper), cv2.MORPH_CLOSE, self.kernel)
        top = int(np.clip(self.scan_top_frac, 0.1, 0.8) * h)
        bottom = int(np.clip(self.scan_bottom_frac, self.scan_top_frac + 0.05, 0.98) * h)
        rows, centers, widths = [], [], []
        prev_center = None

        for y in range(bottom, top, -max(1, self.scan_step_px)):
            whites = self.row_candidates(white, y)
            yellows = self.row_candidates(yellow, y)
            # Right white boundary: choose a candidate to the right of image centre.
            wc = [x for x in whites if x > 0.48 * w]
            # Left yellow boundary: choose the plausible candidate left of the white line.
            yc = [x for x in yellows if x < 0.68 * w]
            right = max(wc) if wc else None
            left = max(yc) if yc else None
            width_guess = self.far_lane_width_px + (self.near_lane_width_px - self.far_lane_width_px) * ((y - top) / max(1.0, bottom - top))

            if left is not None and right is not None and right - left >= self.min_lane_width_px:
                lane_width = right - left
                center = 0.5 * (left + right)
            elif right is not None:
                lane_width = width_guess
                center = right - 0.5 * lane_width
            elif left is not None:
                lane_width = width_guess
                center = left + 0.5 * lane_width
            else:
                continue

            if prev_center is not None and abs(center - prev_center) > self.max_row_jump_frac * w:
                continue
            prev_center = center
            rows.append(float(y))
            centers.append(float(center))
            widths.append(float(max(self.min_lane_width_px, lane_width)))

        if len(rows) < self.min_valid_rows:
            return None
        yv = np.asarray(rows, dtype=np.float64)
        xv = np.asarray(centers, dtype=np.float64)
        # Fit lane centre x as a function of image row. Degree 2 captures gentle bends.
        try:
            coeff = np.polyfit(yv, xv, 2)
        except (ValueError, np.linalg.LinAlgError):
            return None
        predicted = np.polyval(coeff, yv)
        residual = np.abs(xv - predicted)
        med = float(np.median(residual))
        coverage = min(1.0, len(rows) / max(1.0, (bottom - top) / max(1, self.scan_step_px)))
        fit_quality = math.exp(-med / max(3.0, 0.025 * w))
        confidence = float(np.clip(coverage * fit_quality, 0.0, 1.0))
        return coeff, confidence, white, yellow, top, bottom, float(np.median(widths))

    def calculate_command(self, coeff, image_shape, lane_width_px):
        h, w = image_shape[:2]
        top = int(self.scan_top_frac * h)
        bottom = int(self.scan_bottom_frac * h)
        # Approximate image-row-to-forward-distance mapping; not metric calibration.
        near_y = bottom
        far_y = top
        row_span = max(1.0, float(near_y - far_y))
        target_forward_m = min(1.0, max(0.20, self.lookahead_m))
        forward_range_m = 1.15
        target_y = near_y - (target_forward_m / forward_range_m) * row_span
        target_y = float(np.clip(target_y, far_y + 1.0, near_y - 1.0))
        target_u = float(np.polyval(coeff, target_y))

        # Convert target pixel offset to approximate lateral metres using lane width.
        lane_width_px = max(self.min_lane_width_px, lane_width_px)
        lateral_m = (target_u - (w / 2.0)) * self.lane_width_m / lane_width_px
        forward_m = target_forward_m
        denom = forward_m * forward_m + lateral_m * lateral_m
        if denom < 1e-5:
            return None
        curvature = 2.0 * lateral_m / denom

        # Also respond to how quickly the lane centre bends across image rows.
        slope_px_per_row = float(2.0 * coeff[0] * target_y + coeff[1])
        heading_proxy = slope_px_per_row * (row_span / max(1.0, w))
        curvature += self.turn_slowdown_gain * heading_proxy
        curvature = float(np.clip(curvature, -8.0, 8.0))

        curve_speed = math.sqrt(self.max_lateral_accel / max(abs(curvature), 1e-3))
        yaw_speed = self.max_omega / max(abs(curvature), 1e-3)
        # Slow proactively on bends; straight sections retain max_speed.
        turn_factor = 1.0 / (1.0 + self.curvature_slowdown_gain * abs(curvature))
        speed = min(self.max_speed, curve_speed, yaw_speed, self.max_speed * turn_factor)
        if speed < self.min_speed:
            # Preserve low-speed turn capability instead of stopping on every tight bend.
            speed = self.min_speed
        omega = float(np.clip(speed * curvature, -self.max_omega, self.max_omega))
        return speed, omega, curvature, (int(round(target_u)), int(round(target_y)))

    def publish_debug_image(self, image, masks, coeff, target):
        if self.debug_pub is None:
            return
        debug = image.copy()
        h, w = debug.shape[:2]
        for y in range(int(self.scan_top_frac * h), int(self.scan_bottom_frac * h), 3):
            x = int(round(np.polyval(coeff, y)))
            if 0 <= x < w:
                cv2.circle(debug, (x, y), 2, (0, 255, 0), -1)
        if target is not None:
            cv2.circle(debug, target, 7, (0, 0, 255), -1)
        ok, encoded = cv2.imencode(".jpg", debug, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = encoded.tobytes()
            self.debug_pub.publish(msg)

    def image_callback(self, msg):
        now = rospy.Time.now()
        if (now - self.last_process_time).to_sec() < 1.0 / max(self.max_fps, 1.0):
            return
        self.last_process_time = now
        try:
            encoded = np.frombuffer(msg.data, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                self.stop("could not decode camera image")
                return
            estimate = self.estimate_lane(image)
            if estimate is None:
                self.conf_pub.publish(Float32(data=0.0))
                self.stop("lane markings not found reliably")
                return
            coeff, confidence, white, yellow, _top, _bottom, lane_width_px = estimate
            self.conf_pub.publish(Float32(data=confidence))
            if confidence < self.min_confidence:
                self.stop("lane estimate confidence too low")
                return
            command = self.calculate_command(coeff, image.shape, lane_width_px)
            if command is None:
                self.stop("Pure Pursuit target unavailable")
                return
            v, omega, curvature, target = command
            self.publish_cmd(v, omega)
            self.last_valid_time = now
            self.publish_debug_image(image, (white, yellow), coeff, target)
            rospy.loginfo_throttle(2.0, "lane confidence=%.2f curvature=%.2f v=%.3f omega=%.3f", confidence, curvature, v, omega)
        except Exception as exc:
            self.stop("perception/control exception")
            rospy.logerr_throttle(2.0, "Pure Pursuit callback exception: %s", exc)


def main():
    rospy.init_node("pure_pursuit_lane_follower", anonymous=False)
    try:
        PurePursuitLaneFollower()
        rospy.spin()
    except Exception as exc:
        rospy.logfatal("Could not start image-space pure-pursuit lane follower: %s", exc)
        raise


if __name__ == "__main__":
    main()