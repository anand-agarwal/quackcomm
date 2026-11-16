#!/usr/bin/env python3
"""Detect the lane in the camera image and publish d, phi and curvature.

Replaces dt-core's line_detector -> ground_projection -> lane_filter chain for
this demo. See pp_core/lane_vision.py for the method. The robot's own camera
calibration is used: intrinsics from camera_info, the ground homography from
the Duckietown extrinsic file (or config/lane_vision.yaml).
"""

import os
import threading
import time

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage

from pp_core.lane_vision import GroundModel, detect_lane, load_extrinsic_homography
from pp_core.topics import (
    CAMERA_INFO_SUFFIX,
    LANE_DEBUG_IMAGE_SUFFIX,
    LANE_ESTIMATE_SUFFIX,
    RAW_IMAGE_SUFFIX,
    join,
)
from purepursuit.msg import LaneEstimate


class LaneDetectorNode(object):
    def __init__(self):
        import cv2
        cv2.setNumThreads(1)  # the Nano's cores are shared with YOLO and dt-core
        self.cv2 = cv2
        self.vehicle = rospy.get_param("~vehicle_name", os.environ.get("VEHICLE_NAME", "duckiebot"))
        self.image_topic = rospy.get_param("~image_topic", join(self.vehicle, RAW_IMAGE_SUFFIX))
        self.max_fps = float(rospy.get_param("~max_fps", 20.0))
        self.lane_width_m = float(rospy.get_param("~lane_width_m", 0.23))
        self.x_min_m = float(rospy.get_param("~x_min_m", 0.08))
        self.x_max_m = float(rospy.get_param("~x_max_m", 0.7))
        self.max_points = int(rospy.get_param("~max_points_per_color", 600))
        self.max_kappa = float(rospy.get_param("~max_curvature", 10.0))
        self.auto_brightness = bool(rospy.get_param("~auto_brightness", False))
        self.debug_fps = float(rospy.get_param("~debug_fps", 2.0))
        self.colors = {
            key: tuple(int(v) for v in rospy.get_param("~" + key))
            for key in ("white_low", "white_high", "yellow_low", "yellow_high")
            if rospy.has_param("~" + key)
        }

        extrinsic_file = rospy.get_param(
            "~extrinsic_file", "/data/config/calibrations/camera_extrinsic/%s.yaml" % self.vehicle,
        )
        self.homography = load_extrinsic_homography(extrinsic_file)
        if self.homography is not None:
            rospy.loginfo("Ground homography from %s", extrinsic_file)
        else:
            values = rospy.get_param("~homography", [])
            if len(values) != 9:
                raise RuntimeError("No ground homography: %s is unreadable and ~homography is unset" % extrinsic_file)
            self.homography = [float(value) for value in values]
            rospy.loginfo("Ground homography from config (lane_vision.yaml); %s not readable", extrinsic_file)
        self.fallback_K = rospy.get_param("~K", [])
        self.fallback_D = rospy.get_param("~D", [])

        self.model = None
        self.crop_row_full = 0
        self._lock = threading.Lock()
        self._latest = None
        self._sequence = 0
        self._last_debug = 0.0
        self.publisher = rospy.Publisher(join(self.vehicle, LANE_ESTIMATE_SUFFIX), LaneEstimate, queue_size=1)
        self.debug_publisher = rospy.Publisher(
            join(self.vehicle, LANE_DEBUG_IMAGE_SUFFIX), CompressedImage, queue_size=1,
        )
        rospy.Subscriber(join(self.vehicle, CAMERA_INFO_SUFFIX), CameraInfo, self._on_camera_info, queue_size=1)
        rospy.Subscriber(self.image_topic, CompressedImage, self._on_image, queue_size=1, buff_size=2 ** 22)
        if len(self.fallback_K) == 9 and len(self.fallback_D) >= 4:
            self._set_model(self.fallback_K, self.fallback_D, 640, 480, "config")
        thread = threading.Thread(target=self._loop)
        thread.daemon = True
        thread.start()
        rospy.loginfo("Lane detector on %s -> %s", self.image_topic, join(self.vehicle, LANE_ESTIMATE_SUFFIX))

    def _set_model(self, K, D, width, height, origin):
        model = GroundModel(K, D, self.homography, width, height)
        crop = model.horizon_row(self.x_max_m + 0.1)
        self.model = model
        self.crop_row_full = max(0, crop)
        rospy.loginfo("Camera model from %s: %dx%d, processing rows below %d (x <= %.2f m)",
                      origin, width, height, crop, self.x_max_m + 0.1)

    def _on_camera_info(self, msg):
        if len(msg.K) != 9 or len(msg.D) < 4:
            return
        if self.model is not None and getattr(self, "_from_camera_info", False):
            return
        self._from_camera_info = True
        self._set_model(list(msg.K), list(msg.D), msg.width or 640, msg.height or 480, "camera_info")

    def _on_image(self, msg):
        stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()
        with self._lock:
            self._sequence += 1
            self._latest = (self._sequence, msg.data, stamp)

    def _loop(self):
        last = 0
        period = 1.0 / max(self.max_fps, 0.5)
        while not rospy.is_shutdown():
            with self._lock:
                item = self._latest
            if item is None or item[0] == last or self.model is None:
                time.sleep(0.005)
                continue
            last = item[0]
            started = time.time()
            try:
                self._process(item[1], item[2], started)
            except Exception as exc:
                rospy.logerr_throttle(2.0, "Lane detection failed: %s", exc)
            elapsed = time.time() - started
            if elapsed < period:
                time.sleep(period - elapsed)

    def _process(self, data, stamp, started):
        cv2 = self.cv2
        # Decoding straight to half resolution is much cheaper than decoding and resizing.
        small = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
        if small is None:
            raise ValueError("could not decode the camera image")
        # Some OpenCV builds ignore IMREAD_REDUCED_*; halve explicitly then.
        if small.shape[1] > 0.75 * self.model.width:
            small = cv2.resize(small, (self.model.width // 2, self.model.height // 2), interpolation=cv2.INTER_AREA)
        scale = float(self.model.width) / float(small.shape[1])
        top = min(int(self.crop_row_full / scale), small.shape[0] - 2)
        estimate, yellow_ground, white_ground, masks = detect_lane(
            small[top:], self.model, scale, top, self.colors or None, self.max_points,
            self.lane_width_m, self.x_min_m, self.x_max_m, self.auto_brightness, self.max_kappa,
        )
        msg = LaneEstimate()
        msg.header.stamp = stamp
        msg.valid = estimate.valid
        msg.d = estimate.d
        msg.phi = estimate.phi
        msg.curvature = estimate.curvature
        msg.confidence = estimate.confidence
        msg.lane_width = estimate.lane_width or 0.0
        msg.source = estimate.source
        msg.yellow_points = len(yellow_ground)
        msg.white_points = len(white_ground)
        msg.process_ms = (time.time() - started) * 1000.0
        self.publisher.publish(msg)
        if (self.debug_fps > 0 and self.debug_publisher.get_num_connections() > 0
                and time.time() - self._last_debug >= 1.0 / self.debug_fps):
            self._last_debug = time.time()
            self._publish_debug(small, top, scale, estimate, masks, stamp)

    def _draw_arc(self, image, arc, scale, color, thickness=2):
        """Draw the seen stretch of an ArcFit, projected back into the half-size image."""
        cv2 = self.cv2
        ground = arc.points(np.linspace(arc.s_lo, arc.s_hi, 40))
        pixels = self.model.ground_to_pixels(ground)
        # Points outside the lens model's field of view fold back under the
        # distortion polynomial; keep only those that project back to themselves.
        back = self.model.pixels_to_ground(pixels)
        ok = (np.linalg.norm(back - ground, axis=1) < 0.01) & np.all(np.isfinite(pixels), axis=1)
        ok &= (pixels[:, 0] >= 0) & (pixels[:, 0] < self.model.width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < self.model.height)
        pts = (pixels[ok] / scale).round().astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(image, [pts], False, color, thickness)

    def _publish_debug(self, small, top, scale, estimate, masks, stamp):
        """Overlay: masks, line fits and the lane centre the controller receives.

        The centre is green when the estimate is valid and red when it was
        rejected (the controller then ignores it); the reason and any line
        dropped as not parallel are written under the numbers.
        """
        cv2 = self.cv2
        image = small.copy()
        yellow_mask, white_mask = masks
        region = image[top:]
        region[yellow_mask > 0] = (0, 160, 255)
        region[white_mask > 0] = (255, 120, 120)
        for fit, color in ((estimate.yellow, (0, 255, 255)), (estimate.white, (255, 255, 255))):
            if fit is not None:
                self._draw_arc(image, fit, scale, color)
        if estimate.center is not None:
            centre_color = (0, 255, 0) if estimate.valid else (0, 0, 255)
            self._draw_arc(image, estimate.center, scale, centre_color, 3)
        cv2.line(image, (0, top), (image.shape[1] - 1, top), (80, 80, 80), 1)
        lines = ["%s d=%+.3f phi=%+.2f k=%+.2f c=%.2f" % (
            estimate.source, estimate.d, estimate.phi, estimate.curvature, estimate.confidence)]
        if not estimate.valid:
            lines.append("REJECTED: %s" % (estimate.reason or estimate.source))
        lines.extend(estimate.notes)
        for row, text in enumerate(lines):
            color = (0, 255, 0) if row == 0 and estimate.valid else (0, 0, 255) if row == 1 and not estimate.valid else (0, 200, 255)
            cv2.putText(image, text, (4, 14 + 14 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3)
            cv2.putText(image, text, (4, 14 + 14 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        ok, jpg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.format = "jpeg"
            msg.data = jpg.tobytes()
            self.debug_publisher.publish(msg)


def main():
    rospy.init_node("lane_detector_node")
    LaneDetectorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
