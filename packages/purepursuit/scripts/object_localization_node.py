#!/usr/bin/env python3
"""Project YOLO boxes onto the ground and publish the annotated camera image."""

import os

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage

from pp_core.geometry import ground_to_image
from pp_core.lane_vision import GroundModel, load_extrinsic_homography
from pp_core.localize import localize_detection
from pp_core.topics import (
    CAMERA_INFO_SUFFIX,
    DEBUG_IMAGE_SUFFIX,
    DETECTION_ARRAY_SUFFIX,
    GROUND_SEGMENTS_SUFFIX,
    OBSTACLE_DISTANCE_SUFFIX,
    RAW_DETECTIONS_SUFFIX,
    RAW_IMAGE_SUFFIX,
    join,
)
from purepursuit.msg import Detection, DetectionArray, ObstacleStatus


class ObjectLocalizationNode(object):
    def __init__(self):
        self.vehicle = rospy.get_param("~vehicle_name", os.environ.get("VEHICLE_NAME", "duckiebot"))
        self.image_topic = rospy.get_param(
            "~image_topic", join(self.vehicle, rospy.get_param("~image_suffix", RAW_IMAGE_SUFFIX)),
        )
        homography = rospy.get_param("~homography", [])
        self.calibration = {
            "homography": [float(value) for value in homography] if len(homography) == 9 else None,
            "fx": float(rospy.get_param("~fx", 330.0)),
            "fy": float(rospy.get_param("~fy", 330.0)),
            "cx": float(rospy.get_param("~cx", 320.0)),
            "cy": float(rospy.get_param("~cy", 240.0)),
            "height_m": float(rospy.get_param("~camera_height_m", 0.10)),
            "pitch_deg": float(rospy.get_param("~camera_pitch_deg", 15.0)),
            "cam_x_m": float(rospy.get_param("~camera_x_m", 0.0)),
            "max_distance_m": float(rospy.get_param("~max_projection_distance_m", 2.5)),
        }
        self.calibration_from_camera_info = False
        # Calibrated model: the robot's Duckietown extrinsic homography plus
        # camera_info intrinsics and distortion (same as lane_detector_node).
        extrinsic_file = rospy.get_param(
            "~extrinsic_file", "/data/config/calibrations/camera_extrinsic/%s.yaml" % self.vehicle,
        )
        self.ground_homography = load_extrinsic_homography(extrinsic_file)
        source = extrinsic_file
        if self.ground_homography is None:
            values = rospy.get_param("~ground_homography", [])
            self.ground_homography = [float(v) for v in values] if len(values) == 9 else None
            source = "config (localization.yaml)"
        fallback_K = rospy.get_param("~K", [])
        fallback_D = rospy.get_param("~D", [])
        if self.ground_homography is not None and len(fallback_K) == 9 and len(fallback_D) >= 4:
            self.calibration["model"] = GroundModel(fallback_K, fallback_D, self.ground_homography, 640, 480)
        self.homography_source = source
        self.corridor = {
            "x_min_m": float(rospy.get_param("~corridor_x_min_m", 0.05)),
            "x_max_m": float(rospy.get_param("~corridor_x_max_m", 1.2)),
            "half_width_m": float(rospy.get_param("~corridor_half_width_m", 0.12)),
            "uncertain_center_fraction": float(rospy.get_param("~uncertain_center_fraction", 0.35)),
            "tof_gate_m": float(rospy.get_param("~tof_gate_m", 0.08)),
        }
        self.tof = {"available": False, "range_m": None}
        self.latest_image = None
        self.segments = []
        self.publisher = rospy.Publisher(
            join(self.vehicle, DETECTION_ARRAY_SUFFIX), DetectionArray, queue_size=1,
        )
        self.debug_publisher = rospy.Publisher(
            join(self.vehicle, DEBUG_IMAGE_SUFFIX), CompressedImage, queue_size=1,
        )
        rospy.Subscriber(join(self.vehicle, RAW_DETECTIONS_SUFFIX), DetectionArray, self._on_detections, queue_size=1)
        rospy.Subscriber(self.image_topic, CompressedImage, self._on_image, queue_size=1, buff_size=2 ** 22)
        rospy.Subscriber(join(self.vehicle, CAMERA_INFO_SUFFIX), CameraInfo, self._on_camera_info, queue_size=1)
        rospy.Subscriber(join(self.vehicle, OBSTACLE_DISTANCE_SUFFIX), ObstacleStatus, self._on_tof, queue_size=1)
        try:
            from duckietown_msgs.msg import SegmentList
            rospy.Subscriber(
                join(self.vehicle, GROUND_SEGMENTS_SUFFIX), SegmentList, self._on_segments, queue_size=1,
            )
        except Exception as exc:
            rospy.logwarn("Lane overlay disabled: %s", exc)
        if self.calibration.get("model") is not None:
            rospy.loginfo("Object localization using the calibrated camera model (homography from %s).",
                          self.homography_source)
        else:
            rospy.logwarn(
                "No calibrated homography: object localization falls back to a pinhole guess "
                "(pitch %.1f deg, height %.3f m). Distances and the corridor overlay are approximate.",
                self.calibration["pitch_deg"], self.calibration["height_m"],
            )
        rospy.loginfo("Debug overlay image topic is %s.", self.image_topic)

    def _on_camera_info(self, msg):
        if self.ground_homography is not None and len(msg.K) == 9 and len(msg.D) >= 4:
            if not self.calibration_from_camera_info:
                self.calibration["model"] = GroundModel(
                    list(msg.K), list(msg.D), self.ground_homography, msg.width or 640, msg.height or 480,
                )
                self.calibration_from_camera_info = True
                rospy.loginfo("Calibrated camera model updated from camera_info (%dx%d).", msg.width, msg.height)
            return
        if self.calibration["homography"] or len(msg.K) < 6:
            return
        self.calibration["fx"] = float(msg.K[0])
        self.calibration["fy"] = float(msg.K[4])
        self.calibration["cx"] = float(msg.K[2])
        self.calibration["cy"] = float(msg.K[5])
        if not self.calibration_from_camera_info:
            self.calibration_from_camera_info = True
            rospy.loginfo(
                "Using camera_info intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f. Extrinsic pitch and height remain parameters.",
                self.calibration["fx"], self.calibration["fy"], self.calibration["cx"], self.calibration["cy"],
            )

    def _on_tof(self, msg):
        self.tof = {
            "available": bool(msg.valid and not msg.stale and msg.range_m >= 0.0),
            "range_m": None if msg.range_m < 0.0 else float(msg.range_m),
        }

    def _on_image(self, msg):
        self.latest_image = msg

    def _on_segments(self, msg):
        stored = []
        for segment in msg.segments:
            stored.append((int(segment.color), [(point.x, point.y) for point in segment.points]))
        self.segments = stored

    def _on_detections(self, msg):
        localized = []
        width = int(msg.image_width)
        height = int(msg.image_height)
        for detection in msg.detections:
            item = {
                "class_id": int(detection.class_id),
                "class_name": detection.class_name,
                "confidence": float(detection.confidence),
                "x_min": float(detection.x_min),
                "y_min": float(detection.y_min),
                "x_max": float(detection.x_max),
                "y_max": float(detection.y_max),
                "confirmed": bool(detection.confirmed),
            }
            if width > 0 and height > 0:
                item = localize_detection(item, width, height, self.calibration, self.corridor, self.tof)
            else:
                item["distance_m"] = None
                item["lateral_m"] = None
                item["projection_valid"] = False
                item["in_corridor"] = False
                item["distance_from_tof"] = False
            localized.append(item)
        self._publish(msg, localized)
        self._publish_debug(localized)

    def _publish(self, source, localized):
        msg = DetectionArray()
        msg.header.stamp = source.header.stamp
        msg.decode_latency_ms = source.decode_latency_ms
        msg.inference_latency_ms = source.inference_latency_ms
        msg.model_loaded = source.model_loaded
        msg.backend = source.backend
        msg.camera_ok = source.camera_ok
        msg.image_width = source.image_width
        msg.image_height = source.image_height
        for item in localized:
            detection = Detection()
            detection.class_id = int(item["class_id"])
            detection.class_name = item["class_name"]
            detection.confidence = float(item["confidence"])
            detection.x_min = float(item["x_min"])
            detection.y_min = float(item["y_min"])
            detection.x_max = float(item["x_max"])
            detection.y_max = float(item["y_max"])
            distance = item.get("distance_m")
            lateral = item.get("lateral_m")
            detection.distance_m = -1.0 if distance is None else float(distance)
            detection.lateral_m = 0.0 if lateral is None else float(lateral)
            detection.distance_from_tof = bool(item.get("distance_from_tof", False))
            detection.in_corridor = bool(item.get("in_corridor", False))
            detection.projection_valid = bool(item.get("projection_valid", False))
            detection.confirmed = bool(item.get("confirmed", False))
            msg.detections.append(detection)
        self.publisher.publish(msg)

    def _publish_debug(self, localized):
        if self.latest_image is None:
            return
        try:
            self._draw_debug(localized)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Debug image failed: %s", exc)

    def _draw_debug(self, localized):
        import cv2
        encoded = np.frombuffer(self.latest_image.data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            return
        self._draw_corridor(image)
        self._draw_segments(image)
        for item in localized:
            color = (0, 0, 255) if item.get("in_corridor") else (0, 180, 255)
            p1 = (int(item["x_min"]), int(item["y_min"]))
            p2 = (int(item["x_max"]), int(item["y_max"]))
            cv2.rectangle(image, p1, p2, color, 2)
            distance = item.get("distance_m")
            distance_text = "dist ?" if distance is None else "dist %.2fm%s" % (
                distance, " tof" if item.get("distance_from_tof") else "",
            )
            label = "%s %.2f %s" % (item["class_name"], item["confidence"], distance_text)
            cv2.putText(image, label, (p1[0], max(16, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        ok, jpg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = jpg.tobytes()
        self.debug_publisher.publish(msg)

    def _draw_corridor(self, image):
        """Outline of the driving corridor, sampled so the lens distortion shows as curves."""
        import cv2
        x_min, x_max = self.corridor["x_min_m"], self.corridor["x_max_m"]
        half = self.corridor["half_width_m"]
        xs = np.linspace(x_min, x_max, 40)
        outline = [(x, -half) for x in xs] + [(x_max, y) for y in np.linspace(-half, half, 10)] + \
                  [(x, half) for x in xs[::-1]] + [(x_min, y) for y in np.linspace(half, -half, 10)]
        run = []
        runs = []
        for ground_x, ground_y in outline:
            pixel = ground_to_image(ground_x, ground_y, self.calibration)
            if pixel is None:
                if len(run) > 1:
                    runs.append(run)
                run = []
                continue
            run.append((int(round(pixel[0])), int(round(pixel[1]))))
        if len(run) > 1:
            runs.append(run)
        for points in runs:
            cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, (0, 255, 0), 1)

    def _draw_segments(self, image):
        import cv2
        for color, points in self.segments:
            bgr = (0, 255, 255) if color == 1 else (255, 255, 255)
            for ground_x, ground_y in points:
                pixel = ground_to_image(ground_x, ground_y, self.calibration)
                if pixel is None:
                    continue
                cv2.circle(image, (int(round(pixel[0])), int(round(pixel[1]))), 2, bgr, -1)


def main():
    rospy.init_node("object_localization_node")
    ObjectLocalizationNode()
    rospy.spin()


if __name__ == "__main__":
    main()
