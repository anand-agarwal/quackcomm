#!/usr/bin/env python3
"""Latest-frame YOLO detector.

The camera callback only stores the newest compressed frame. Inference runs
on a worker thread, so a slow network cannot fill a queue or block the lane
controller and the safety supervisor, which are separate processes.
"""

import os
import threading
import time

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage

from pp_core.coco import COCO_NAMES, DEFAULT_OBSTACLE_CLASSES
from pp_core.inference import YoloRunner
from pp_core.paths import resolve_model_path
from pp_core.topics import RAW_DETECTIONS_SUFFIX, RAW_IMAGE_SUFFIX, join
from pp_core.tracking import DetectionTracker
from pp_core.yolo_postprocess import decode_predictions, expected_columns
from purepursuit.msg import Detection, DetectionArray


class YoloDetectorNode(object):
    def __init__(self):
        self.vehicle = rospy.get_param("~vehicle_name", os.environ.get("VEHICLE_NAME", "duckiebot"))
        default_image = join(self.vehicle, RAW_IMAGE_SUFFIX)
        self.image_topic = rospy.get_param("~image_topic", default_image)
        self.model_path = resolve_model_path(rospy.get_param("~model_path", ""))
        self.conf_threshold = float(rospy.get_param("~conf_threshold", 0.45))
        self.iou_threshold = float(rospy.get_param("~iou_threshold", 0.45))
        self.min_box_px = float(rospy.get_param("~min_box_px", 8.0))
        self.max_fps = float(rospy.get_param("~max_fps", 2.0))
        self.min_hits = int(rospy.get_param("~min_hits", 2))
        self.immediate_confidence = float(rospy.get_param("~immediate_confidence", 0.75))
        allowed = rospy.get_param("~classes_of_interest", DEFAULT_OBSTACLE_CLASSES)
        self.allowed = list(allowed)
        self.runner = YoloRunner(self.model_path)
        # Configured names win, then the names stored in the ONNX file, then COCO.
        names = list(rospy.get_param("~class_names", []))
        if names and self.runner.class_names and names != self.runner.class_names:
            rospy.logwarn("class_names %s differ from the model's own names %s. Using class_names.",
                          names, self.runner.class_names)
        self.class_names = names or list(self.runner.class_names) or list(COCO_NAMES)
        self._shape_checked = False
        self.tracker = DetectionTracker(
            min_hits=self.min_hits,
            immediate_confidence=self.immediate_confidence,
        )
        self._lock = threading.Lock()
        self._latest = None
        self._sequence = 0
        self._stop = False
        self.publisher = rospy.Publisher(
            join(self.vehicle, RAW_DETECTIONS_SUFFIX), DetectionArray, queue_size=1,
        )
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True
        self.thread.start()
        self.subscriber = rospy.Subscriber(
            self.image_topic, CompressedImage, self._on_image,
            queue_size=1, buff_size=2 ** 22,
        )
        rospy.on_shutdown(self._shutdown)
        if self.runner.ready:
            rospy.loginfo(
                "YOLO backend %s, input %s, model %s, image %s",
                self.runner.backend_name, self.runner.network_size,
                self.model_path, self.image_topic,
            )
        else:
            rospy.logerr("YOLO model is not runnable: %s", self.runner.error)
        self._publish(None, 0.0, 0.0, camera_ok=False)

    def _shutdown(self):
        self._stop = True

    def _on_image(self, msg):
        with self._lock:
            self._sequence += 1
            self._latest = (self._sequence, msg.data, rospy.Time.now())

    def _loop(self):
        last_sequence = 0
        while not self._stop and not rospy.is_shutdown():
            with self._lock:
                item = self._latest
            if item is None or item[0] == last_sequence:
                time.sleep(0.01)
                continue
            last_sequence = item[0]
            started = time.time()
            try:
                decode_started = time.time()
                image = self._decode(item[1])
                decode_ms = (time.time() - decode_started) * 1000.0
                if image is None:
                    self._publish(None, decode_ms, 0.0, camera_ok=False, stamp=item[2])
                else:
                    infer_started = time.time()
                    prediction = self.runner.infer(image) if self.runner.ready else None
                    infer_ms = (time.time() - infer_started) * 1000.0
                    detections = []
                    if prediction is not None:
                        if not self._shape_checked:
                            self._shape_checked = True
                            self._check_output_shape(prediction)
                        height, width = image.shape[:2]
                        detections = decode_predictions(
                            prediction, width, height, self.runner.network_size,
                            self.conf_threshold, self.iou_threshold,
                            self.class_names, self.allowed, self.min_box_px,
                        )
                    confirmed = self.tracker.update(detections)
                    self._publish(confirmed, decode_ms, infer_ms, camera_ok=True, stamp=item[2], image_shape=image.shape)
            except Exception as exc:
                rospy.logerr_throttle(2.0, "YOLO inference failed: %s", exc)
                self._publish(None, 0.0, 0.0, camera_ok=False, stamp=item[2])
            elapsed = time.time() - started
            period = 1.0 / max(self.max_fps, 0.1)
            if elapsed < period:
                time.sleep(period - elapsed)

    def _decode(self, data):
        import cv2
        encoded = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def _check_output_shape(self, prediction):
        """A wrong class list decodes silently into garbage, so say so once."""
        columns = expected_columns(prediction)
        count = len(self.class_names)
        if columns in (4 + count, 5 + count):
            rospy.loginfo("YOLO output has %d columns for %d classes %s.", columns, count, self.class_names)
        else:
            rospy.logerr("YOLO output has %d columns, which fits neither %d (v8) nor %d (v5) for "
                         "the %d class_names %s. Fix class_names in yolo.yaml.",
                         columns, 4 + count, 5 + count, count, self.class_names)

    def _publish(self, detections, decode_ms, infer_ms, camera_ok, stamp=None, image_shape=None):
        msg = DetectionArray()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.decode_latency_ms = float(decode_ms)
        msg.inference_latency_ms = float(infer_ms)
        msg.model_loaded = bool(self.runner.ready)
        msg.backend = self.runner.backend_name
        msg.camera_ok = bool(camera_ok)
        if image_shape is not None:
            msg.image_height = int(image_shape[0])
            msg.image_width = int(image_shape[1])
        if detections:
            for item in detections:
                detection = Detection()
                detection.class_id = int(item["class_id"])
                detection.class_name = item["class_name"]
                detection.confidence = float(item["confidence"])
                detection.x_min = float(item["x_min"])
                detection.y_min = float(item["y_min"])
                detection.x_max = float(item["x_max"])
                detection.y_max = float(item["y_max"])
                detection.distance_m = -1.0
                detection.lateral_m = 0.0
                detection.distance_from_tof = False
                detection.in_corridor = False
                detection.projection_valid = False
                detection.confirmed = bool(item.get("confirmed", False))
                msg.detections.append(detection)
        self.publisher.publish(msg)


def main():
    rospy.init_node("yolo_detector_node")
    YoloDetectorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
