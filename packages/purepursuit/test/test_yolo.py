import unittest

import numpy as np

from pp_core.coco import COCO_NAMES
from pp_core.tracking import DetectionTracker
from pp_core.yolo_postprocess import decode_predictions, letterbox_params, network_box_to_image


class YoloPostprocessTests(unittest.TestCase):
    def test_letterbox_maps_a_yolov5_box_back_to_the_image(self):
        prediction = np.zeros((1, 25200, 85), dtype=np.float32)
        prediction[0, 100, 0] = 320.0
        prediction[0, 100, 1] = 320.0
        prediction[0, 100, 2] = 40.0
        prediction[0, 100, 3] = 40.0
        prediction[0, 100, 4] = 0.95
        prediction[0, 100, 5] = 1.0
        detections = decode_predictions(
            prediction, 640, 480, 640, 0.45, 0.45, COCO_NAMES, ["person"],
        )
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection["class_name"], "person")
        self.assertEqual(detection["class_id"], 0)
        self.assertAlmostEqual(detection["confidence"], 0.95)
        self.assertAlmostEqual((detection["x_min"] + detection["x_max"]) * 0.5, 320.0, delta=1.0)
        self.assertAlmostEqual((detection["y_min"] + detection["y_max"]) * 0.5, 240.0, delta=1.0)

    def test_yolov8_three_class_duckietown_output(self):
        # Ultralytics YOLOv8 export at 320 for duckie, duckiebot, cone: (1, 7, 2100).
        names = ["duckie", "duckiebot", "cone"]
        prediction = np.zeros((1, 7, 2100), dtype=np.float32)
        prediction[0, :4, 50] = (160.0, 160.0, 30.0, 20.0)
        prediction[0, 5, 50] = 0.88  # duckiebot
        prediction[0, :4, 60] = (40.0, 40.0, 20.0, 20.0)
        prediction[0, 4, 60] = 0.10  # weak duckie, below threshold
        detections = decode_predictions(prediction, 640, 480, 320, 0.45, 0.45, names, names)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_name"], "duckiebot")
        self.assertAlmostEqual(detections[0]["confidence"], 0.88, places=5)
        self.assertAlmostEqual((detections[0]["x_min"] + detections[0]["x_max"]) * 0.5, 320.0, delta=1.0)

    def test_low_confidence_and_unlisted_class_are_removed(self):
        prediction = np.zeros((1, 20, 85), dtype=np.float32)
        prediction[0, 0, :4] = (100, 100, 30, 30)
        prediction[0, 0, 4] = 0.20
        prediction[0, 0, 5] = 1.0
        prediction[0, 1, :4] = (200, 200, 40, 40)
        prediction[0, 1, 4] = 0.90
        prediction[0, 1, 6] = 1.0  # bicycle is allowed, airplane is not used here
        kept = decode_predictions(prediction, 640, 480, 640, 0.45, 0.45, COCO_NAMES, ["person"])
        self.assertEqual(kept, [])

    def test_single_frame_false_positive_is_not_confirmed(self):
        tracker = DetectionTracker(min_hits=2, immediate_confidence=0.75)
        weak = [{
            "class_id": 0, "class_name": "person", "confidence": 0.50,
            "x_min": 10, "y_min": 10, "x_max": 40, "y_max": 80,
        }]
        self.assertEqual(tracker.update(weak), [])
        confirmed = tracker.update(weak)
        self.assertEqual(len(confirmed), 1)
        self.assertTrue(confirmed[0]["confirmed"])

    def test_high_confidence_detection_is_confirmed_immediately(self):
        tracker = DetectionTracker(min_hits=2, immediate_confidence=0.75)
        strong = [{
            "class_id": 2, "class_name": "car", "confidence": 0.90,
            "x_min": 0, "y_min": 0, "x_max": 20, "y_max": 20,
        }]
        self.assertEqual(len(tracker.update(strong)), 1)

    def test_network_box_inverse_uses_padding(self):
        ratio, pad_x, pad_y = letterbox_params(640, 480, 640)
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, 80)
        box = network_box_to_image((0, 80, 10, 90), ratio, pad_x, pad_y, 640, 480)
        self.assertEqual(box, (0.0, 0.0, 10.0, 10.0))


if __name__ == "__main__":
    unittest.main()


class OnnxMetadataTests(unittest.TestCase):
    def test_ultralytics_names_metadata(self):
        from pp_core.inference import parse_names_metadata
        self.assertEqual(
            parse_names_metadata("{0: 'duckie', 1: 'duckiebot', 2: 'cone'}"),
            ["duckie", "duckiebot", "cone"],
        )
        self.assertEqual(parse_names_metadata("['a', 'b']"), ["a", "b"])
        self.assertEqual(parse_names_metadata(None), [])
        self.assertEqual(parse_names_metadata("not python"), [])

    def test_expected_columns_for_v5_and_v8_layouts(self):
        from pp_core.yolo_postprocess import expected_columns
        self.assertEqual(expected_columns(np.zeros((1, 25200, 85))), 85)
        self.assertEqual(expected_columns(np.zeros((1, 7, 2100))), 7)
