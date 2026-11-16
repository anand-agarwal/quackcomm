"""Decode YOLOv5 (N, 85) and YOLOv8-style (N, 4+C) predictions.

Coordinates are converted from the letterboxed network image back to the
original camera image. This module does not run a network.
"""

import numpy as np

from pp_core.coco import COCO_NAMES


def letterbox_params(width, height, network_size):
    """Scale and padding used by a YOLO letterbox. The image itself is not resized."""
    if width <= 0 or height <= 0 or network_size <= 0:
        raise ValueError("image and network dimensions must be positive")
    ratio = min(float(network_size) / float(height), float(network_size) / float(width))
    resized_w = int(round(width * ratio))
    resized_h = int(round(height * ratio))
    pad_x = (network_size - resized_w) // 2
    pad_y = (network_size - resized_h) // 2
    return ratio, pad_x, pad_y


def network_box_to_image(box, ratio, pad_x, pad_y, width, height):
    x1 = (box[0] - pad_x) / ratio
    y1 = (box[1] - pad_y) / ratio
    x2 = (box[2] - pad_x) / ratio
    y2 = (box[3] - pad_y) / ratio
    x1 = float(np.clip(x1, 0, max(0, width - 1)))
    y1 = float(np.clip(y1, 0, max(0, height - 1)))
    x2 = float(np.clip(x2, 0, max(0, width - 1)))
    y2 = float(np.clip(y2, 0, max(0, height - 1)))
    return x1, y1, x2, y2


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def nms(boxes, scores, iou_threshold):
    if not scores:
        return []
    order = list(np.argsort(np.asarray(scores))[::-1])
    keep = []
    while order:
        current = order.pop(0)
        keep.append(current)
        remaining = []
        for index in order:
            if iou(boxes[current], boxes[index]) < iou_threshold:
                remaining.append(index)
        order = remaining
    return keep


def _as_rows(prediction):
    array = np.asarray(prediction)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("YOLO output must have shape (N, C) or (1, N, C) or (1, C, N)")
    # YOLOv8 exports are often (4+classes, N).
    if array.shape[0] < array.shape[1] and array.shape[0] <= 256:
        array = array.T
    return array.astype(np.float64, copy=False)


def expected_columns(prediction):
    """Number of values per candidate box in a raw YOLO output."""
    return int(_as_rows(prediction).shape[1])


def _xywh_to_xyxy(box):
    center_x, center_y, width, height = box
    return [
        center_x - width * 0.5,
        center_y - height * 0.5,
        center_x + width * 0.5,
        center_y + height * 0.5,
    ]


def decode_predictions(prediction, image_width, image_height, network_size,
                       conf_threshold, iou_threshold, class_names=None,
                       allowed_classes=None, min_box_px=8.0):
    """Return a list of detection dicts in original-image pixels."""
    rows = _as_rows(prediction)
    if class_names is None:
        class_names = COCO_NAMES
    allowed = None
    if allowed_classes:
        allowed = set(allowed_classes)
    ratio, pad_x, pad_y = letterbox_params(image_width, image_height, network_size)
    boxes = []
    scores = []
    class_ids = []
    columns = rows.shape[1]
    # Drop low-confidence rows with numpy first. A 640 YOLOv5 output has 25200
    # rows, and looping over all of them in Python costs tens of ms on the Nano.
    if columns >= 5:
        has_objectness = columns == 4 + 1 + len(class_names) or columns > 6 and columns != 4 + len(class_names)
        if has_objectness:
            best = rows[:, 4] * rows[:, 5:].max(axis=1)
        else:
            best = rows[:, 4:].max(axis=1)
        rows = rows[best >= conf_threshold]
    for row in rows:
        if columns == 4 + 1 + len(class_names) or columns > 6 and columns != 4 + len(class_names):
            objectness = float(row[4])
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            confidence = objectness * float(class_scores[class_id])
            box = _xywh_to_xyxy(row[:4])
        elif columns >= 5:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            box = _xywh_to_xyxy(row[:4])
        else:
            continue
        if confidence < conf_threshold or class_id >= len(class_names):
            continue
        name = class_names[class_id]
        if allowed is not None and name not in allowed:
            continue
        # Some exports emit normalized coordinates.
        if max(abs(box[0]), abs(box[1]), abs(box[2]), abs(box[3])) <= 1.5:
            box = [
                box[0] * network_size,
                box[1] * network_size,
                box[2] * network_size,
                box[3] * network_size,
            ]
        image_box = network_box_to_image(box, ratio, pad_x, pad_y, image_width, image_height)
        if image_box[2] - image_box[0] < min_box_px or image_box[3] - image_box[1] < min_box_px:
            continue
        boxes.append(image_box)
        scores.append(confidence)
        class_ids.append(class_id)
    selected = nms(boxes, scores, iou_threshold)
    detections = []
    for index in selected:
        x1, y1, x2, y2 = boxes[index]
        detections.append({
            "class_id": int(class_ids[index]),
            "class_name": class_names[class_ids[index]],
            "confidence": float(scores[index]),
            "x_min": float(x1),
            "y_min": float(y1),
            "x_max": float(x2),
            "y_max": float(y2),
        })
    return detections
