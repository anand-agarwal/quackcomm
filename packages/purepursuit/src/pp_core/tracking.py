"""Greedy IoU tracker.

A detection is confirmed after `min_hits` overlapping observations, or on the
first observation when its confidence is at least `immediate_confidence`.
ToF stops do not wait for this confirmation.
"""

from pp_core.yolo_postprocess import iou


class Track(object):
    def __init__(self, detection):
        self.detection = dict(detection)
        self.hits = 1
        self.misses = 0

    def confirm(self, min_hits, immediate_confidence):
        confidence = float(self.detection.get("confidence", 0.0))
        return self.hits >= min_hits or confidence >= immediate_confidence


class DetectionTracker(object):
    def __init__(self, iou_threshold=0.3, max_misses=2, min_hits=2, immediate_confidence=0.75):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.immediate_confidence = immediate_confidence
        self.tracks = []

    def update(self, detections):
        unmatched = set(range(len(detections)))
        for track in self.tracks:
            best_index = None
            best_iou = self.iou_threshold
            for index in list(unmatched):
                score = iou(
                    (
                        track.detection["x_min"], track.detection["y_min"],
                        track.detection["x_max"], track.detection["y_max"],
                    ),
                    (
                        detections[index]["x_min"], detections[index]["y_min"],
                        detections[index]["x_max"], detections[index]["y_max"],
                    ),
                )
                if score >= best_iou:
                    best_iou = score
                    best_index = index
            if best_index is None:
                track.misses += 1
            else:
                unmatched.remove(best_index)
                track.detection = dict(detections[best_index])
                track.hits += 1
                track.misses = 0
        self.tracks = [track for track in self.tracks if track.misses <= self.max_misses]
        for index in unmatched:
            self.tracks.append(Track(detections[index]))
        confirmed = []
        for track in self.tracks:
            if track.misses == 0 and track.confirm(self.min_hits, self.immediate_confidence):
                item = dict(track.detection)
                item["confirmed"] = True
                item["hits"] = track.hits
                confirmed.append(item)
        return confirmed
