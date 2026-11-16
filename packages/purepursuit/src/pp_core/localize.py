"""Turn a bounding box into a ground-plane obstacle observation."""

from pp_core.geometry import (
    bbox_bottom_center,
    in_driving_corridor,
    overlaps_image_center,
    pixel_to_ground,
)


def localize_detection(detection, image_width, image_height, calibration, corridor, tof=None):
    """Return a copy of `detection` with corridor and distance fields.

    `tof` is a dict with `available` and `range_m`, or None.
    """
    result = dict(detection)
    u_pixel, v_pixel = bbox_bottom_center(
        detection["x_min"], detection["y_min"], detection["x_max"], detection["y_max"],
    )
    point = None
    if calibration is not None:
        point = pixel_to_ground(u_pixel, v_pixel, calibration)
    result["projection_valid"] = point is not None
    distance = None
    lateral = None
    if point is not None:
        distance = float(point[0])
        lateral = float(point[1])
        result["in_corridor"] = in_driving_corridor(
            point[0], point[1],
            float(corridor["x_min_m"]), float(corridor["x_max_m"]),
            float(corridor["half_width_m"]),
        )
    else:
        result["in_corridor"] = overlaps_image_center(
            detection["x_min"], detection["x_max"], image_width,
            float(corridor.get("uncertain_center_fraction", 0.35)),
        )
    from_tof = False
    if tof and tof.get("available") and tof.get("range_m") is not None:
        in_beam = result["in_corridor"] and (
            lateral is None or abs(lateral) <= float(corridor.get("tof_gate_m", 0.08))
        )
        if in_beam and (distance is None or float(tof["range_m"]) < distance):
            distance = float(tof["range_m"])
            from_tof = True
    result["distance_m"] = distance
    result["lateral_m"] = lateral
    result["distance_from_tof"] = from_tof
    return result
