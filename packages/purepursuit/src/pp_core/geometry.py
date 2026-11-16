"""Camera and ground-plane geometry.

Robot frame: x forward, y left, z up, origin on the ground under the camera.
Camera optical frame: z along the optical axis, x right, y down.

`pitch_deg` is positive when the optical axis is tilted down. The default
pitch and height are approximate DB21 mounting values, not a calibrated
extrinsic. A measured homography replaces the pinhole model when it is set.

The bottom-centre of a bounding box is only a proxy for the ground contact
point. Tall objects and occluded feet make that proxy wrong, so callers must
treat a failed or distant projection as uncertain and use the ToF beam for
close-range distance.
"""

import math


def _rotate_y(vector, pitch_deg):
    pitch = math.radians(pitch_deg)
    cosine = math.cos(pitch)
    sine = math.sin(pitch)
    x_value, y_value, z_value = vector
    return [
        cosine * x_value + sine * z_value,
        y_value,
        -sine * x_value + cosine * z_value,
    ]


def _unit(vector):
    norm = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    if norm < 1e-9:
        return None
    return [vector[0] / norm, vector[1] / norm, vector[2] / norm]


def project_pixel_to_ground(u, v, fx, fy, cx, cy, height_m, pitch_deg,
                            cam_x_m=0.0, max_distance_m=2.5):
    """Intersect a pixel ray with the ground plane z = 0.

    Returns (x_forward_m, y_left_m), or None when the ray does not meet the
    ground ahead of the robot inside max_distance_m.
    """
    if fx <= 1e-6 or fy <= 1e-6 or height_m <= 0.0:
        return None
    ray_opt = _unit([(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0])
    if ray_opt is None:
        return None
    # Optical (x right, y down, z forward) to robot (x forward, y left, z up).
    ray_robot = [ray_opt[2], -ray_opt[0], -ray_opt[1]]
    direction = _rotate_y(ray_robot, pitch_deg)
    if direction[2] >= -1e-4:
        return None
    distance = -float(height_m) / direction[2]
    if distance <= 0.0:
        return None
    ground_x = float(cam_x_m) + distance * direction[0]
    ground_y = distance * direction[1]
    if ground_x <= 0.02 or ground_x > max_distance_m:
        return None
    return ground_x, ground_y


def ground_to_pixel(ground_x, ground_y, fx, fy, cx, cy, height_m, pitch_deg, cam_x_m=0.0):
    """Inverse of project_pixel_to_ground. Returns (u, v), or None if behind the camera."""
    if fx <= 1e-6 or fy <= 1e-6:
        return None
    vector = [ground_x - cam_x_m, ground_y, -height_m]
    pitch = math.radians(pitch_deg)
    cosine = math.cos(pitch)
    sine = math.sin(pitch)
    unpitched_x = cosine * vector[0] - sine * vector[2]
    unpitched_y = vector[1]
    unpitched_z = sine * vector[0] + cosine * vector[2]
    optical_z = unpitched_x
    optical_x = -unpitched_y
    optical_y = -unpitched_z
    if optical_z <= 1e-4:
        return None
    return fx * (optical_x / optical_z) + cx, fy * (optical_y / optical_z) + cy


def apply_homography(homography, u, v):
    """Apply a row-major 3x3 homography. Returns None if the scale vanishes."""
    if homography is None or len(homography) != 9:
        return None
    x_value = homography[0] * u + homography[1] * v + homography[2]
    y_value = homography[3] * u + homography[4] * v + homography[5]
    scale = homography[6] * u + homography[7] * v + homography[8]
    if abs(scale) < 1e-8:
        return None
    return x_value / scale, y_value / scale


def invert_homography(homography):
    if homography is None or len(homography) != 9:
        return None
    a, b, c, d, e, f, g, h, i = homography
    determinant = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(determinant) < 1e-12:
        return None
    return [
        (e * i - f * h) / determinant,
        (c * h - b * i) / determinant,
        (b * f - c * e) / determinant,
        (f * g - d * i) / determinant,
        (a * i - c * g) / determinant,
        (c * d - a * f) / determinant,
        (d * h - e * g) / determinant,
        (b * g - a * h) / determinant,
        (a * e - b * d) / determinant,
    ]


def pixel_to_ground(u, v, calibration):
    model = calibration.get("model")
    if model is not None:
        # Calibrated camera (intrinsics + distortion + Duckietown ground homography).
        ground_x, ground_y = model.pixels_to_ground([[u, v]])[0]
        max_distance = float(calibration.get("max_distance_m", 2.5))
        if not (0.02 < ground_x <= max_distance):
            return None
        return float(ground_x), float(ground_y)
    homography = calibration.get("homography")
    if homography:
        point = apply_homography(homography, u, v)
        if point is None:
            return None
        ground_x, ground_y = point
        max_distance = float(calibration.get("max_distance_m", 2.5))
        if ground_x <= 0.02 or ground_x > max_distance:
            return None
        return ground_x, ground_y
    return project_pixel_to_ground(
        u, v,
        float(calibration["fx"]), float(calibration["fy"]),
        float(calibration["cx"]), float(calibration["cy"]),
        float(calibration["height_m"]), float(calibration["pitch_deg"]),
        float(calibration.get("cam_x_m", 0.0)),
        float(calibration.get("max_distance_m", 2.5)),
    )


def ground_to_image(ground_x, ground_y, calibration):
    model = calibration.get("model")
    if model is not None:
        pixel = model.ground_to_pixels([[ground_x, ground_y]])[0]
        # Outside the lens model's field of view the distortion polynomial folds
        # points back into the image; reject anything that does not round-trip.
        back = model.pixels_to_ground([pixel])[0]
        if abs(back[0] - ground_x) > 0.01 or abs(back[1] - ground_y) > 0.01:
            return None
        return float(pixel[0]), float(pixel[1])
    homography = calibration.get("homography")
    if homography:
        inverse = invert_homography(homography)
        if inverse is None:
            return None
        return apply_homography(inverse, ground_x, ground_y)
    return ground_to_pixel(
        ground_x, ground_y,
        float(calibration["fx"]), float(calibration["fy"]),
        float(calibration["cx"]), float(calibration["cy"]),
        float(calibration["height_m"]), float(calibration["pitch_deg"]),
        float(calibration.get("cam_x_m", 0.0)),
    )


def in_driving_corridor(ground_x, ground_y, x_min_m, x_max_m, half_width_m):
    if ground_x is None or ground_y is None:
        return False
    return x_min_m <= ground_x <= x_max_m and abs(ground_y) <= half_width_m


def bbox_bottom_center(x_min, y_min, x_max, y_max):
    return 0.5 * (x_min + x_max), y_max


def overlaps_image_center(x_min, x_max, image_width, center_fraction):
    if image_width <= 0:
        return False
    margin = 0.5 * (1.0 - center_fraction) * image_width
    center_left = margin
    center_right = image_width - margin
    return x_max >= center_left and x_min <= center_right
