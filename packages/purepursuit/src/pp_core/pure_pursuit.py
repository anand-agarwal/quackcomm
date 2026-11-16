"""Lane centreline and Pure Pursuit curvature.

Ground segments are in the robot frame used by Duckietown ground projection:
x forward and y left, in metres. Yellow (color 1) is the left lane boundary
and white (color 0) is the right boundary.

The curvature law is the one already used by the image-space follower:

    curvature = 2 * lateral / (forward^2 + lateral^2)
    omega = clip(speed * curvature)

Positive lateral (left of the robot axis) produces positive curvature and
therefore a left turn. Positive omega is counter-clockwise, which is the
Duckietown Twist2DStamped convention.

LanePose fallback assumes the dt-core sign: positive d means the robot is to
the left of the lane centre, and positive phi means it is yawed to the left.
Both signs are parameters so a robot with the opposite calibration can be
corrected without changing the pursuit law.
"""

import math

WHITE = 0
YELLOW = 1


def clip(value, low, high):
    return max(low, min(high, value))


def pursuit_command(forward_m, lateral_m, max_speed, min_speed, max_omega,
                    max_lateral_accel, curvature_slowdown_gain):
    # In a tight turn the target can be well to the side; only a target at
    # the robot (or behind it) has no meaningful pursuit arc.
    if forward_m <= 0.0 or math.hypot(forward_m, lateral_m) < 0.05:
        return None
    denominator = forward_m * forward_m + lateral_m * lateral_m
    if denominator < 1e-5:
        return None
    curvature = 2.0 * lateral_m / denominator
    curvature = clip(curvature, -8.0, 8.0)
    curve_speed = math.sqrt(max_lateral_accel / max(abs(curvature), 1e-3))
    yaw_speed = max_omega / max(abs(curvature), 1e-3)
    turn_factor = 1.0 / (1.0 + curvature_slowdown_gain * abs(curvature))
    speed = min(max_speed, curve_speed, yaw_speed, max_speed * turn_factor)
    if speed < min_speed:
        speed = min_speed
    omega = clip(speed * curvature, -max_omega, max_omega)
    return {
        "v": float(speed),
        "omega": float(omega),
        "curvature": float(curvature),
        "forward_m": float(forward_m),
        "lateral_m": float(lateral_m),
    }


def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def centerline_from_segments(segments, lane_width_m, bin_m=0.05,
                             x_min_m=0.05, x_max_m=1.2, max_jump_m=0.18):
    """Build (x, y) centre points from white and yellow ground segments.

    `segments` is an iterable of (color, [(x, y), ...]).
    """
    bins = {}
    for color, points in segments:
        if color not in (WHITE, YELLOW):
            continue
        key = "yellow" if color == YELLOW else "white"
        for point in points:
            x_value = float(point[0])
            y_value = float(point[1])
            if x_value < x_min_m or x_value > x_max_m:
                continue
            index = int(x_value / bin_m)
            slot = bins.setdefault(index, {"yellow": [], "white": []})
            slot[key].append(y_value)
    centers = []
    previous_y = None
    for index in sorted(bins):
        slot = bins[index]
        x_value = (index + 0.5) * bin_m
        yellow = slot["yellow"]
        white = slot["white"]
        # Our right boundary is the white line right of the yellow centre line.
        # White left of the yellow belongs to the opposite lane's outer edge.
        if yellow:
            white = [y for y in white if y < _median(yellow)]
        else:
            white = [y for y in white if y < 0.5 * lane_width_m]
        if white:
            white = [max(white)] if len(white) == 1 else [y for y in white if y >= max(white) - 0.05]
        if yellow and white:
            center_y = 0.5 * (_median(yellow) + _median(white))
        elif white:
            # Right boundary. The lane centre is to its left.
            center_y = _median(white) + 0.5 * lane_width_m
        elif yellow:
            center_y = _median(yellow) - 0.5 * lane_width_m
        else:
            continue
        if previous_y is not None and abs(center_y - previous_y) > max_jump_m:
            continue
        previous_y = center_y
        centers.append((x_value, center_y))
    return centers


def point_at_lookahead(points, lookahead_m):
    if not points:
        return None
    if points[0][0] >= lookahead_m:
        return points[0]
    travelled = 0.0
    previous = points[0]
    for point in points[1:]:
        step = math.hypot(point[0] - previous[0], point[1] - previous[1])
        if step < 1e-6:
            previous = point
            continue
        if travelled + step >= lookahead_m:
            fraction = (lookahead_m - travelled) / step
            return (
                previous[0] + fraction * (point[0] - previous[0]),
                previous[1] + fraction * (point[1] - previous[1]),
            )
        travelled += step
        previous = point
    return points[-1]


def target_from_lane_pose(d_m, phi_rad, lookahead_m, d_sign=1.0, phi_sign=1.0):
    """Point on the lane centre expressed in the robot frame."""
    lateral = d_sign * float(d_m)
    heading = phi_sign * float(phi_rad)
    forward = lookahead_m * math.cos(heading) - lateral * math.sin(heading)
    side = -lookahead_m * math.sin(heading) - lateral * math.cos(heading)
    if forward <= 0.05:
        return None
    return forward, side


def adaptive_lookahead(speed, base_m=0.20, gain_s=0.6, min_m=0.18, max_m=0.35):
    """Lookahead that grows with speed: short enough to react, long enough to be smooth."""
    return clip(base_m + gain_s * max(0.0, float(speed)), min_m, max_m)


def target_on_lane_curve(d_m, phi_rad, curvature, lookahead_m):
    """Point on the lane centre lookahead_m ahead, in the robot frame (x forward, y left).

    The centre line in the robot frame is approximately
    y(x) = -d - tan(phi) x + curvature x^2 / 2, so the target already carries the
    road's bend. Pure Pursuit to this point gives curvature feed-forward plus
    feedback on d and phi: kappa_cmd ~ curvature - 2 d / L^2 - 2 phi / L.
    """
    forward = float(lookahead_m)
    lateral = -float(d_m) - math.tan(float(phi_rad)) * forward + 0.5 * float(curvature) * forward * forward
    if forward <= 0.05:
        return None
    return forward, lateral


def arc_point(d_m, phi_rad, curvature, s):
    """Point at arc length s along the lane-centre circle described by (d, phi, curvature).

    Same convention as pp_core.lane_vision.ArcFit: the circle's point nearest
    the robot is at s = 0, and s grows in the direction away from the robot.
    """
    theta = -float(phi_rad)
    tx, ty = math.cos(theta), math.sin(theta)
    nx, ny = -ty, tx
    px, py = -float(d_m) * nx, -float(d_m) * ny
    k = float(curvature)
    if abs(k) < 1e-6:
        return px + s * tx, py + s * ty
    along, side = math.sin(k * s) / k, (1.0 - math.cos(k * s)) / k
    return px + along * tx + side * nx, py + along * ty + side * ny


def turn_lookahead(lookahead_m, curvature, radius_factor=1.2, min_m=0.12):
    """Shorten the lookahead in tight turns so the target stays on the turn.

    A Duckietown turn is a quarter circle: past ~1.4 radii of chord the target
    would sit on the circle beyond the turn's end.
    """
    if abs(curvature) < 1e-6:
        return float(lookahead_m)
    return float(max(min_m, min(lookahead_m, radius_factor / abs(curvature))))


def target_on_arc(d_m, phi_rad, curvature, lookahead_m, samples=240):
    """Point on the lane-centre circle at straight-line distance lookahead_m, in the robot frame.

    Exact for arcs of any curvature (target_on_lane_curve's quadratic is only
    valid for gentle bends). For a robot on the centre line, Pure Pursuit to
    this point commands exactly the road's curvature, so the turn is
    fed forward; offsets in d and phi move the point and add the correction.
    Returns the point nearest lookahead_m when the circle never gets that far.
    """
    lookahead_m = float(lookahead_m)
    k = abs(float(curvature))
    s_max = 2.0 * lookahead_m + abs(float(d_m))
    if k > 1e-6:
        s_max = min(s_max, math.pi / k)
    previous_s, previous_r = 0.0, None
    best, best_gap = None, float("inf")
    for i in range(1, samples + 1):
        s = s_max * i / samples
        x, y = arc_point(d_m, phi_rad, curvature, s)
        r = math.hypot(x, y)
        if previous_r is not None and previous_r < lookahead_m <= r:
            lo, hi = previous_s, s
            for _ in range(30):  # bisection on the crossing
                mid = 0.5 * (lo + hi)
                if math.hypot(*arc_point(d_m, phi_rad, curvature, mid)) < lookahead_m:
                    lo = mid
                else:
                    hi = mid
            return arc_point(d_m, phi_rad, curvature, 0.5 * (lo + hi))
        if abs(r - lookahead_m) < best_gap:
            best, best_gap = (x, y), abs(r - lookahead_m)
        previous_s, previous_r = s, r
    return best


def speed_for_curvature(curvature, max_speed, min_speed, max_omega, max_lateral_accel,
                        curvature_slowdown_gain):
    """(v, omega) for a commanded curvature, slowing for bends (same rule as pursuit_command)."""
    curvature = clip(curvature, -8.0, 8.0)
    curve_speed = math.sqrt(max_lateral_accel / max(abs(curvature), 1e-3))
    yaw_speed = max_omega / max(abs(curvature), 1e-3)
    turn_factor = 1.0 / (1.0 + curvature_slowdown_gain * abs(curvature))
    speed = max(min_speed, min(max_speed, curve_speed, yaw_speed, max_speed * turn_factor))
    return float(speed), float(clip(speed * curvature, -max_omega, max_omega))


def command_from_lane_state(d_m, phi_rad, curvature, lookahead_m, limits):
    """Steer by the lane's curvature plus feedback on d and phi.

        kappa_cmd = kappa * cos(phi) / (1 - kappa * d) - 2 d / L^2 - 2 sin(phi) / L

    The first term is the curvature of the lane's parallel through the
    robot, the turn fed forward. The gains are Pure Pursuit's own small-angle
    gains for lookahead L, so on a straight this is Pure Pursuit; but they
    stay at full strength in a turn, where geometric Pure Pursuit to a point
    on a 0.18 m circle loses most of its correction (a 4 cm offset added 0.4
    1/m instead of 1.2). Error dynamics in distance travelled: natural length
    L / sqrt(2), damping 1 / sqrt(2), at any speed.

    Returns the pursuit_command dict; forward_m / lateral_m are the arc
    target at L (for the status message).
    """
    d, phi, kappa = float(d_m), float(phi_rad), float(curvature)
    lookahead_m = max(float(lookahead_m), 0.05)
    feed_forward = kappa * math.cos(phi) / max(0.2, 1.0 - kappa * d)
    kappa_cmd = feed_forward - 2.0 * d / lookahead_m ** 2 - 2.0 * math.sin(phi) / lookahead_m
    kappa_cmd = clip(kappa_cmd, -8.0, 8.0)
    speed, omega = speed_for_curvature(
        kappa_cmd, limits["max_speed"], limits["min_speed"], limits["max_omega"],
        limits["max_lateral_accel"], limits["curvature_slowdown_gain"],
    )
    target = target_on_arc(d, phi, kappa, lookahead_m) or (lookahead_m, 0.0)
    return {
        "v": speed,
        "omega": omega,
        "curvature": float(kappa_cmd),
        "forward_m": float(target[0]),
        "lateral_m": float(target[1]),
    }


def command_from_target(target, limits):
    if target is None:
        return None
    return pursuit_command(
        target[0], target[1],
        limits["max_speed"], limits["min_speed"], limits["max_omega"],
        limits["max_lateral_accel"], limits["curvature_slowdown_gain"],
    )


def lane_confidence(center_count, expected_count):
    if expected_count <= 0:
        return 0.0
    return max(0.0, min(1.0, float(center_count) / float(expected_count)))
