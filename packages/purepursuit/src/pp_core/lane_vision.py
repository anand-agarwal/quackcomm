"""Lane detection from the camera image, in the robot ground frame.

Pipeline (no ROS in this module, so it can be unit tested):

1. HSV colour masks for the yellow centre line and the white edge line on a
   half-resolution frame, below a horizon row.
2. Mask pixels are sampled, undistorted with the camera intrinsics, turned
   into camera-normalized coordinates ((u - cx) / fx, (v - cy) / fy) and
   projected to the ground with the robot's extrinsic homography. This is the
   same chain dt-core's ground_projection_node uses (dt_computer_vision:
   rectify_pixel -> pixel2vector -> vector2ground); on myrobot2 it reproduces
   dt-core's ground points to a median of 2 mm.
3. Each line gets a robust circle-arc fit (x forward, y left, metres):
   Duckietown lines are straight segments and arcs, so a circle describes
   them at any curvature, including a right turn's ~0.18 m radius that a
   quadratic in x cannot follow. The two lines of a lane are concentric, so
   each gives the lane centre by moving it half a lane width along its
   normal; with both, the two centres are averaged.
4. The centre arc gives the dt-core LanePose conventions plus curvature:
   d > 0 robot left of centre, phi > 0 robot yawed left, curvature > 0 road
   turning left.
"""

import math

import numpy as np

YELLOW = "yellow"
WHITE = "white"


# --------------------------------------------------------------------------
# Camera model


class GroundModel(object):
    """Pixel (full-resolution, distorted) <-> ground (x forward, y left) mapping."""

    def __init__(self, K, D, homography, width, height):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(D, dtype=np.float64).reshape(-1)
        self.H = np.asarray(homography, dtype=np.float64).reshape(3, 3)
        self.H_inv = np.linalg.inv(self.H)
        self.width = int(width)
        self.height = int(height)

    def pixels_to_ground(self, pixels):
        """(N, 2) full-resolution distorted pixels -> (N, 2) ground points."""
        import cv2
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        if len(pixels) == 0:
            return np.zeros((0, 2))
        normalized = cv2.undistortPoints(pixels, self.K, self.D).reshape(-1, 2)
        homogeneous = np.c_[normalized, np.ones(len(normalized))] @ self.H.T
        scale = homogeneous[:, 2:3]
        scale[np.abs(scale) < 1e-9] = 1e-9
        return homogeneous[:, :2] / scale

    def ground_to_pixels(self, points):
        """(N, 2) ground points -> (N, 2) full-resolution distorted pixels."""
        import cv2
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(points) == 0:
            return np.zeros((0, 2))
        homogeneous = np.c_[points, np.ones(len(points))] @ self.H_inv.T
        normalized = homogeneous[:, :2] / homogeneous[:, 2:3]
        object_points = np.c_[normalized, np.ones(len(normalized))].reshape(-1, 1, 3)
        pixels, _ = cv2.projectPoints(object_points, np.zeros(3), np.zeros(3), self.K, self.D)
        return pixels.reshape(-1, 2)

    def horizon_row(self, x_max_m, column=None):
        """Smallest full-resolution row whose ground point is still within x_max_m."""
        column = self.K[0, 2] if column is None else column
        rows = np.arange(0, self.height, 2, dtype=np.float64)
        ground = self.pixels_to_ground(np.c_[np.full(len(rows), column), rows])
        ok = (ground[:, 0] > 0.0) & (ground[:, 0] <= x_max_m)
        if not ok.any():
            return 0
        return int(rows[np.argmax(ok)])


def load_extrinsic_homography(path):
    """Ground homography (9 floats) from a Duckietown camera_extrinsic YAML, or None."""
    import yaml
    try:
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
    except (IOError, OSError, yaml.YAMLError):
        return None
    values = data.get("homography")
    if isinstance(values, list) and len(values) == 9:
        return [float(value) for value in values]
    return None


# --------------------------------------------------------------------------
# Colour masks


DEFAULT_COLORS = {
    # OpenCV HSV: H 0-180, S and V 0-255. Starting points from dt-core's
    # line detector; tune with the debug image for the room's lighting.
    "white_low": (0, 0, 150),
    "white_high": (180, 70, 255),
    "yellow_low": (22, 45, 60),
    "yellow_high": (40, 255, 255),
}


def color_masks(bgr, colors=None, auto_brightness=False, reference_v=130.0):
    """Return (yellow_mask, white_mask) as uint8 images.

    With auto_brightness the V thresholds follow the frame's median
    brightness (clipped to 0.6-1.4x), so a darker room lowers them.
    """
    import cv2
    colors = dict(DEFAULT_COLORS, **(colors or {}))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    scale = 1.0
    if auto_brightness:
        median_v = float(np.median(hsv[:, :, 2]))
        scale = min(1.4, max(0.6, median_v / max(reference_v, 1.0)))

    def bounds(low, high):
        low = np.array(low, dtype=np.float64)
        high = np.array(high, dtype=np.float64)
        low[2] = min(250.0, low[2] * scale)
        return low.astype(np.uint8), high.astype(np.uint8)

    yellow = cv2.inRange(hsv, *bounds(colors["yellow_low"], colors["yellow_high"]))
    white = cv2.inRange(hsv, *bounds(colors["white_low"], colors["white_high"]))
    return yellow, white


def sample_mask(mask, max_points, row_offset=0, scale=1.0, rng=None):
    """Pixel coordinates (u, v) of mask pixels, at most max_points, in full-resolution units."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.zeros((0, 2))
    if len(rows) > max_points:
        rng = rng or np.random.default_rng(0)
        keep = rng.choice(len(rows), size=max_points, replace=False)
        rows, cols = rows[keep], cols[keep]
    return np.c_[(cols + 0.5) * scale, (rows + row_offset + 0.5) * scale]




# --------------------------------------------------------------------------
# Line model: circle arcs
#
# Duckietown lines are straight segments and circular arcs (a right turn's
# lane centre has a radius of ~0.18 m, a left turn's ~0.41 m). A circle fits
# both exactly at any curvature, and the two lines of one lane are concentric
# circles, so the lane centre is a line moved sideways along its normal.


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class ArcFit(object):
    """A painted line (or the lane centre) as a circle arc; straight when kappa == 0.

    Stored relative to the robot (x forward, y left) in the dt-core LanePose
    convention, extended with curvature. p0 is the point of the full circle
    (or line) closest to the robot:

    - d: robot offset from the curve along the curve's left normal at p0
      (d > 0: robot left of the curve)
    - phi: robot heading minus the curve's direction at p0 (phi > 0: yawed left)
    - kappa: signed curvature (> 0: the curve turns left)

    The curve's direction points away from the robot. s is arc length from p0
    along it; [s_lo, s_hi] is the stretch where the line was seen.
    """

    STRAIGHT = 1e-6

    def __init__(self, d, phi, kappa, s_lo=0.0, s_hi=0.0, inliers=0, rms=0.0):
        self.d = float(d)
        self.phi = float(phi)
        self.kappa = float(kappa)
        self.s_lo = float(s_lo)
        self.s_hi = float(s_hi)
        self.inliers = int(inliers)
        self.rms = float(rms)

    @property
    def theta0(self):
        return -self.phi

    def _frame(self):
        theta = self.theta0
        t = np.array([math.cos(theta), math.sin(theta)])
        n = np.array([-t[1], t[0]])
        return -self.d * n, t, n

    @property
    def span(self):
        return self.s_hi - self.s_lo

    def points(self, s):
        """(N, 2) points at arc lengths s."""
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        p0, t, n = self._frame()
        if abs(self.kappa) < self.STRAIGHT:
            return p0 + s[:, None] * t
        k = self.kappa
        return p0 + (np.sin(k * s) / k)[:, None] * t + ((1.0 - np.cos(k * s)) / k)[:, None] * n

    def heading(self, s):
        return self.theta0 + self.kappa * np.asarray(s, dtype=np.float64)

    def locate(self, points):
        """(s, offset) of each point: arc length of its foot point and signed offset (> 0 left)."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        p0, t, n = self._frame()
        q = points - p0
        if abs(self.kappa) < self.STRAIGHT:
            return q @ t, q @ n
        k = self.kappa
        center = p0 + n / k
        u = points - center
        u0 = p0 - center
        angle = np.arctan2(u0[0] * u[:, 1] - u0[1] * u[:, 0], u @ u0)
        offset = math.copysign(1.0, k) * (1.0 / abs(k) - np.hypot(u[:, 0], u[:, 1]))
        return angle / k, offset

    def shifted(self, delta):
        """The parallel curve delta metres to the left (negative: right), or None.

        None when the shift passes through the turn centre (no such curve).
        """
        factor = 1.0 - self.kappa * float(delta)
        if factor < 0.05:
            return None
        return ArcFit(self.d - delta, self.phi, self.kappa / factor,
                      self.s_lo * factor, self.s_hi * factor, self.inliers, self.rms)

    @property
    def start_range(self):
        """Distance from the robot to the nearest seen end of the line."""
        ends = self.points([self.s_lo, self.s_hi])
        return float(np.hypot(ends[:, 0], ends[:, 1]).min())


def _arc_from_circle(center, radius):
    to_robot = -np.asarray(center, dtype=np.float64)
    dist = float(np.hypot(*to_robot))
    if dist < 1e-6 or radius <= 0.0:
        return None
    p0 = center + radius * to_robot / dist
    radial = (p0 - center) / radius
    t = np.array([-radial[1], radial[0]])
    if t[0] < 0.0:
        t = -t
    n = np.array([-t[1], t[0]])
    kappa = 1.0 / radius if float((center - p0) @ n) > 0.0 else -1.0 / radius
    return ArcFit(float(-p0 @ n), -math.atan2(t[1], t[0]), kappa)


def _arc_from_line(point, direction):
    u = np.asarray(direction, dtype=np.float64)
    norm = float(np.hypot(*u))
    if norm < 1e-9:
        return None
    u = u / norm
    if u[0] < 0.0:
        u = -u
    n = np.array([-u[1], u[0]])
    p0 = point - (point @ u) * u
    return ArcFit(float(-p0 @ n), -math.atan2(u[1], u[0]), 0.0)


def arc_through(p1, p2, p3):
    """Circle (or line, if collinear) through three points, as an ArcFit."""
    p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p1, p2, p3))
    det = 2.0 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
    if abs(det) < 1e-9:
        return _arc_from_line(p1, p3 - p1)
    a2, b2, c2 = p1 @ p1, p2 @ p2, p3 @ p3
    cx = ((a2 - b2) * (p3[1] - p1[1]) - (a2 - c2) * (p2[1] - p1[1])) / -det
    cy = ((a2 - b2) * (p3[0] - p1[0]) - (a2 - c2) * (p2[0] - p1[0])) / det
    center = np.array([cx, cy])
    return _arc_from_circle(center, float(np.hypot(*(p1 - center))))


def balance_by_distance(points, bin_m=0.05, max_per_bin=25, rng=None):
    """Cap the number of points per range bin (distance from the robot).

    Near the robot a few centimetres of line cover many pixels, so raw mask
    samples are dominated by the near end and the far end barely constrains
    the direction and curvature. Equalizing per bin fixes that.
    """
    if len(points) == 0:
        return points
    rng = rng or np.random.default_rng(0)
    bins = np.floor(np.hypot(points[:, 0], points[:, 1]) / bin_m).astype(np.int64)
    keep = []
    for value in np.unique(bins):
        index = np.nonzero(bins == value)[0]
        if len(index) > max_per_bin:
            index = rng.choice(index, size=max_per_bin, replace=False)
        keep.append(index)
    return points[np.concatenate(keep)]


def _refine(arc, points, fix_kappa=None, iterations=4, max_kappa=10.0):
    """Gauss-Newton on (d, phi[, kappa]) minimizing the geometric offsets."""
    params = np.array([arc.d, arc.phi, arc.kappa if fix_kappa is None else fix_kappa])
    free = 3 if fix_kappa is None else 2
    steps = np.array([1e-5, 1e-5, 1e-4])
    for _ in range(iterations):
        base = ArcFit(*params).locate(points)[1]
        J = np.empty((len(points), free))
        for j in range(free):
            moved = params.copy()
            moved[j] += steps[j]
            J[:, j] = (ArcFit(*moved).locate(points)[1] - base) / steps[j]
        try:
            delta, *_ = np.linalg.lstsq(J, -base, rcond=None)
        except np.linalg.LinAlgError:
            break
        params[:free] += delta
        params[2] = float(np.clip(params[2], -max_kappa, max_kappa))
        if np.abs(delta).max() < 1e-6:
            break
    return ArcFit(*params)


def fit_line(points, inlier_m=0.03, iterations=60, min_points=12,
             min_span_curved_m=0.12, max_kappa=10.0, near_scale_m=0.3,
             curve_gain_m=0.002, rng=None):
    """Robust circle-arc fit of a painted line's centre.

    RANSAC over three-point circles (and two-point lines) finds the line among
    outliers; the inlier band is wider than half a white line (2.4 cm), so the
    whole painted width is inside it. Gauss-Newton on the geometric distance
    then centres the arc. A line seen over less than min_span_curved_m is fit
    straight: a short arc does not determine its curvature.

    Candidates are scored by inliers weighted by nearness (exp(-range /
    near_scale_m)): after a turn a line continues straight, and the piece
    nearest the robot is the one to steer by. A curved fit is kept only when
    its rms is at least curve_gain_m below the straight fit's: the wide white
    line, cut by the image border, otherwise picks up a false bend.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < min_points:
        return None
    rng = rng or np.random.default_rng(0)
    points = balance_by_distance(points, rng=rng)
    n = len(points)
    if n < min_points:
        return None
    weights = np.exp(-np.hypot(points[:, 0], points[:, 1]) / near_scale_m)
    best_mask, best_score, best_arc = None, 0.0, None
    for i in range(iterations):
        size = 2 if i % 3 == 2 else 3
        sample = points[rng.choice(n, size=size, replace=False)]
        spread = np.hypot(*(sample.max(axis=0) - sample.min(axis=0)))
        if spread < 0.06:
            continue
        arc = _arc_from_line(sample[0], sample[1] - sample[0]) if size == 2 else arc_through(*sample)
        if arc is None or abs(arc.kappa) > max_kappa:
            continue
        mask = np.abs(arc.locate(points)[1]) < inlier_m
        if mask.sum() < min_points:
            continue
        score = float(weights[mask].sum())
        if score > best_score:
            best_score, best_mask, best_arc = score, mask, arc
    if best_mask is None:
        return None
    inliers = points[best_mask]
    arc = best_arc
    for _ in range(2):  # refine, then re-select the band around the refined arc
        straight = _refine(arc, inliers, fix_kappa=0.0)
        arc = _refine(arc, inliers, max_kappa=max_kappa)
        s, offset = arc.locate(inliers)
        rms_arc = float(np.sqrt(np.mean(offset ** 2)))
        rms_straight = float(np.sqrt(np.mean(straight.locate(inliers)[1] ** 2)))
        if np.ptp(s) < min_span_curved_m or rms_straight - rms_arc < curve_gain_m:
            arc = straight
        band = points[np.abs(arc.locate(points)[1]) < inlier_m]
        if len(band) < min_points:
            break
        inliers = band
    s, offset = arc.locate(inliers)
    arc.s_lo, arc.s_hi = float(s.min()), float(s.max())
    arc.inliers = len(inliers)
    arc.rms = float(np.sqrt(np.mean(offset ** 2)))
    return arc


# --------------------------------------------------------------------------
# Lane estimate


class LaneEstimate(object):
    def __init__(self, valid, d=0.0, phi=0.0, curvature=0.0, confidence=0.0,
                 source="none", center=None, yellow=None, white=None, lane_width=None,
                 reason="", notes=None):
        self.valid = bool(valid)
        self.d = float(d)
        self.phi = float(phi)
        self.curvature = float(curvature)
        self.confidence = float(confidence)
        self.source = source
        self.center = center
        self.yellow = yellow
        self.white = white
        self.lane_width = lane_width
        # Why the estimate is invalid, and what was dropped on the way (for the debug image).
        self.reason = reason
        self.notes = list(notes or [])


def _line_confidence(fit, min_points):
    if fit is None:
        return 0.0
    count_score = min(1.0, fit.inliers / float(4 * min_points))
    span_score = min(1.0, fit.span / 0.4)
    rms_score = max(0.0, 1.0 - fit.rms / 0.03)
    return count_score * (0.5 + 0.5 * span_score) * (0.5 + 0.5 * rms_score)


def _line_score(fit, min_points, near_m=0.2, far_m=0.6):
    """Line confidence, discounted when the line starts far ahead.

    d and phi are read at the robot, so a line seen only far away is
    extrapolated a long way and its direction error is amplified.
    """
    if fit is None:
        return 0.0
    near = 1.0 - 0.5 * min(1.0, max(0.0, (fit.start_range - near_m) / (far_m - near_m)))
    return _line_confidence(fit, min_points) * near


def pair_gaps(yellow, white, samples=12, extend_m=0.3):
    """Compare the white arc with the yellow arc where they run side by side.

    For points along the seen part of the white arc whose foot point on the
    yellow arc is within its seen part (+- extend_m), returns (gaps,
    heading_diffs, overlap_m): the perpendicular distance to the yellow arc
    (> 0 when white is right of yellow), the difference between the two
    directions there, and the length of yellow the samples cover. Two
    boundaries of one lane are concentric, so the gap is the lane width and the
    heading difference ~0 everywhere, on straights and in turns.
    Returns (None, None, 0.0) when they do not run side by side.
    """
    s_white = np.linspace(white.s_lo, white.s_hi, samples)
    s_yellow, offset = yellow.locate(white.points(s_white))
    keep = (s_yellow >= yellow.s_lo - extend_m) & (s_yellow <= yellow.s_hi + extend_m)
    if keep.sum() < 2:
        return None, None, 0.0
    covered = np.clip(s_yellow[keep], yellow.s_lo, yellow.s_hi)
    headings = np.abs(_wrap(white.heading(s_white[keep]) - yellow.heading(s_yellow[keep])))
    return -offset[keep], headings, float(np.ptp(covered))


def pair_is_consistent(yellow, white, lane_width_m, width_tolerance_m=0.08,
                       parallel_tolerance_m=0.06, near_heading_tolerance_rad=0.15,
                       heading_tolerance_rad=0.25, min_overlap_m=0.1):
    """True when yellow and white look like the two boundaries of one lane.

    Checked along the whole stretch where they run side by side, not at one
    point: the gap must stay near the lane width and change by at most
    parallel_tolerance_m, and the two must point the same way, most strictly
    at the near end, which sets d and phi at the robot. A white fit that
    matches the width at one distance while running at another angle (a stray
    piece of track, or an unstable fit of a short line) would otherwise be
    averaged into the centre and tilt it.

    Measured on myrobot2 (parked on a straight, 2026-11-16): the rejected
    white fits were 0.17-0.67 rad off the yellow at the near end.
    """
    gaps, headings, overlap = pair_gaps(yellow, white)
    if gaps is None or overlap < min_overlap_m:
        return False
    if np.abs(gaps - lane_width_m).max() > width_tolerance_m:
        return False
    if float(gaps.max() - gaps.min()) > parallel_tolerance_m:
        return False
    return float(headings[0]) <= near_heading_tolerance_rad and float(headings.max()) <= heading_tolerance_rad


def estimate_lane(yellow_points, white_points, lane_width_m=0.23, x_min_m=0.08, x_max_m=0.7,
                  min_points=25, inlier_m=0.03, width_tolerance_m=0.08, max_abs_phi=0.8,
                  max_kappa=10.0, rng=None):
    """Fit the yellow and white lines as arcs and return a LaneEstimate."""
    rng = rng or np.random.default_rng(0)
    notes = []

    def window(points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        keep = (points[:, 0] >= x_min_m) & (points[:, 0] <= x_max_m) & (np.abs(points[:, 1]) < 0.6)
        return points[keep]

    def fit(points):
        return fit_line(points, inlier_m=inlier_m, min_points=min_points, max_kappa=max_kappa, rng=rng)

    yellow_points = window(yellow_points)
    white_points = window(white_points)
    yellow = fit(yellow_points)

    # Our right edge is the white line to the right of the yellow centre line.
    # White left of the yellow line belongs to the opposite lane.
    if yellow is not None and len(white_points):
        white_points = white_points[yellow.locate(white_points)[1] < -0.5 * lane_width_m]
    elif len(white_points):
        white_points = white_points[white_points[:, 1] < 0.5 * lane_width_m]
    white = fit(white_points)

    measured_width = None
    if yellow is not None and white is not None:
        if pair_is_consistent(yellow, white, lane_width_m, width_tolerance_m):
            measured_width = float(np.median(pair_gaps(yellow, white)[0]))
        elif _line_score(yellow, min_points) >= _line_score(white, min_points):
            notes.append("white dropped: not parallel")
            white = None
        else:
            notes.append("yellow dropped: not parallel")
            yellow = None

    half = 0.5 * (measured_width or lane_width_m)
    from_yellow = yellow.shifted(-half) if yellow is not None else None
    from_white = white.shifted(half) if white is not None else None
    if yellow is not None and from_yellow is None:
        notes.append("yellow: offset past turn centre")
    if white is not None and from_white is None:
        notes.append("white: offset past turn centre")

    if from_yellow is not None and from_white is not None:
        wy, ww = _line_confidence(yellow, min_points), _line_confidence(white, min_points)
        total = max(wy + ww, 1e-6)
        center = ArcFit(
            (wy * from_yellow.d + ww * from_white.d) / total,
            (wy * from_yellow.phi + ww * from_white.phi) / total,
            (wy * from_yellow.kappa + ww * from_white.kappa) / total,
            min(from_yellow.s_lo, from_white.s_lo), max(from_yellow.s_hi, from_white.s_hi),
            yellow.inliers + white.inliers, 0.5 * (yellow.rms + white.rms),
        )
        confidence = min(1.0, 0.5 * (_line_confidence(yellow, min_points)
                                     + _line_confidence(white, min_points)) + 0.2)
        source = "both"
    elif from_yellow is not None:
        center, confidence, source = from_yellow, 0.75 * _line_confidence(yellow, min_points), YELLOW
    elif from_white is not None:
        center, confidence, source = from_white, 0.75 * _line_confidence(white, min_points), WHITE
    else:
        reason = "no lines" if yellow is None and white is None else "no usable centre"
        return LaneEstimate(False, yellow=yellow, white=white, reason=reason, notes=notes)

    d, phi, curvature = center.d, center.phi, center.kappa
    # A lane whose seen part starts at more than ~45 degrees to the robot is not
    # one we are following (facing a border or crossing a line); report nothing
    # rather than steer at it. The direction is taken where the lane is seen,
    # not at the circle's point nearest the robot: approaching a turn, the arc
    # ahead extended back to the robot meets it at a large angle while the
    # lane itself still runs straight ahead.
    near_heading = float(_wrap(center.heading(center.s_lo)))
    if abs(near_heading) > max_abs_phi:
        return LaneEstimate(False, d, phi, curvature, 0.0, "implausible", center, yellow, white,
                            measured_width, "lane at %+.2f rad (max %.2f)"
                            % (near_heading, max_abs_phi), notes)
    if abs(curvature) > max_kappa:
        return LaneEstimate(False, d, phi, curvature, 0.0, "implausible", center, yellow, white,
                            measured_width, "curvature %+.1f (max %.1f)" % (curvature, max_kappa), notes)
    return LaneEstimate(True, d, phi, curvature, confidence, source, center, yellow, white,
                        measured_width, "", notes)


def detect_lane(bgr_small, model, scale, row_offset_small, colors=None, max_points=600,
                lane_width_m=0.23, x_min_m=0.08, x_max_m=0.7, auto_brightness=False,
                max_kappa=10.0, rng=None):
    """Full pipeline on a (cropped) small frame. Returns (LaneEstimate, yellow_ground, white_ground, masks)."""
    rng = rng or np.random.default_rng(0)
    yellow_mask, white_mask = color_masks(bgr_small, colors, auto_brightness)
    yellow_px = sample_mask(yellow_mask, max_points, row_offset_small, scale, rng)
    white_px = sample_mask(white_mask, max_points, row_offset_small, scale, rng)
    yellow_ground = model.pixels_to_ground(yellow_px)
    white_ground = model.pixels_to_ground(white_px)
    estimate = estimate_lane(yellow_ground, white_ground, lane_width_m, x_min_m, x_max_m,
                             max_kappa=max_kappa, rng=rng)
    return estimate, yellow_ground, white_ground, (yellow_mask, white_mask)
