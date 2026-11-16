"""Validate one forward range reading.

Units are metres, matching sensor_msgs/Range. A reading of +inf is treated as
"no return inside the sensor range" when inf_means_clear is true. That is the
usual VL53L1X behaviour and can be turned off if a driver uses another sentinel.
"""

import math


def median(values):
    if not values:
        raise ValueError("median of an empty sample")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


class RangeFilter(object):
    def __init__(self, window):
        self.window = max(1, int(window))
        self.samples = []

    def push(self, value):
        self.samples.append(float(value))
        if len(self.samples) > self.window:
            self.samples = self.samples[-self.window:]
        return median(self.samples)


def assess_range(range_m, min_range_m, max_range_m, stop_distance_m, clear_distance_m,
                 age_s, timeout_s, inf_means_clear=True):
    """Classify a filtered range sample.

    `dangerously_close` is reported from the numeric value even when the sample
    is stale. The safety supervisor decides whether a stale value may stop the
    robot. `clear` is never true for a stale or invalid sample.
    """
    stale = age_s is None or age_s > timeout_s
    result = {
        "range_m": None,
        "valid": False,
        "stale": stale,
        "dangerously_close": False,
        "clear": False,
        "reason": "invalid",
    }
    if range_m is None or isinstance(range_m, float) and math.isnan(range_m):
        result["reason"] = "nan"
        return result
    if math.isinf(range_m):
        if inf_means_clear and range_m > 0.0:
            result["valid"] = True
            result["range_m"] = float(max_range_m) if max_range_m and max_range_m > 0.0 else None
            result["clear"] = not stale
            result["reason"] = "no return" if not stale else "stale no return"
        else:
            result["reason"] = "infinite"
        return result
    value = float(range_m)
    if value < 0.0:
        result["reason"] = "negative"
        return result
    if min_range_m and value < min_range_m:
        result["valid"] = True
        result["range_m"] = value
        result["dangerously_close"] = True
        result["reason"] = "below sensor minimum" if not stale else "stale below sensor minimum"
        return result
    if max_range_m and max_range_m > 0.0 and value > max_range_m:
        result["valid"] = True
        result["range_m"] = float(max_range_m)
        result["clear"] = not stale
        result["reason"] = "beyond sensor maximum" if not stale else "stale beyond sensor maximum"
        return result
    result["valid"] = True
    result["range_m"] = value
    if value <= stop_distance_m:
        result["dangerously_close"] = True
        result["reason"] = "inside stop distance" if not stale else "stale inside stop distance"
    elif value >= clear_distance_m:
        result["clear"] = not stale
        result["reason"] = "clear" if not stale else "stale clear reading"
    else:
        result["reason"] = "inside brake region" if not stale else "stale brake region"
    return result
