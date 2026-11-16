"""Decide the single authorized chassis command.

YOLO says what an object is and where it sits in the corridor. The ToF says
how close something is on the forward beam. Pure Pursuit only proposes a
command. This function is the only place that turns a proposal into an
authorized velocity.
"""

import math


def clip(value, low, high):
    return max(low, min(high, value))


def _nearest_corridor_obstacle(obstacles):
    chosen = None
    for obstacle in obstacles:
        if not obstacle.get("confirmed") or not obstacle.get("in_corridor"):
            continue
        if chosen is None:
            chosen = obstacle
            continue
        distance = obstacle.get("distance_m")
        chosen_distance = chosen.get("distance_m")
        if distance is not None and (chosen_distance is None or distance < chosen_distance):
            chosen = obstacle
    return chosen


def _stopped(state, reason, obstacle_related, desired_v=0.0, desired_omega=0.0):
    return {
        "state": state,
        "reason": reason,
        "v": 0.0,
        "omega": 0.0,
        "desired_v": float(desired_v),
        "desired_omega": float(desired_omega),
        "obstacle_related": obstacle_related,
    }


def _moving(state, reason, v_value, omega_value, desired_v, desired_omega, obstacle_related=False):
    return {
        "state": state,
        "reason": reason,
        "v": float(v_value),
        "omega": float(omega_value),
        "desired_v": float(desired_v),
        "desired_omega": float(desired_omega),
        "obstacle_related": obstacle_related,
    }


def _scale(desired_v, desired_omega, speed, allow_spin=False):
    """Scale the desired command to an allowed speed, keeping its curvature.

    A rotation in place (desired v ~ 0, the controller's lane search) does
    not move the robot toward anything. It passes unscaled with allow_spin,
    which only the path-clear decision sets; while braking it is stopped.
    """
    desired_v = max(0.0, float(desired_v))
    speed = max(0.0, float(speed))
    if desired_v <= 1e-6:
        return 0.0, float(desired_omega) if allow_spin else 0.0
    fraction = clip(speed / desired_v, 0.0, 1.0)
    return speed, float(desired_omega) * fraction


def _brake_speed(distance_m, stop_distance_m, desired_v, decel):
    room = max(0.0, float(distance_m) - float(stop_distance_m))
    cap = math.sqrt(max(0.0, 2.0 * float(decel) * room))
    return min(max(0.0, desired_v), cap)


def _approach_speed(distance_m, stop_distance_m, brake_distance_m, desired_v, decel):
    """Linear slowdown across the brake region, also limited by constant deceleration."""
    span = max(1e-3, float(brake_distance_m) - float(stop_distance_m))
    fraction = clip((float(distance_m) - float(stop_distance_m)) / span, 0.0, 1.0)
    return min(max(0.0, desired_v) * fraction, _brake_speed(distance_m, stop_distance_m, desired_v, decel))


class StopLatch(object):
    """Hold a stop until the corridor has been clear for `hold_s` seconds."""

    def __init__(self):
        self.held = False
        self.clear_since = None

    def apply(self, now, decision, hold_s):
        if decision["state"] in ("STOP", "FAILSAFE") and decision.get("obstacle_related"):
            self.held = True
            self.clear_since = None
            return decision
        if not self.held:
            return decision
        if decision["state"] == "CLEAR":
            if self.clear_since is None:
                self.clear_since = now
            if (now - self.clear_since) < hold_s:
                held = _stopped(
                    "STOP", "clear-path hysteresis", True,
                    decision["desired_v"], decision["desired_omega"],
                )
                return held
            self.held = False
            self.clear_since = None
            return decision
        self.clear_since = None
        return _stopped(
            "STOP", "stop hysteresis (%s)" % decision["reason"], True,
            decision["desired_v"], decision["desired_omega"],
        )


def evaluate_safety(inputs, params):
    """Return an authorization dict. `inputs` and `params` are plain dicts."""
    desired_v = max(0.0, float(inputs.get("desired_v", 0.0)))
    desired_omega = float(inputs.get("desired_omega", 0.0))
    max_speed = float(params["max_speed"])
    desired_v = min(desired_v, max_speed)

    if not inputs.get("allow_motion", False):
        return _stopped("STOP", "motion disabled", False, desired_v, desired_omega)
    if inputs.get("startup_active", False):
        return _stopped("STOP", "startup hold", False, desired_v, desired_omega)
    if params.get("require_fsm", True):
        allowed = params.get("allowed_fsm_states", ["LANE_FOLLOWING"])
        if inputs.get("fsm_state") not in allowed:
            state = inputs.get("fsm_state")
            reason = "waiting for FSM" if state is None else "FSM is %s" % state
            return _stopped("STOP", reason, False, desired_v, desired_omega)
    if not inputs.get("desired_fresh", False):
        return _stopped("STOP", "pure pursuit command is stale", False, desired_v, desired_omega)

    yolo_available = bool(
        inputs.get("yolo_fresh") and inputs.get("yolo_model_ok") and inputs.get("camera_ok")
    )
    tof_available = bool(inputs.get("tof_valid") and not inputs.get("tof_stale"))
    tof_range = inputs.get("tof_range_m")
    stop_distance = float(params["stop_distance_m"])
    clear_distance = float(params["clear_distance_m"])
    brake_distance = float(params["brake_distance_m"])
    decel = float(params["brake_decel_m_s2"])

    if not yolo_available and not tof_available:
        return _stopped("FAILSAFE", "YOLO and ToF are both unavailable", True, desired_v, desired_omega)

    if tof_available and tof_range is not None and tof_range <= stop_distance:
        return _stopped("STOP", "ToF obstacle inside stop distance", True, desired_v, desired_omega)

    threat = _nearest_corridor_obstacle(inputs.get("obstacles") or [])
    if threat is not None:
        distance = threat.get("distance_m")
        projection_ok = bool(threat.get("projection_valid")) and distance is not None
        if not tof_available:
            if not projection_ok or distance <= brake_distance:
                return _stopped(
                    "STOP", "ToF unavailable and an object is in the corridor",
                    True, desired_v, desired_omega,
                )
            speed, omega = _scale(
                desired_v, desired_omega,
                _approach_speed(distance, stop_distance, brake_distance, desired_v, decel),
            )
            return _moving(
                "BRAKE", "braking for a corridor object without ToF",
                speed, omega, desired_v, desired_omega, True,
            )
        if not projection_ok:
            reference = tof_range if tof_range is not None else stop_distance
            speed, omega = _scale(
                desired_v, desired_omega,
                min(params["uncertain_speed_m_s"], _brake_speed(reference, stop_distance, desired_v, decel)),
            )
            return _moving(
                "BRAKE", "corridor object with uncertain ground projection",
                speed, omega, desired_v, desired_omega, True,
            )
        if distance <= stop_distance:
            return _stopped("STOP", "object in the driving corridor", True, desired_v, desired_omega)
        if distance <= brake_distance:
            speed, omega = _scale(
                desired_v, desired_omega,
                _approach_speed(distance, stop_distance, brake_distance, desired_v, decel),
            )
            return _moving(
                "BRAKE", "braking for an object in the driving corridor",
                speed, omega, desired_v, desired_omega, True,
            )

    if tof_available and tof_range is not None and tof_range < clear_distance:
        speed, omega = _scale(
            desired_v, desired_omega,
            _approach_speed(tof_range, stop_distance, clear_distance, desired_v, decel),
        )
        return _moving(
            "BRAKE", "ToF reading is inside the brake region",
            speed, omega, desired_v, desired_omega, True,
        )

    speed = desired_v
    reason = "path clear"
    if not yolo_available:
        speed *= float(params["yolo_unavailable_speed_scale"])
        reason = "path clear, YOLO unavailable, ToF clear"
    elif not tof_available:
        speed *= float(params["tof_unavailable_speed_scale"])
        reason = "path clear, ToF unavailable, no corridor object"
    speed, omega = _scale(desired_v, desired_omega, speed, allow_spin=True)
    return _moving("CLEAR", reason, speed, omega, desired_v, desired_omega, False)
