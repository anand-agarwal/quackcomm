#!/usr/bin/env python3
"""Pure Pursuit lane follower.

Lane sources, in order of preference (``~primary_source`` picks the first):

- ``vision``: purepursuit/lane_estimate from lane_detector_node (our own
  detector), Kalman-tracked and predicted forward over the camera delay with
  the measured robot velocity, then Pure Pursuit to the point of the lane's
  centre *arc* at a speed-dependent lookahead (shortened in tight turns).
  On an arc that commands the road's own curvature, so turns are fed forward.
- ``lane_pose``: dt-core's lane_filter_node/lane_pose (straight-lane target).
- ``ground_segments``: dt-core's ground-projected segments.

When no source has a lane, pp_core.recovery decides: coast (or, in a turn,
finish the turn) on the tracker's prediction, then search by rotating toward
where the lane was last seen, then stop. ControllerStatus.status names the
state: tracking, coast, turn_memory, search, lost, waiting.

This node publishes a desired command only. The safety supervisor is the
publisher on lane_controller_node/car_cmd.
"""

import os

import rospy
from std_msgs.msg import Float32

from pp_core.lane_tracking import LaneTracker
from pp_core.pure_pursuit import (
    centerline_from_segments,
    command_from_target,
    lane_confidence,
    point_at_lookahead,
    target_from_lane_pose,
)
from pp_core.recovery import LaneLossRecovery
from pp_core.vision_pursuit import VisionPursuit
from pp_core.topics import (
    CONTROLLER_STATUS_SUFFIX,
    DESIRED_COMMAND_SUFFIX,
    FILTERED_SEGMENTS_SUFFIX,
    GROUND_SEGMENTS_SUFFIX,
    LANE_CONFIDENCE_SUFFIX,
    LANE_ESTIMATE_SUFFIX,
    LANE_POSE_SUFFIX,
    VELOCITY_SUFFIX,
    join,
)
from purepursuit.msg import ControllerStatus, LaneEstimate


class PurePursuitControllerNode(object):
    def __init__(self):
        self.vehicle = rospy.get_param("~vehicle_name", os.environ.get("VEHICLE_NAME", "duckiebot"))
        self.limits = {
            "max_speed": float(rospy.get_param("~max_speed", 0.10)),
            "min_speed": float(rospy.get_param("~min_speed", 0.035)),
            "max_omega": float(rospy.get_param("~max_omega", 1.8)),
            "max_lateral_accel": float(rospy.get_param("~max_lateral_accel", 0.18)),
            "curvature_slowdown_gain": float(rospy.get_param("~curvature_slowdown_gain", 0.65)),
        }
        self.lookahead_m = float(rospy.get_param("~lookahead_m", 0.42))
        self.lane_width_m = float(rospy.get_param("~lane_width_m", 0.23))
        self.min_confidence = float(rospy.get_param("~min_confidence", 0.25))
        self.pose_timeout_s = float(rospy.get_param("~pose_timeout_s", 0.4))
        self.d_sign = float(rospy.get_param("~lane_pose_d_sign", 1.0))
        self.phi_sign = float(rospy.get_param("~lane_pose_phi_sign", 1.0))
        self.x_min_m = float(rospy.get_param("~x_min_m", 0.05))
        self.x_max_m = float(rospy.get_param("~x_max_m", 1.2))
        # lane_pose is dt-core's histogram-filtered lane estimate. The raw ground
        # segments also contain the far lane's lines, so they are the fallback.
        self.primary_source = str(rospy.get_param("~primary_source", "lane_pose"))
        self.max_target_lateral_m = float(rospy.get_param("~max_target_lateral_m", 0.20))
        # Vision source: tracker (delay compensation), arc Pure Pursuit and
        # lane-loss recovery, in pp_core.vision_pursuit (no ROS, tested closed loop).
        tracker = LaneTracker(
            process_noise=rospy.get_param("~tracker_process_noise", [0.002, 0.05, 2.0]),
            measurement_noise=rospy.get_param("~tracker_measurement_noise", [0.01, 0.05, 0.6]),
            reset_after_s=float(rospy.get_param("~tracker_reset_s", 1.0)),
            max_kappa=float(rospy.get_param("~max_curvature", 10.0)),
        )
        recovery = LaneLossRecovery(
            coast_s=float(rospy.get_param("~coast_s", 0.6)),
            turn_kappa=float(rospy.get_param("~turn_curvature", 1.5)),
            turn_angle_rad=float(rospy.get_param("~turn_angle_rad", 1.5708)),
            max_turn_memory_s=float(rospy.get_param("~max_turn_memory_s", 3.0)),
            search_enabled=bool(rospy.get_param("~search_enabled", True)),
            search_omega=float(rospy.get_param("~search_omega", 2.0)),
            search_forward_speed=float(rospy.get_param("~search_forward_speed", 0.0)),
            search_angle_rad=float(rospy.get_param("~search_angle_rad", 0.6)),
            search_max_angle_rad=float(rospy.get_param("~search_max_angle_rad", 1.5)),
            search_timeout_s=float(rospy.get_param("~search_timeout_s", 6.0)),
        )
        self.pursuit = VisionPursuit(
            self.limits,
            lookahead={
                "base_m": float(rospy.get_param("~lookahead_base_m", 0.20)),
                "gain_s": float(rospy.get_param("~lookahead_gain_s", 0.6)),
                "min_m": float(rospy.get_param("~lookahead_min_m", 0.18)),
                "max_m": float(rospy.get_param("~lookahead_max_m", 0.35)),
            },
            tracker=tracker,
            recovery=recovery,
            vision_timeout_s=float(rospy.get_param("~vision_timeout_s", 0.5)),
            vision_min_confidence=float(rospy.get_param("~vision_min_confidence", 0.2)),
            turn_radius_factor=float(rospy.get_param("~turn_lookahead_radius_factor", 1.2)),
            turn_lookahead_min_m=float(rospy.get_param("~turn_lookahead_min_m", 0.12)),
            coast_speed_scale=float(rospy.get_param("~coast_speed_scale", 1.0)),
        )
        self._last_control = None
        self.segments = []
        self.segment_time = None
        self.lane_pose = None
        self.lane_pose_time = None
        self.publisher = rospy.Publisher(
            join(self.vehicle, DESIRED_COMMAND_SUFFIX), self._twist_type(), queue_size=1,
        )
        self.status_publisher = rospy.Publisher(
            join(self.vehicle, CONTROLLER_STATUS_SUFFIX), ControllerStatus, queue_size=1,
        )
        self.confidence_publisher = rospy.Publisher(
            join(self.vehicle, LANE_CONFIDENCE_SUFFIX), Float32, queue_size=1,
        )
        self._subscribe_lane_topics()
        rospy.Timer(rospy.Duration(0.05), self._control)
        rospy.on_shutdown(lambda: self._publish_desired(0.0, 0.0))
        rospy.loginfo("Pure Pursuit desired command on %s", join(self.vehicle, DESIRED_COMMAND_SUFFIX))

    def _twist_type(self):
        from duckietown_msgs.msg import Twist2DStamped
        return Twist2DStamped

    def _subscribe_lane_topics(self):
        from duckietown_msgs.msg import LanePose, SegmentList
        rospy.Subscriber(join(self.vehicle, GROUND_SEGMENTS_SUFFIX), SegmentList, self._on_segments, queue_size=1)
        rospy.Subscriber(join(self.vehicle, FILTERED_SEGMENTS_SUFFIX), SegmentList, self._on_filtered_segments, queue_size=1)
        rospy.Subscriber(join(self.vehicle, LANE_POSE_SUFFIX), LanePose, self._on_pose, queue_size=1)
        from duckietown_msgs.msg import Twist2DStamped
        rospy.Subscriber(join(self.vehicle, LANE_ESTIMATE_SUFFIX), LaneEstimate, self._on_lane_estimate, queue_size=5)
        # Executed (v, omega) from kinematics_node; it includes joystick commands.
        rospy.Subscriber(join(self.vehicle, VELOCITY_SUFFIX), Twist2DStamped, self._on_velocity, queue_size=10)

    def _store_segments(self, msg):
        try:
            stored = []
            for segment in msg.segments:
                stored.append((int(segment.color), [(point.x, point.y) for point in segment.points]))
        except Exception as exc:
            rospy.logerr_throttle(2.0, "Could not read lane segments: %s", exc)
            return
        self.segments = stored
        self.segment_time = rospy.Time.now()

    def _on_segments(self, msg):
        self._store_segments(msg)

    def _on_filtered_segments(self, msg):
        # Ground projection is preferred. Use the filtered list only as a fallback.
        if self.segment_time is None or (rospy.Time.now() - self.segment_time).to_sec() > self.pose_timeout_s:
            self._store_segments(msg)

    def _on_velocity(self, msg):
        now = rospy.Time.now().to_sec()
        stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0 else now
        self.pursuit.on_velocity(stamp, now, msg.v, msg.omega)

    def _on_lane_estimate(self, msg):
        self.pursuit.on_estimate(msg.header.stamp.to_sec(), rospy.Time.now().to_sec(), msg.valid,
                                 msg.d, msg.phi, msg.curvature, msg.confidence)

    def _command_from_vision(self):
        return self.pursuit.vision_command(rospy.Time.now().to_sec())

    def _command_from(self, target_method):
        """Wrap a target source (lane_pose, segments) as a command source."""
        target, confidence = target_method()
        command = command_from_target(target, self.limits) if target is not None else None
        return command, confidence

    def _fresh_within(self, stamp, timeout_s):
        return stamp is not None and (rospy.Time.now() - stamp).to_sec() <= timeout_s

    def _on_pose(self, msg):
        self.lane_pose = msg
        self.lane_pose_time = rospy.Time.now()

    def _fresh(self, stamp):
        return stamp is not None and (rospy.Time.now() - stamp).to_sec() <= self.pose_timeout_s

    def _target_from_segments(self):
        if not self._fresh(self.segment_time):
            return None, 0.0
        centers = centerline_from_segments(
            self.segments, self.lane_width_m,
            x_min_m=self.x_min_m, x_max_m=self.x_max_m,
        )
        expected = max(1, int((min(self.lookahead_m, self.x_max_m) - self.x_min_m) / 0.05))
        confidence = lane_confidence(len(centers), expected)
        if confidence < self.min_confidence:
            return None, confidence
        target = point_at_lookahead(centers, self.lookahead_m)
        # The lane centre cannot be more than about one lane width to the side
        # at the lookahead distance. A farther point comes from another lane's
        # lines in the raw segment list.
        if target is None or abs(target[1]) > self.max_target_lateral_m:
            return None, confidence
        return target, confidence

    def _target_from_lane_pose(self):
        if not self._fresh(self.lane_pose_time) or self.lane_pose is None:
            return None, 0.0
        if not getattr(self.lane_pose, "in_lane", True):
            return None, 0.0
        target = target_from_lane_pose(
            self.lane_pose.d, self.lane_pose.phi, self.lookahead_m, self.d_sign, self.phi_sign,
        )
        return target, 1.0 if target is not None else 0.0

    def _control(self, _event):
        available = {
            "vision": self._command_from_vision,
            "lane_pose": lambda: self._command_from(self._target_from_lane_pose),
            "ground_segments": lambda: self._command_from(self._target_from_segments),
        }
        order = [self.primary_source] + [name for name in ("vision", "lane_pose", "ground_segments")
                                         if name != self.primary_source]
        sources = [(name, available[name]) for name in order if name in available]
        command = None
        source = "none"
        confidence = 0.0
        for name, method in sources:
            candidate, candidate_confidence = method()
            if candidate is not None:
                command, source, confidence = candidate, name, candidate_confidence
                break
            confidence = max(confidence, candidate_confidence)
        now = rospy.Time.now().to_sec()
        dt = 0.05 if self._last_control is None else min(0.2, max(0.0, now - self._last_control))
        self._last_control = now
        if command is not None:
            if source == "vision":
                self.pursuit.tracking(dt, command)
            self._drive(command["v"], command["omega"], confidence, command, source, "tracking")
        else:
            # No lane from any source: coast / finish the turn / search / stop.
            v_value, omega_value, command, status = self.pursuit.recover(now, dt)
            self._drive(v_value, omega_value, confidence, command,
                        "prediction" if command is not None else "none", status)
        self.confidence_publisher.publish(Float32(data=float(confidence)))

    def _drive(self, v_value, omega_value, confidence, command, source, status):
        self.pursuit.commanded(v_value, omega_value)
        self._publish_desired(v_value, omega_value)
        if command is None:
            self._publish_status(confidence, 0.0, 0.0, 0.0, v_value, source, status, omega_value)
        else:
            self._publish_status(confidence, command["curvature"], command["forward_m"],
                                 command["lateral_m"], v_value, source, status, omega_value)

    def _publish_desired(self, v_value, omega_value):
        msg = self._twist_type()()
        msg.header.stamp = rospy.Time.now()
        msg.v = float(v_value)
        msg.omega = float(omega_value)
        self.publisher.publish(msg)

    def _publish_status(self, confidence, curvature, forward_m, lateral_m, v_value, source, status, omega_value=0.0):
        msg = ControllerStatus()
        msg.header.stamp = rospy.Time.now()
        msg.confidence = float(confidence)
        msg.curvature = float(curvature)
        msg.lookahead_x = float(forward_m)
        msg.lookahead_y = float(lateral_m)
        msg.desired_v = float(v_value)
        msg.desired_omega = float(omega_value)
        msg.source = source
        msg.status = status
        self.status_publisher.publish(msg)


def main():
    rospy.init_node("purepursuit_controller_node")
    PurePursuitControllerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
