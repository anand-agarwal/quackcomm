#!/usr/bin/env python3
"""Authorize the only chassis command published by this demo."""

import os
import threading

import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_srvs.srv import SetBool

from pp_core.safety import StopLatch, evaluate_safety
from pp_core.topics import (
    DETECTION_ARRAY_SUFFIX,
    DESIRED_COMMAND_SUFFIX,
    EMERGENCY_STOP_SUFFIX,
    FINAL_COMMAND_SUFFIX,
    FSM_MODE_SUFFIX,
    LANE_POSE_SUFFIX,
    LATENCY_SUFFIX,
    OBSTACLE_DISTANCE_SUFFIX,
    SAFETY_STATE_SUFFIX,
    WHEELS_COMMAND_SUFFIX,
    join,
)
from purepursuit.msg import DetectionArray, ObstacleStatus, SafetyState


class SafetySupervisorNode(object):
    def __init__(self):
        self.vehicle = rospy.get_param("~vehicle_name", os.environ.get("VEHICLE_NAME", "duckiebot"))
        self.final_topic = rospy.get_param("~final_cmd_topic", join(self.vehicle, FINAL_COMMAND_SUFFIX))
        if self.final_topic.rstrip("/").endswith(WHEELS_COMMAND_SUFFIX):
            raise RuntimeError("Refusing to publish wheel commands on %s" % self.final_topic)
        if self.final_topic.rstrip("/").endswith(EMERGENCY_STOP_SUFFIX):
            raise RuntimeError("Refusing to publish the emergency-stop topic")
        self.params = {
            "max_speed": float(rospy.get_param("~max_speed", 0.10)),
            "stop_distance_m": float(rospy.get_param("~stop_distance_m", 0.22)),
            "clear_distance_m": float(rospy.get_param("~clear_distance_m", 0.32)),
            "brake_distance_m": float(rospy.get_param("~brake_distance_m", 0.70)),
            "brake_decel_m_s2": float(rospy.get_param("~brake_decel_m_s2", 0.40)),
            "uncertain_speed_m_s": float(rospy.get_param("~uncertain_speed_m_s", 0.04)),
            "yolo_unavailable_speed_scale": float(rospy.get_param("~yolo_unavailable_speed_scale", 0.5)),
            "tof_unavailable_speed_scale": float(rospy.get_param("~tof_unavailable_speed_scale", 0.4)),
            "require_fsm": bool(rospy.get_param("~require_fsm", True)),
            "allowed_fsm_states": list(rospy.get_param("~allowed_fsm_states", ["LANE_FOLLOWING"])),
        }
        self.tof_timeout_s = float(rospy.get_param("~tof_timeout_s", 0.30))
        self.yolo_timeout_s = float(rospy.get_param("~yolo_timeout_s", 0.8))
        self.command_timeout_s = float(rospy.get_param("~command_timeout_s", 0.4))
        self.clear_hold_s = float(rospy.get_param("~clear_hold_s", 0.45))
        self.startup_hold_s = float(rospy.get_param("~startup_hold_s", 2.0))
        self.disable_stock_controller = bool(rospy.get_param("~disable_stock_lane_controller", True))
        self.started_at = rospy.Time.now()
        self.latch = StopLatch()
        self.detections = None
        self.tof = None
        self.desired = None
        self.fsm_state = None
        from duckietown_msgs.msg import FSMState, Twist2DStamped
        self._twist_type = Twist2DStamped
        self.command_publisher = rospy.Publisher(self.final_topic, Twist2DStamped, queue_size=1)
        self.state_publisher = rospy.Publisher(
            join(self.vehicle, SAFETY_STATE_SUFFIX), SafetyState, queue_size=1,
        )
        self.latency_publisher = rospy.Publisher(
            join(self.vehicle, LATENCY_SUFFIX), DiagnosticArray, queue_size=1,
        )
        # dt-core nodes are DTROS nodes. Their on/off switch is the std_srvs/SetBool
        # service ~switch, which fsm_node calls on every state change. When it is
        # off, DTROS drops everything the node publishes. The FSM turns the stock
        # lane_controller_node on when it enters LANE_FOLLOWING, so it is turned off
        # again after every FSM change and re-checked once a second.
        self.stock_nodes = list(rospy.get_param("~stock_nodes_to_disable", ["lane_controller_node"]))
        self.switch_period_s = float(rospy.get_param("~stock_switch_period_s", 1.0))
        # The perception half of dt-core must stay on. On ente, lane_filter_node
        # starts switched off and fsm_node does not always switch it on, which
        # leaves no lane_pose and Pure Pursuit reports "lane unavailable". When
        # lane_pose has been silent for lane_pose_stale_s, switch these back on.
        self.enable_nodes = list(rospy.get_param(
            "~stock_nodes_to_enable", ["lane_filter_node", "line_detector_node", "ground_projection_node"],
        ))
        self.lane_pose_stale_s = float(rospy.get_param("~lane_pose_stale_s", 1.0))
        self.lane_pose_time = None
        self._switch_wakeup = threading.Event()
        self._switch_ok = {}
        if (self.disable_stock_controller and self.stock_nodes) or self.enable_nodes:
            thread = threading.Thread(target=self._stock_nodes_loop)
            thread.daemon = True
            thread.start()
        rospy.Subscriber(join(self.vehicle, DETECTION_ARRAY_SUFFIX), DetectionArray, self._on_detections, queue_size=1)
        rospy.Subscriber(join(self.vehicle, OBSTACLE_DISTANCE_SUFFIX), ObstacleStatus, self._on_tof, queue_size=1)
        rospy.Subscriber(join(self.vehicle, DESIRED_COMMAND_SUFFIX), Twist2DStamped, self._on_desired, queue_size=1)
        rospy.Subscriber(join(self.vehicle, FSM_MODE_SUFFIX), FSMState, self._on_fsm, queue_size=1)
        from duckietown_msgs.msg import LanePose
        rospy.Subscriber(join(self.vehicle, LANE_POSE_SUFFIX), LanePose, self._on_lane_pose, queue_size=1)
        rospy.Timer(rospy.Duration(0.05), self._supervise)
        rospy.on_shutdown(lambda: self._publish_command(0.0, 0.0))
        rospy.loginfo("Safety supervisor is the only demo publisher on %s", self.final_topic)
        rospy.loginfo("Emergency stop %s is left to the wheel driver.", join(self.vehicle, EMERGENCY_STOP_SUFFIX))

    def _on_detections(self, msg):
        self.detections = msg

    def _on_tof(self, msg):
        self.tof = msg

    def _on_desired(self, msg):
        self.desired = msg

    def _on_lane_pose(self, _msg):
        self.lane_pose_time = rospy.Time.now()

    def _on_fsm(self, msg):
        state = getattr(msg, "state", None)
        if state != self.fsm_state:
            # fsm_node has just switched its active nodes back on. Switch ours off.
            self._switch_wakeup.set()
        self.fsm_state = state

    def _stock_nodes_loop(self):
        # Runs on its own thread so a slow or missing service never delays the
        # 20 Hz supervision timer.
        while not rospy.is_shutdown():
            if self.disable_stock_controller:
                for node in self.stock_nodes:
                    self._switch_off(node)
            if self.enable_nodes and self._lane_pose_stale():
                for node in self.enable_nodes:
                    self._switch_on(node)
            self._switch_wakeup.wait(self.switch_period_s)
            self._switch_wakeup.clear()

    def _lane_pose_stale(self):
        if self.lane_pose_time is None:
            return (rospy.Time.now() - self.started_at).to_sec() > self.lane_pose_stale_s
        return (rospy.Time.now() - self.lane_pose_time).to_sec() > self.lane_pose_stale_s

    def _switch_on(self, node):
        service = join(self.vehicle, "%s/switch" % node)
        key = "on:" + node
        try:
            rospy.wait_for_service(service, timeout=0.5)
            reply = rospy.ServiceProxy(service, SetBool)(True)
        except Exception as exc:
            if self._switch_ok.get(key, True):
                rospy.logwarn("No lane_pose and could not switch on %s: %s. Retrying.", service, exc)
            self._switch_ok[key] = False
            return
        self._switch_ok[key] = True
        message = getattr(reply, "message", "")
        if "[off] to [on]" in message:
            rospy.logwarn("No lane_pose: %s was switched off. Switched it on.", node)

    def _switch_off(self, node):
        service = join(self.vehicle, "%s/switch" % node)
        try:
            rospy.wait_for_service(service, timeout=0.5)
            rospy.ServiceProxy(service, SetBool)(False)
        except Exception as exc:
            if self._switch_ok.get(node, True):
                rospy.logwarn("Could not switch off %s: %s. Retrying.", service, exc)
            self._switch_ok[node] = False
            return
        if not self._switch_ok.get(node, False):
            if node == "lane_controller_node":
                rospy.loginfo("Switched off %s. This demo now owns lane_controller_node/car_cmd.", service)
            else:
                rospy.loginfo("Switched off %s (replaced by this demo).", service)
        self._switch_ok[node] = True

    def _age(self, stamp):
        if stamp is None:
            return None
        return (rospy.Time.now() - stamp).to_sec()

    def _supervise(self, _event):
        cycle_started = rospy.Time.now()
        now = rospy.Time.now().to_sec()
        yolo_age = None if self.detections is None else self._age(self.detections.header.stamp)
        command_age = None if self.desired is None else self._age(self.desired.header.stamp)
        tof_age = None if self.tof is None else self._age(self.tof.header.stamp)
        obstacles = []
        if self.detections is not None:
            for detection in self.detections.detections:
                obstacles.append({
                    "in_corridor": bool(detection.in_corridor),
                    "distance_m": None if detection.distance_m < 0.0 else float(detection.distance_m),
                    "projection_valid": bool(detection.projection_valid),
                    "confirmed": bool(detection.confirmed),
                })
        inputs = {
            "allow_motion": bool(rospy.get_param("~allow_motion", False)),
            "startup_active": (rospy.Time.now() - self.started_at).to_sec() < self.startup_hold_s,
            "fsm_state": self.fsm_state,
            "desired_v": 0.0 if self.desired is None else float(self.desired.v),
            "desired_omega": 0.0 if self.desired is None else float(self.desired.omega),
            "desired_fresh": command_age is not None and command_age <= self.command_timeout_s,
            "tof_valid": bool(self.tof.valid) if self.tof is not None else False,
            "tof_stale": True if self.tof is None else bool(self.tof.stale or (tof_age is not None and tof_age > self.tof_timeout_s)),
            "tof_range_m": None if self.tof is None or self.tof.range_m < 0.0 else float(self.tof.range_m),
            "yolo_fresh": yolo_age is not None and yolo_age <= self.yolo_timeout_s,
            "yolo_model_ok": bool(self.detections.model_loaded) if self.detections is not None else False,
            "camera_ok": bool(self.detections.camera_ok) if self.detections is not None else False,
            "obstacles": obstacles,
        }
        decision = self.latch.apply(now, evaluate_safety(inputs, self.params), self.clear_hold_s)
        self._publish_command(decision["v"], decision["omega"])
        self._publish_state(decision, inputs)
        fusion_ms = (rospy.Time.now() - cycle_started).to_sec() * 1000.0
        self._publish_latency(yolo_age, tof_age, command_age, fusion_ms)

    def _publish_command(self, v_value, omega_value):
        msg = self._twist_type()
        msg.header.stamp = rospy.Time.now()
        msg.v = float(v_value)
        msg.omega = float(omega_value)
        self.command_publisher.publish(msg)

    def _publish_state(self, decision, inputs):
        msg = SafetyState()
        msg.header.stamp = rospy.Time.now()
        msg.state = decision["state"]
        msg.reason = decision["reason"]
        msg.authorized_v = float(decision["v"])
        msg.authorized_omega = float(decision["omega"])
        msg.desired_v = float(decision["desired_v"])
        msg.desired_omega = float(decision["desired_omega"])
        msg.yolo_fresh = bool(inputs["yolo_fresh"] and inputs["yolo_model_ok"] and inputs["camera_ok"])
        msg.tof_fresh = bool(inputs["tof_valid"] and not inputs["tof_stale"])
        msg.control_fresh = bool(inputs["desired_fresh"])
        msg.motion_allowed = bool(inputs["allow_motion"] and not inputs["startup_active"])
        msg.fsm_state = "" if inputs["fsm_state"] is None else str(inputs["fsm_state"])
        self.state_publisher.publish(msg)

    def _publish_latency(self, yolo_age, tof_age, command_age, fusion_ms):
        status = DiagnosticStatus()
        status.name = "purepursuit"
        status.level = DiagnosticStatus.OK
        status.message = "pipeline ages"
        values = {
            "yolo_age_ms": -1.0 if yolo_age is None else yolo_age * 1000.0,
            "yolo_decode_ms": 0.0 if self.detections is None else float(self.detections.decode_latency_ms),
            "yolo_inference_ms": 0.0 if self.detections is None else float(self.detections.inference_latency_ms),
            "tof_age_ms": -1.0 if tof_age is None else tof_age * 1000.0,
            "command_age_ms": -1.0 if command_age is None else command_age * 1000.0,
            "fusion_ms": fusion_ms,
        }
        for key, value in values.items():
            status.values.append(KeyValue(key=key, value="%.3f" % value))
        msg = DiagnosticArray()
        msg.header.stamp = rospy.Time.now()
        msg.status.append(status)
        self.latency_publisher.publish(msg)


def main():
    rospy.init_node("safety_supervisor_node")
    SafetySupervisorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
