#!/usr/bin/env python3
"""Forward ToF obstacle monitor.

This node can request a stop on its own. The safety supervisor does not wait
for a YOLO detection when the range is inside the stop distance.
"""

import os

import rospy
from sensor_msgs.msg import Range

from pp_core.tof import RangeFilter, assess_range
from pp_core.topics import OBSTACLE_DISTANCE_SUFFIX, TOF_RANGE_SUFFIX, join
from purepursuit.msg import ObstacleStatus


class TofObstacleNode(object):
    def __init__(self):
        self.vehicle = rospy.get_param("~vehicle_name", os.environ.get("VEHICLE_NAME", "duckiebot"))
        self.range_topic = rospy.get_param("~range_topic", join(self.vehicle, TOF_RANGE_SUFFIX))
        self.stop_distance_m = float(rospy.get_param("~stop_distance_m", 0.22))
        self.clear_distance_m = float(rospy.get_param("~clear_distance_m", 0.32))
        self.timeout_s = float(rospy.get_param("~timeout_s", 0.30))
        self.inf_means_clear = bool(rospy.get_param("~inf_means_clear", True))
        self.zero_means_no_return = bool(rospy.get_param("~zero_means_no_return", True))
        self.filter = RangeFilter(int(rospy.get_param("~median_window", 3)))
        self.publisher = rospy.Publisher(
            join(self.vehicle, OBSTACLE_DISTANCE_SUFFIX), ObstacleStatus, queue_size=1,
        )
        self.last_msg_time = None
        self.last_status = None
        rospy.Subscriber(self.range_topic, Range, self._on_range, queue_size=1)
        rospy.Timer(rospy.Duration(0.05), self._watchdog)
        rospy.loginfo("ToF obstacle node listening on %s", self.range_topic)

    def _on_range(self, msg):
        now = rospy.Time.now()
        self.last_msg_time = now
        stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0.0 else now
        age = (now - stamp).to_sec()
        numeric = float(msg.range)
        if self.zero_means_no_return and numeric == 0.0:
            # The DB21J ente driver publishes exactly 0.0 when the beam gets no
            # return (seen on myrobot: 140 of 150 samples were 0.0, the rest 100.0).
            # A real reflection is never exactly zero, so map it to +inf.
            numeric = float("inf")
        if numeric == numeric and numeric != float("inf") and numeric != float("-inf") and numeric >= 0.0:
            numeric = self.filter.push(numeric)
        status = assess_range(
            numeric, float(msg.min_range), float(msg.max_range),
            self.stop_distance_m, self.clear_distance_m,
            age, self.timeout_s, self.inf_means_clear,
        )
        status["min_range_m"] = float(msg.min_range)
        status["max_range_m"] = float(msg.max_range)
        self.last_status = status
        self._publish(status, stamp)

    def _watchdog(self, _event):
        now = rospy.Time.now()
        if self.last_msg_time is None:
            age = None
        else:
            age = (now - self.last_msg_time).to_sec()
        if age is None or age > self.timeout_s:
            previous = self.last_status or {}
            status = assess_range(
                previous.get("range_m"),
                previous.get("min_range_m", 0.0),
                previous.get("max_range_m", 0.0),
                self.stop_distance_m, self.clear_distance_m,
                age if age is not None else self.timeout_s + 1.0,
                self.timeout_s, self.inf_means_clear,
            )
            if self.last_msg_time is None:
                status["reason"] = "no ToF message yet"
                status["valid"] = False
                status["stale"] = True
                status["clear"] = False
            self._publish(status, now)

    def _publish(self, status, stamp):
        msg = ObstacleStatus()
        msg.header.stamp = stamp
        msg.range_m = -1.0 if status["range_m"] is None else float(status["range_m"])
        msg.min_range_m = float(status.get("min_range_m", 0.0))
        msg.max_range_m = float(status.get("max_range_m", 0.0))
        msg.valid = bool(status["valid"])
        msg.stale = bool(status["stale"])
        msg.dangerously_close = bool(status["dangerously_close"])
        msg.clear = bool(status["clear"])
        msg.reason = status["reason"]
        self.publisher.publish(msg)


def main():
    rospy.init_node("tof_obstacle_node")
    TofObstacleNode()
    rospy.spin()


if __name__ == "__main__":
    main()
