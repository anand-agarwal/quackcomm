#!/usr/bin/env python3
"""Record observed message intervals on a live ROS master.

Topics that do not publish during the window are reported as not observed.
No timing value in this script is a default or an estimate.
"""

import os
import sys
import time


def load_subscribers(vehicle):
    import rospy
    from diagnostic_msgs.msg import DiagnosticArray
    from sensor_msgs.msg import CompressedImage, Range

    topics = [
        ("/%s/camera_node/image/compressed" % vehicle, CompressedImage),
        ("/%s/anti_instagram_node/corrected_image/compressed" % vehicle, CompressedImage),
        ("/%s/front_center_tof_driver_node/range" % vehicle, Range),
        ("/%s/purepursuit/yolo_debug_image/compressed" % vehicle, CompressedImage),
        ("/%s/purepursuit/latency" % vehicle, DiagnosticArray),
    ]
    try:
        from purepursuit.msg import DetectionArray, ObstacleStatus, SafetyState
        topics.extend([
            ("/%s/purepursuit/detection_array" % vehicle, DetectionArray),
            ("/%s/purepursuit/obstacle_distance" % vehicle, ObstacleStatus),
            ("/%s/purepursuit/safety_state" % vehicle, SafetyState),
        ])
    except ImportError as exc:
        print("Custom purepursuit messages are not built yet: %s" % exc)
    try:
        from duckietown_msgs.msg import SegmentList, Twist2DStamped
        topics.extend([
            ("/%s/ground_projection_node/lineseglist_out" % vehicle, SegmentList),
            ("/%s/lane_controller_node/car_cmd" % vehicle, Twist2DStamped),
        ])
    except ImportError as exc:
        print("duckietown_msgs is not importable: %s" % exc)
    return topics


def main():
    try:
        import rospy
    except ImportError:
        print("rospy is not installed in this environment. Run this inside the Duckiebot ROS container.")
        return 1
    vehicle = os.environ.get("VEHICLE_NAME", "myrobot2")
    duration = float(os.environ.get("BENCHMARK_SECONDS", "10"))
    rospy.init_node("purepursuit_benchmark", anonymous=True)
    topics = load_subscribers(vehicle)
    stats = {topic: {"count": 0, "last": None, "intervals": [], "ages": []} for topic, _ in topics}

    def callback(msg, topic):
        now = time.time()
        slot = stats[topic]
        slot["count"] += 1
        if slot["last"] is not None:
            slot["intervals"].append(now - slot["last"])
        slot["last"] = now
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is not None and stamp.to_sec() > 0.0:
            slot["ages"].append(max(0.0, rospy.Time.now().to_sec() - stamp.to_sec()))

    for topic, message_type in topics:
        rospy.Subscriber(topic, message_type, callback, callback_args=topic, queue_size=1)
    print("Listening for %.1f s. Keep the robot stationary." % duration)
    time.sleep(duration)
    print("%-72s %8s %12s %12s" % ("topic", "count", "period_ms", "header_age_ms"))
    for topic, _message_type in topics:
        slot = stats[topic]
        if slot["count"] == 0:
            print("%-72s %8s %12s %12s" % (topic, "0", "not observed", "not observed"))
            continue
        period = "n/a"
        age = "n/a"
        if slot["intervals"]:
            period = "%.1f" % (1000.0 * sum(slot["intervals"]) / len(slot["intervals"]))
        if slot["ages"]:
            age = "%.1f" % (1000.0 * sum(slot["ages"]) / len(slot["ages"]))
        print("%-72s %8d %12s %12s" % (topic, slot["count"], period, age))
    print("YOLO decode and inference times are fields inside /%s/purepursuit/latency." % vehicle)
    print("rostopic echo /%s/purepursuit/latency" % vehicle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
