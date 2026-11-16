#!/usr/bin/env python3
"""Compare live topic types with the types this demo expects.

Run this on a machine that can see the Duckiebot ROS master. It does not
publish commands. A mismatch means the message definition must be checked
before driving.
"""

import os
import sys

EXPECTED = {
    "anti_instagram_node/corrected_image/compressed": "sensor_msgs/CompressedImage",
    "camera_node/image/compressed": "sensor_msgs/CompressedImage",
    "camera_node/camera_info": "sensor_msgs/CameraInfo",
    "ground_projection_node/lineseglist_out": "duckietown_msgs/SegmentList",
    "lane_filter_node/seglist_filtered": "duckietown_msgs/SegmentList",
    "lane_filter_node/lane_pose": "duckietown_msgs/LanePose",
    "front_center_tof_driver_node/range": "sensor_msgs/Range",
    "fsm_node/mode": "duckietown_msgs/FSMState",
    "lane_controller_node/car_cmd": "duckietown_msgs/Twist2DStamped",
    "car_cmd_switch_node/cmd": "duckietown_msgs/Twist2DStamped",
}


def main():
    try:
        import rosgraph
    except ImportError:
        print("rosgraph is not installed in this environment. Run this inside the Duckiebot ROS container.")
        return 1
    vehicle = os.environ.get("VEHICLE_NAME", "myrobot2")
    if len(sys.argv) > 1:
        vehicle = sys.argv[1]
    master = rosgraph.Master("/purepursuit_validate_topics")
    try:
        published = dict(master.getTopicTypes())
    except Exception as exc:
        print("Could not contact the ROS master: %s" % exc)
        return 1
    failures = 0
    for suffix, expected in EXPECTED.items():
        topic = "/%s/%s" % (vehicle, suffix)
        actual = published.get(topic)
        if actual is None:
            print("MISSING  %s" % topic)
            failures += 1
        elif actual != expected:
            print("MISMATCH %s expected %s observed %s" % (topic, expected, actual))
            failures += 1
        else:
            print("OK       %s [%s]" % (topic, actual))
    if failures:
        print("%d topic check(s) failed. Do not enable motion until these match." % failures)
        return 1
    print("All expected topics match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
