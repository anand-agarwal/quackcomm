#!/usr/bin/env python3
"""Start or stop lane following without the keyboard-control window.

    rosrun purepursuit autopilot.py start   # FSM -> LANE_FOLLOWING
    rosrun purepursuit autopilot.py stop    # FSM -> NORMAL_JOYSTICK_CONTROL
    rosrun purepursuit autopilot.py status

This calls fsm_node/set_state, the same state change the keyboard's `a` and `s`
keys cause. The safety supervisor authorises motion only in LANE_FOLLOWING.
"""

import os
import sys

import rospy

STATES = {"start": "LANE_FOLLOWING", "stop": "NORMAL_JOYSTICK_CONTROL"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("start", "stop", "status"):
        print(__doc__)
        return 2
    action = sys.argv[1]
    vehicle = os.environ.get("VEHICLE_NAME", "duckiebot")
    if len(sys.argv) > 2:
        vehicle = sys.argv[2]
    rospy.init_node("purepursuit_autopilot_cli", anonymous=True, disable_signals=True)
    from duckietown_msgs.msg import FSMState
    from duckietown_msgs.srv import SetFSMState

    mode_topic = "/%s/fsm_node/mode" % vehicle
    if action != "status":
        service = "/%s/fsm_node/set_state" % vehicle
        rospy.wait_for_service(service, timeout=5.0)
        rospy.ServiceProxy(service, SetFSMState)(STATES[action])
    state = rospy.wait_for_message(mode_topic, FSMState, timeout=5.0).state
    print("FSM state: %s" % state)
    if action != "status" and state != STATES[action]:
        print("FSM did not change to %s" % STATES[action])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
