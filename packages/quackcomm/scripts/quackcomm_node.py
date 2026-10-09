#!/usr/bin/env python3

import rospy
from std_msgs.msg import String


def main():
    rospy.init_node("quackcomm_node")

    publisher = rospy.Publisher(
        "~status",
        String,
        queue_size=1,
        latch=True,
    )

    message = String(data="quackcomm is running")
    publisher.publish(message)

    rospy.loginfo(message.data)
    rospy.spin()


if __name__ == "__main__":
    main()
