#!/usr/bin/env python3
"""Click four measured ground-plane rectangle corners in a camera frame.

Click order: far-left, far-right, near-right, near-left.
The physical rectangle must match bev_width_m x bev_length_m in controller.yaml.
"""
import os
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage

POINT_NAMES = ["far-left", "far-right", "near-right", "near-left"]


def main():
    rospy.init_node("calibrate_bev", anonymous=True)
    vehicle = rospy.get_param("~vehicle_name", os.environ.get("VEHICLE_NAME", "duckiebot"))
    topic = rospy.get_param(
        "~image_topic", f"/{vehicle}/camera_node/image/compressed"
    )
    output = os.path.expanduser(rospy.get_param("~output", "~/bev_points.yaml"))
    msg = rospy.wait_for_message(topic, CompressedImage, timeout=15.0)
    image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not decode camera frame")

    points = []
    window = "BEV calibration: far-left, far-right, near-right, near-left"
    display = image.copy()

    def on_mouse(event, x, y, _flags, _param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))
            display = image.copy()
            for i, point in enumerate(points):
                p = (int(point[0]), int(point[1]))
                cv2.circle(display, p, 5, (0, 0, 255), -1)
                cv2.putText(display, POINT_NAMES[i], (p[0] + 6, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while not rospy.is_shutdown() and len(points) < 4:
        cv2.imshow(window, display)
        if cv2.waitKey(30) & 0xFF == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Calibration cancelled")
    cv2.destroyAllWindows()

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    flat = [v for point in points for v in point]
    with open(output, "w") as f:
        f.write("src_points: [" + ", ".join(f"{v:.1f}" for v in flat) + "]\n")
    print(f"Saved {output}")
    print("src_points: " + str(flat))


if __name__ == "__main__":
    main()
