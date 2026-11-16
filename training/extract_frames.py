#!/usr/bin/env python3
"""Save camera frames from a Duckiebot rosbag as JPEG files for labelling.

Uses the pure-Python `rosbags` reader, so ROS does not need to be installed.

Record on the robot (from a container with ROS, e.g. dts-run-quackcomm):
    rosbag record -O /data/track1.bag /myrobot2/camera_node/image/compressed
Copy it off:
    scp duckie@myrobot2.local:/data/track1.bag .
Extract about 2 frames per second:
    python3 training/extract_frames.py track1.bag frames/track1 --fps 2
"""

import argparse
import os
import sys

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag")
    parser.add_argument("output")
    parser.add_argument("--topic", default="", help="default: the first */camera_node/image/compressed")
    parser.add_argument("--fps", type=float, default=2.0, help="frames kept per second of recording")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    period_ns = int(1e9 / max(args.fps, 0.01))
    prefix = os.path.splitext(os.path.basename(args.bag))[0]
    typestore = get_typestore(Stores.ROS1_NOETIC)
    saved = 0
    with AnyReader([os.path.abspath(args.bag)], default_typestore=typestore) as reader:
        topics = [c for c in reader.connections if c.msgtype == "sensor_msgs/msg/CompressedImage"]
        if args.topic:
            topics = [c for c in topics if c.topic == args.topic]
        else:
            topics = [c for c in topics if c.topic.endswith("camera_node/image/compressed")][:1]
        if not topics:
            print("No compressed camera topic found. Topics: %s" % sorted({c.topic for c in reader.connections}))
            return 1
        print("Reading %s" % topics[0].topic)
        last = None
        for connection, timestamp, raw in reader.messages(connections=topics):
            if last is not None and timestamp - last < period_ns:
                continue
            last = timestamp
            msg = reader.deserialize(raw, connection.msgtype)
            path = os.path.join(args.output, "%s_%019d.jpg" % (prefix, timestamp))
            with open(path, "wb") as handle:
                handle.write(bytes(msg.data))
            saved += 1
    print("Saved %d frames to %s" % (saved, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
