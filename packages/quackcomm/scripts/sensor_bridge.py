#!/usr/bin/env python3

import os
import time
import logging

import roslibpy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# Robot connections
BOT1_HOST = os.getenv("MYROBOT_HOST", "myrobot.local")
BOT2_HOST = os.getenv("MYROBOT2_HOST", "myrobot2.local")

BOT1_PORT = int(os.getenv("MYROBOT_ROSBRIDGE_PORT", "9001"))
BOT2_PORT = int(os.getenv("MYROBOT2_ROSBRIDGE_PORT", "9001"))

# Master encoder inputs
LEFT_ENCODER_TOPIC = (
    "/myrobot/left_wheel_encoder_driver_node/tick"
)
RIGHT_ENCODER_TOPIC = (
    "/myrobot/right_wheel_encoder_driver_node/tick"
)

ENCODER_TYPE = "duckietown_msgs/WheelEncoderStamped"

# Encoder outputs on the follower
BOT2_LEFT_ENCODER_TOPIC = (
    "/quackcomm/peer/bot1/left_encoder"
)
BOT2_RIGHT_ENCODER_TOPIC = (
    "/quackcomm/peer/bot1/right_encoder"
)

PUBLISH_PERIOD_S = 0.02
MAX_ENCODER_AGE_S = 0.30


class EncoderRelay:
    def __init__(self):
        self.left_message = None
        self.right_message = None

        self.left_time = None
        self.right_time = None

    def left_callback(self, message):
     self.left_message = dict(message)
     self.left_time = time.monotonic()


    def right_callback(self, message):
      self.right_message = dict(message)
      self.right_time = time.monotonic()


def main():
    bot1 = roslibpy.Ros(
        host=BOT1_HOST,
        port=BOT1_PORT,
    )

    bot2 = roslibpy.Ros(
        host=BOT2_HOST,
        port=BOT2_PORT,
    )

    relay = EncoderRelay()

    left_subscriber = None
    right_subscriber = None
    left_publisher = None
    right_publisher = None

    try:
        logging.info("Connecting to master and follower")

        bot1.run()
        bot2.run()

        deadline = time.monotonic() + 15.0

        while time.monotonic() < deadline:
            if bot1.is_connected and bot2.is_connected:
                break
            time.sleep(0.2)

        if not bot1.is_connected:
            raise RuntimeError("Master rosbridge connection failed")

        if not bot2.is_connected:
            raise RuntimeError("Follower rosbridge connection failed")

        # Publish the original encoder message types.
        left_publisher = roslibpy.Topic(
            bot2,
            BOT2_LEFT_ENCODER_TOPIC,
            ENCODER_TYPE,
            queue_size=1,
        )

        right_publisher = roslibpy.Topic(
            bot2,
            BOT2_RIGHT_ENCODER_TOPIC,
            ENCODER_TYPE,
            queue_size=1,
        )

        left_publisher.advertise()
        right_publisher.advertise()

        left_subscriber = roslibpy.Topic(
            bot1,
            LEFT_ENCODER_TOPIC,
            ENCODER_TYPE,
            queue_length=1,
        )

        right_subscriber = roslibpy.Topic(
            bot1,
            RIGHT_ENCODER_TOPIC,
            ENCODER_TYPE,
            queue_length=1,
        )

        left_subscriber.subscribe(relay.left_callback)
        right_subscriber.subscribe(relay.right_callback)

        logging.info("Direct encoder relay started")

        while bot1.is_connected and bot2.is_connected:
            time.sleep(PUBLISH_PERIOD_S)

            now = time.monotonic()

            left_message = relay.left_message
            right_message = relay.right_message

            left_time = relay.left_time
            right_time = relay.right_time

            # Do not forward stale encoder readings.
            if (
                left_message is None
                or right_message is None
                or left_time is None
                or right_time is None
            ):
                continue

            if (
                now - left_time > MAX_ENCODER_AGE_S
                or now - right_time > MAX_ENCODER_AGE_S
            ):
                continue

            left_publisher.publish(
                roslibpy.Message(left_message)
            )

            right_publisher.publish(
                roslibpy.Message(right_message)
            )

    except KeyboardInterrupt:
        logging.info("Stopping encoder relay")

    except Exception:
        logging.exception("Encoder relay failed")

    finally:
        for topic in (left_subscriber, right_subscriber):
            if topic is not None:
                try:
                    topic.unsubscribe()
                except Exception:
                    pass

        for topic in (left_publisher, right_publisher):
            if topic is not None:
                try:
                    topic.unadvertise()
                except Exception:
                    pass

        bot1.terminate()
        bot2.terminate()

        logging.info("Encoder relay stopped")


if __name__ == "__main__":
    main()