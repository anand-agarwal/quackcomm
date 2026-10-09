#!/usr/bin/env python3

import time
import logging
import roslibpy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

BOT1_HOST = "127.0.0.1"
BOT2_HOST = "127.0.0.2"
ROSBRIDGE_PORT = 9001

SOURCE_TOPIC = "/bot1/front_center_tof_driver_node/range"
DEST_TOPIC = "/quackcomm/peer/bot1/tof_range"


def main():
    bot1 = roslibpy.Ros(
        host=BOT1_HOST,
        port=ROSBRIDGE_PORT,
    )

    bot2 = roslibpy.Ros(
        host=BOT2_HOST,
        port=ROSBRIDGE_PORT,
    )

    logging.info("Connecting to bot1 and bot2 rosbridge servers...")

    bot1.run()
    bot2.run()

    try:
        deadline = time.time() + 15

        while time.time() < deadline:
            if bot1.is_connected and bot2.is_connected:
                break
            time.sleep(0.2)

        if not bot1.is_connected:
            raise RuntimeError("Could not connect to bot1 rosbridge")

        if not bot2.is_connected:
            raise RuntimeError("Could not connect to bot2 rosbridge")

        logging.info("Connected to both robots.")

        publisher = roslibpy.Topic(
            bot2,
            DEST_TOPIC,
            "sensor_msgs/Range",
        )

        subscriber = roslibpy.Topic(
            bot1,
            SOURCE_TOPIC,
            "sensor_msgs/Range",
        )

        publisher.advertise()

        def forward_range(message):
            publisher.publish(roslibpy.Message(message))
            logging.info(
                "Forwarded ToF reading: range=%s",
                message.get("range"),
            )

        subscriber.subscribe(forward_range)

        logging.info("Forwarding %s -> %s", SOURCE_TOPIC, DEST_TOPIC)

        while bot1.is_connected and bot2.is_connected:
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("Stopping bridge.")

    finally:
        try:
            subscriber.unsubscribe()
        except Exception:
            pass

        try:
            publisher.unadvertise()
        except Exception:
            pass

        bot1.terminate()
        bot2.terminate()


if __name__ == "__main__":
    main()
