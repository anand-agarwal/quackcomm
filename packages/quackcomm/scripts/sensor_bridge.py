#!/usr/bin/env python3

import os
import time
import logging
import roslibpy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# --------------------------------------------------
# Robot connection settings
# --------------------------------------------------

BOT1_HOST = os.getenv("MYROBOT_HOST", "myrobot.local")
BOT2_HOST = os.getenv("MYROBOT2_HOST", "myrobot2.local")

BOT1_PORT = int(os.getenv("MYROBOT_ROSBRIDGE_PORT", "9001"))
BOT2_PORT = int(os.getenv("MYROBOT2_ROSBRIDGE_PORT", "9001"))

# --------------------------------------------------
# Source topics: existing command topics
# --------------------------------------------------

BOT1_SOURCE_TOPIC = "/myrobot/kinematics_node/velocity"
BOT2_SOURCE_TOPIC = "/myrobot2/kinematics_node/velocity"

# --------------------------------------------------
# Destination topics: peer commands received
# These are telemetry topics, not motor commands.
# --------------------------------------------------

BOT1_RECEIVE_TOPIC = "/quackcomm/peer/bot2/cmd"
BOT2_RECEIVE_TOPIC = "/quackcomm/peer/bot1/cmd"

MESSAGE_TYPE = "duckietown_msgs/Twist2DStamped"

# Forward at most 20 messages per second per direction.
# The queue length of 1 favors recent commands over old ones.
THROTTLE_MS = 50


def make_forward_callback(publisher, direction):
    """Create a callback that forwards commands to the peer."""

    last_log_time = [0.0]

    def forward_command(message):
        try:
            v = float(message["v"])
            omega = float(message["omega"])

            # Forward the original message without changing
            # the speed or steering fields.
            publisher.publish(
                roslibpy.Message(dict(message))
            )

            # Avoid printing a log line for every message.
            now = time.monotonic()

            if now - last_log_time[0] >= 1.0:
                logging.info(
                    "%s | v=%.3f m/s | omega=%.3f rad/s",
                    direction,
                    v,
                    omega,
                )
                last_log_time[0] = now

        except (KeyError, TypeError, ValueError):
            logging.exception(
                "Invalid command received in direction %s",
                direction,
            )

        except Exception:
            logging.exception(
                "Failed to forward command in direction %s",
                direction,
            )

    return forward_command


def main():
    bot1 = roslibpy.Ros(
        host=BOT1_HOST,
        port=BOT1_PORT,
    )

    bot2 = roslibpy.Ros(
        host=BOT2_HOST,
        port=BOT2_PORT,
    )

    # Initialize resources so cleanup works even if setup fails.
    bot1_source = None
    bot2_source = None

    bot1_receiver = None
    bot2_receiver = None

    logging.info(
        "Connecting to Bot 1 at %s:%s and Bot 2 at %s:%s",
        BOT1_HOST,
        BOT1_PORT,
        BOT2_HOST,
        BOT2_PORT,
    )

    try:
        # Connect to both ROSBridge servers.
        bot1.run()
        bot2.run()

        # Wait for both connections.
        deadline = time.monotonic() + 15

        while time.monotonic() < deadline:
            if bot1.is_connected and bot2.is_connected:
                break

            time.sleep(0.2)

        if not bot1.is_connected:
            raise RuntimeError(
                f"Could not connect to Bot 1: "
                f"{BOT1_HOST}:{BOT1_PORT}"
            )

        if not bot2.is_connected:
            raise RuntimeError(
                f"Could not connect to Bot 2: "
                f"{BOT2_HOST}:{BOT2_PORT}"
            )

        logging.info("Connected to both robots.")

        # ==================================================
        # Publishers: publish peer data onto the receiving
        # robot's ROS master.
        # ==================================================

        bot1_receiver = roslibpy.Topic(
            bot1,
            BOT1_RECEIVE_TOPIC,
            MESSAGE_TYPE,
            queue_size=1,
        )

        bot2_receiver = roslibpy.Topic(
            bot2,
            BOT2_RECEIVE_TOPIC,
            MESSAGE_TYPE,
            queue_size=1,
        )

        bot1_receiver.advertise()
        bot2_receiver.advertise()

        # ==================================================
        # Subscribers: read each robot's local command.
        # ==================================================

        bot1_source = roslibpy.Topic(
            bot1,
            BOT1_SOURCE_TOPIC,
            MESSAGE_TYPE,
            throttle_rate=THROTTLE_MS,
            queue_length=1,
        )

        bot2_source = roslibpy.Topic(
            bot2,
            BOT2_SOURCE_TOPIC,
            MESSAGE_TYPE,
            throttle_rate=THROTTLE_MS,
            queue_length=1,
        )

        # ==================================================
        # Direction 1: Bot 1 -> Bot 2
        # ==================================================

        bot1_source.subscribe(
            make_forward_callback(
                bot2_receiver,
                "BOT1 -> BOT2",
            )
        )

        # ==================================================
        # Direction 2: Bot 2 -> Bot 1
        # ==================================================

        bot2_source.subscribe(
            make_forward_callback(
                bot1_receiver,
                "BOT2 -> BOT1",
            )
        )

        logging.info("Bidirectional speed bridge started.")

        logging.info(
            "Bot 1 source:      %s",
            BOT1_SOURCE_TOPIC,
        )
        logging.info(
            "Bot 1 receives:    %s",
            BOT1_RECEIVE_TOPIC,
        )

        logging.info(
            "Bot 2 source:      %s",
            BOT2_SOURCE_TOPIC,
        )
        logging.info(
            "Bot 2 receives:    %s",
            BOT2_RECEIVE_TOPIC,
        )

        logging.info(
            "Forwarding messages of type %s",
            MESSAGE_TYPE,
        )

        # Keep the bridge running while both connections survive.
        while bot1.is_connected and bot2.is_connected:
            time.sleep(1)

        logging.warning(
            "A ROSBridge connection was lost. "
            "The bridge is stopping."
        )

    except KeyboardInterrupt:
        logging.info("Stopping bridge.")

    except Exception:
        logging.exception("Bridge encountered an error.")

    finally:
        # Unsubscribe from source topics.
        for topic in (bot1_source, bot2_source):
            if topic is not None:
                try:
                    topic.unsubscribe()
                except Exception:
                    pass

        # Unadvertise the receiving topics.
        for topic in (bot1_receiver, bot2_receiver):
            if topic is not None:
                try:
                    topic.unadvertise()
                except Exception:
                    pass

        bot1.terminate()
        bot2.terminate()

        logging.info("Bridge stopped.")


if __name__ == "__main__":
    main()