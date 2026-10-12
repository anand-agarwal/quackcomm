#!/usr/bin/env python3

import time
import math
import logging
import roslibpy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

BOT2_HOST = "myrobot2.local"
BOT2_PORT = 9001

LEADER_TOPIC = "/quackcomm/peer/bot1/cmd"

# Initial parameters for algorithm testing.
INITIAL_TARGET_SPEED = 0.19
SPEED_CHANGE_THRESHOLD = 0.03
MAX_ACCELERATION = 0.20
MAX_DECELERATION = 0.30
MESSAGE_TIMEOUT = 1.0
UPDATE_INTERVAL = 0.1


def main():
    bot2 = roslibpy.Ros(
        host=BOT2_HOST,
        port=BOT2_PORT,
    )

    state = {
        "leader_speed": None,
        "last_received": None,
        "previous_leader_speed": None,
        "target_speed": INITIAL_TARGET_SPEED,
    }

    try:
        bot2.run()

        deadline = time.monotonic() + 15

        while time.monotonic() < deadline:
            if bot2.is_connected:
                break
            time.sleep(0.2)

        if not bot2.is_connected:
            raise RuntimeError("Could not connect to Bot 2")

        logging.info("Connected to Bot 2")

        def on_leader_command(message):
            try:
                speed = float(message["v"])

                if not math.isfinite(speed):
                    logging.warning("Ignoring non-finite speed")
                    return

                now = time.monotonic()

                previous = state["leader_speed"]

                state["leader_speed"] = speed
                state["last_received"] = now

                if previous is None:
                    logging.info(
                        "Initial leader speed received: %.3f m/s",
                        speed,
                    )
                    return

                change = speed - previous

                if abs(change) >= SPEED_CHANGE_THRESHOLD:
                    logging.info(
                        "SPEED_CHANGE detected: %.3f -> %.3f m/s",
                        previous,
                        speed,
                    )

                state["previous_leader_speed"] = previous

            except (KeyError, TypeError, ValueError):
                logging.exception("Invalid leader message")

        subscriber = roslibpy.Topic(
            bot2,
            LEADER_TOPIC,
            "duckietown_msgs/Twist2DStamped",
            queue_length=1,
        )

        subscriber.subscribe(on_leader_command)

        logging.info(
            "Monitoring leader speed on %s",
            LEADER_TOPIC,
        )

        last_update = time.monotonic()

        while bot2.is_connected:
            now = time.monotonic()
            elapsed = now - last_update

            if elapsed < UPDATE_INTERVAL:
                time.sleep(0.01)
                continue

            last_update = now

            leader_speed = state["leader_speed"]
            received_at = state["last_received"]

            if leader_speed is None or received_at is None:
                continue

            if now - received_at > MESSAGE_TIMEOUT:
                logging.warning(
                    "Leader data is stale. "
                    "No speed command will be generated."
                )
                continue

            current_target = state["target_speed"]
            error = leader_speed - current_target

            if abs(error) < 0.005:
                continue

            # Limit the change in target speed.
            if error > 0:
                max_change = MAX_ACCELERATION * elapsed
            else:
                max_change = MAX_DECELERATION * elapsed

            change = max(
                -max_change,
                min(max_change, error),
            )

            state["target_speed"] = current_target + change

            logging.info(
                "Follower target: %.3f m/s | Leader: %.3f m/s",
                state["target_speed"],
                leader_speed,
            )

    except KeyboardInterrupt:
        logging.info("Stopping follower monitor")

    finally:
        bot2.terminate()


if __name__ == "__main__":
    main()