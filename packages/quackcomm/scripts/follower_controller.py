#!/usr/bin/env python3

"""
DB21J wheel-motion follower controller.

Goal:
    Replicate the master's LEFT and RIGHT driven-wheel velocities,
    including direction changes, using encoder feedback.

Master encoder topics (relayed by your bridge):
    /quackcomm/peer/bot1/left_encoder
    /quackcomm/peer/bot1/right_encoder

Follower encoder topics:
    /<VEHICLE_NAME>/left_wheel_encoder_driver_node/tick
    /<VEHICLE_NAME>/right_wheel_encoder_driver_node/tick

Command topic:
    /<VEHICLE_NAME>/joy_mapper_node/car_cmd

The passive roller/caster is not driven and does not enter the
differential-drive wheel-speed equations.
"""

import os
import math
import threading
import time

import rospy

from duckietown_msgs.msg import WheelEncoderStamped, Twist2DStamped


# ============================================================
# CONFIGURATION
# ============================================================

BOT_NAME = os.getenv("VEHICLE_NAME", "myrobot2")

# Wheel geometry. These are defaults; verify against your robot.
WHEEL_RADIUS_M = 0.0318
WHEEL_BASE_M = 0.102

# Used if a message does not provide a valid encoder resolution.
DEFAULT_ENCODER_RESOLUTION = 135.0

# Stop a wheel's target if its encoder has not changed for this long.
# Increase slightly if your encoder produces very few ticks at low speed.
MASTER_WHEEL_STOP_TIMEOUT = 0.25

# Stop if master encoder messages themselves stop arriving.
MASTER_MESSAGE_TIMEOUT = 0.60

# Main controller loop.
CONTROL_RATE_HZ = 30.0

# Encoder velocity filtering:
# Higher alpha reacts faster to direction changes but is noisier.
VELOCITY_FILTER_ALPHA = 0.70

# Maximum permitted linear speed of either driven wheel, in m/s.
# Increase only if your DB21J can safely reach higher speeds.
MAX_WHEEL_SPEED_MPS = 0.25

# Ignore extremely small estimated speeds.
VELOCITY_DEADBAND_MPS = 0.002

# Wheel-speed feedback gains.
# These are starting values; tune only after encoder signs are correct.
WHEEL_KP = 0.60
WHEEL_KI = 0.40

# Limit the feedback correction to avoid large sudden commands.
MAX_FEEDBACK_CORRECTION_MPS = 0.08

# Limit the accumulated integral term to prevent windup.
MAX_INTEGRAL_ERROR = 0.20


# ============================================================
# ENCODER SIGN CALIBRATION
# ============================================================
#
# For EACH wheel, rotate the wheel so that it would roll the robot
# FORWARD. Observe whether its encoder count increases or decreases.
#
# Set the sign to:
#     +1.0 if counts increase during forward wheel rotation
#     -1.0 if counts decrease during forward wheel rotation
#
# Calibrate all four independently. Do not assume left/right or
# master/follower encoders have the same sign convention.
#
# These defaults preserve the previous configuration. They MUST be
# verified experimentally before expecting correct direction matching.

MASTER_LEFT_SIGN = 1.0
MASTER_RIGHT_SIGN = 1.0

FOLLOWER_LEFT_SIGN = 1.0
FOLLOWER_RIGHT_SIGN = 1.0


# ============================================================
# TOPICS
# ============================================================

MASTER_LEFT_TOPIC = "/quackcomm/peer/bot1/left_encoder"
MASTER_RIGHT_TOPIC = "/quackcomm/peer/bot1/right_encoder"

FOLLOWER_LEFT_TOPIC = (
    "/" + BOT_NAME + "/left_wheel_encoder_driver_node/tick"
)
FOLLOWER_RIGHT_TOPIC = (
    "/" + BOT_NAME + "/right_wheel_encoder_driver_node/tick"
)

COMMAND_TOPIC = "/" + BOT_NAME + "/joy_mapper_node/car_cmd"


# ============================================================
# ENCODER VELOCITY ESTIMATOR
# ============================================================

class EncoderVelocity:
    """
    Converts cumulative encoder counts into signed wheel linear velocity.

    The velocity estimate updates when the count changes. If no count
    change occurs before the timeout, the estimated velocity becomes zero.

    Local monotonic time is used instead of message timestamps to avoid
    clock synchronization issues between the two robots.
    """

    def __init__(self, name, encoder_sign):
        self.name = name
        self.encoder_sign = float(encoder_sign)

        self.lock = threading.Lock()

        self.initialized = False
        self.count = 0
        self.resolution = DEFAULT_ENCODER_RESOLUTION

        self.last_change_time = None
        self.last_message_time = None

        self.filtered_velocity = 0.0

    def callback(self, msg):
        now = time.monotonic()

        count = int(msg.data)

        try:
            resolution = float(msg.resolution)
        except (AttributeError, TypeError, ValueError):
            resolution = DEFAULT_ENCODER_RESOLUTION

        if not math.isfinite(resolution) or resolution <= 0:
            resolution = DEFAULT_ENCODER_RESOLUTION

        with self.lock:
            self.last_message_time = now

            if not self.initialized:
                self.initialized = True
                self.count = count
                self.resolution = resolution
                self.last_change_time = now
                self.filtered_velocity = 0.0
                return

            delta_count = count - self.count

            if delta_count == 0:
                # Do not update last_change_time here. It must represent
                # the last actual encoder count change.
                return

            if self.last_change_time is None:
                self.count = count
                self.last_change_time = now
                self.resolution = resolution
                return

            dt = now - self.last_change_time

            # Always advance the baseline when a count change is received.
            old_resolution = self.resolution
            self.count = count
            self.last_change_time = now
            self.resolution = resolution

            if dt <= 0.0:
                return

            # Use the resolution associated with the interval just ended.
            if old_resolution <= 0.0:
                old_resolution = DEFAULT_ENCODER_RESOLUTION

            distance_per_tick = (
                2.0 * math.pi * WHEEL_RADIUS_M / old_resolution
            )

            instantaneous_velocity = (
                delta_count
                * distance_per_tick
                * self.encoder_sign
                / dt
            )

            # Low-pass filter. This also smooths the quantization of ticks.
            alpha = VELOCITY_FILTER_ALPHA

            self.filtered_velocity = (
                alpha * instantaneous_velocity
                + (1.0 - alpha) * self.filtered_velocity
            )

    def get_velocity(self, now):
        """
        Return signed linear wheel velocity in m/s.

        Return zero if no count change has been observed recently.
        """

        with self.lock:
            if not self.initialized or self.last_change_time is None:
                return 0.0

            if now - self.last_change_time > MASTER_WHEEL_STOP_TIMEOUT:
                return 0.0

            return self.filtered_velocity

    def message_is_stale(self, now, timeout):
        with self.lock:
            if self.last_message_time is None:
                return True

            return (now - self.last_message_time) > timeout


# ============================================================
# WHEEL SPEED CONTROLLER
# ============================================================

class WheelSpeedController:
    """
    Applies bounded PI feedback independently to each wheel.

    The output is a corrected desired wheel linear velocity, not a motor
    PWM value. The two wheel velocities are subsequently converted into
    the body Twist command.
    """

    def __init__(self, name):
        self.name = name
        self.integral = 0.0
        self.last_target = 0.0

    def update(self, target_velocity, measured_velocity, dt):
        target_velocity = max(
            -MAX_WHEEL_SPEED_MPS,
            min(MAX_WHEEL_SPEED_MPS, target_velocity)
        )

        # If the master wheel stops, command zero rather than attempting
        # to drive the follower backwards to correct a transient error.
        if abs(target_velocity) < VELOCITY_DEADBAND_MPS:
            self.integral = 0.0
            self.last_target = 0.0
            return 0.0

        # Reset integral when the master reverses direction.
        if (
            self.last_target != 0.0
            and target_velocity * self.last_target < 0.0
        ):
            self.integral = 0.0

        self.last_target = target_velocity

        error = target_velocity - measured_velocity

        dt = max(0.001, min(dt, 0.2))

        self.integral += error * dt
        self.integral = max(
            -MAX_INTEGRAL_ERROR,
            min(MAX_INTEGRAL_ERROR, self.integral)
        )

        correction = WHEEL_KP * error + WHEEL_KI * self.integral

        correction = max(
            -MAX_FEEDBACK_CORRECTION_MPS,
            min(MAX_FEEDBACK_CORRECTION_MPS, correction)
        )

        corrected_velocity = target_velocity + correction

        corrected_velocity = max(
            -MAX_WHEEL_SPEED_MPS,
            min(MAX_WHEEL_SPEED_MPS, corrected_velocity)
        )

        return corrected_velocity


# ============================================================
# MAIN NODE
# ============================================================

class FollowerController:

    def __init__(self):
        rospy.init_node("follower_controller", anonymous=False)

        # Allow runtime overrides through ROS parameters.
        self.master_left_topic = rospy.get_param(
            "~master_left_topic", MASTER_LEFT_TOPIC
        )
        self.master_right_topic = rospy.get_param(
            "~master_right_topic", MASTER_RIGHT_TOPIC
        )

        self.follower_left_topic = rospy.get_param(
            "~follower_left_topic", FOLLOWER_LEFT_TOPIC
        )
        self.follower_right_topic = rospy.get_param(
            "~follower_right_topic", FOLLOWER_RIGHT_TOPIC
        )

        self.command_topic = rospy.get_param(
            "~command_topic", COMMAND_TOPIC
        )

        # Independent encoder sign configuration.
        master_left_sign = rospy.get_param(
            "~master_left_sign", MASTER_LEFT_SIGN
        )
        master_right_sign = rospy.get_param(
            "~master_right_sign", MASTER_RIGHT_SIGN
        )
        follower_left_sign = rospy.get_param(
            "~follower_left_sign", FOLLOWER_LEFT_SIGN
        )
        follower_right_sign = rospy.get_param(
            "~follower_right_sign", FOLLOWER_RIGHT_SIGN
        )

        self.master_left = EncoderVelocity(
            "master_left", master_left_sign
        )
        self.master_right = EncoderVelocity(
            "master_right", master_right_sign
        )

        self.follower_left = EncoderVelocity(
            "follower_left", follower_left_sign
        )
        self.follower_right = EncoderVelocity(
            "follower_right", follower_right_sign
        )

        self.left_controller = WheelSpeedController("left")
        self.right_controller = WheelSpeedController("right")

        self.publisher = rospy.Publisher(
            self.command_topic,
            Twist2DStamped,
            queue_size=1
        )

        # Master encoder subscriptions.
        rospy.Subscriber(
            self.master_left_topic,
            WheelEncoderStamped,
            self.master_left.callback,
            queue_size=20
        )

        rospy.Subscriber(
            self.master_right_topic,
            WheelEncoderStamped,
            self.master_right.callback,
            queue_size=20
        )

        # Follower encoder subscriptions.
        rospy.Subscriber(
            self.follower_left_topic,
            WheelEncoderStamped,
            self.follower_left.callback,
            queue_size=20
        )

        rospy.Subscriber(
            self.follower_right_topic,
            WheelEncoderStamped,
            self.follower_right.callback,
            queue_size=20
        )

        self.last_loop_time = time.monotonic()

        rospy.on_shutdown(self.stop_robot)

        rospy.loginfo("============================================")
        rospy.loginfo("DB21J wheel-motion follower started")
        rospy.loginfo("Robot: %s", BOT_NAME)
        rospy.loginfo("Master left:  %s", self.master_left_topic)
        rospy.loginfo("Master right: %s", self.master_right_topic)
        rospy.loginfo("Follower left:  %s", self.follower_left_topic)
        rospy.loginfo("Follower right: %s", self.follower_right_topic)
        rospy.loginfo("Command: %s", self.command_topic)
        rospy.loginfo(
            "Encoder signs: ML=%s MR=%s FL=%s FR=%s",
            master_left_sign,
            master_right_sign,
            follower_left_sign,
            follower_right_sign
        )
        rospy.loginfo("============================================")

    @staticmethod
    def clamp(value, lower, upper):
        return max(lower, min(upper, value))

    def publish_wheel_commands(self, left_velocity, right_velocity):
        """
        Convert left/right wheel linear velocities to differential-drive
        body velocity and angular velocity.

        v     = (v_left + v_right) / 2
        omega = (v_right - v_left) / wheel_base
        """

        v = (left_velocity + right_velocity) / 2.0

        omega = (
            right_velocity - left_velocity
        ) / WHEEL_BASE_M

        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = v
        msg.omega = omega

        self.publisher.publish(msg)

    def stop_robot(self):
        """
        Publish zero commands repeatedly to improve the chance that the
        downstream command subscriber receives a stop command on shutdown.
        """

        if not hasattr(self, "publisher"):
            return

        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = 0.0
        msg.omega = 0.0

        for _ in range(3):
            try:
                self.publisher.publish(msg)
                time.sleep(0.03)
            except Exception:
                break

    def run(self):
        rate = rospy.Rate(CONTROL_RATE_HZ)

        while not rospy.is_shutdown():
            now = time.monotonic()

            dt = now - self.last_loop_time
            self.last_loop_time = now
            dt = max(0.001, min(dt, 0.2))

            # If master messages disappear, fail safe and stop.
            left_stale = self.master_left.message_is_stale(
                now, MASTER_MESSAGE_TIMEOUT
            )
            right_stale = self.master_right.message_is_stale(
                now, MASTER_MESSAGE_TIMEOUT
            )

            if left_stale or right_stale:
                self.left_controller.integral = 0.0
                self.right_controller.integral = 0.0

                self.publish_wheel_commands(0.0, 0.0)

                rospy.logwarn_throttle(
                    2.0,
                    "Master encoder message stale. Stopping follower."
                )

                rate.sleep()
                continue

            # Estimate each wheel's signed physical velocity.
            master_left_velocity = self.master_left.get_velocity(now)
            master_right_velocity = self.master_right.get_velocity(now)

            follower_left_velocity = self.follower_left.get_velocity(now)
            follower_right_velocity = self.follower_right.get_velocity(now)

            # Bound the requested master velocities to the follower's
            # configured safe operating range.
            target_left = self.clamp(
                master_left_velocity,
                -MAX_WHEEL_SPEED_MPS,
                MAX_WHEEL_SPEED_MPS
            )

            target_right = self.clamp(
                master_right_velocity,
                -MAX_WHEEL_SPEED_MPS,
                MAX_WHEEL_SPEED_MPS
            )

            # Independent feedback for each driven wheel.
            command_left = self.left_controller.update(
                target_left,
                follower_left_velocity,
                dt
            )

            command_right = self.right_controller.update(
                target_right,
                follower_right_velocity,
                dt
            )

            # Convert wheel commands to the DB21J body Twist command.
            self.publish_wheel_commands(
                command_left,
                command_right
            )

            rospy.loginfo_throttle(
                1.0,
                (
                    "MASTER L/R: %.3f / %.3f m/s | "
                    "FOLLOWER L/R: %.3f / %.3f m/s | "
                    "CMD WHEELS L/R: %.3f / %.3f m/s"
                ),
                master_left_velocity,
                master_right_velocity,
                follower_left_velocity,
                follower_right_velocity,
                command_left,
                command_right
            )

            rate.sleep()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        node = FollowerController()
        node.run()

    except rospy.ROSInterruptException:
        pass