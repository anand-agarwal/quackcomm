#!/usr/bin/env python3
"""Leader/follower controller for two DB21J Duckiebots.

Layer 1: encoder-based wheel velocity estimation + bounded PI corrections.
Layer 2: odometry-based trajectory tracking to a fixed-delay leader reference.

IMPORTANT LIMITATIONS
---------------------
* The supplied sensor_bridge relays encoder counts only, not global pose.
  This script therefore integrates wheel odometry for BOTH robots.
* The two initial poses must be configured in the SAME coordinate frame.
  Wheel odometry drifts; this is not global localization or a guarantee of
  exact trajectory reproduction over long runs.
* Start this node and the bridge before moving the leader so the path history
  is captured. The delay history is initially back-filled with the leader's
  initial pose.
* The controller is DISARMED by default. Set ~enable_controller:=true only
  after checking signs, initial poses, topics, wheel geometry, and clear space.
* All four encoder signs must be calibrated experimentally.

Reference model:
    q_d(t) = q_leader(t - trajectory_delay)

Pose errors in follower body frame:
    e_x =  cos(theta_f)*(x_d-x_f) + sin(theta_f)*(y_d-y_f)
    e_y = -sin(theta_f)*(x_d-x_f) + cos(theta_f)*(y_d-y_f)
    e_theta = wrap(theta_d-theta_f)

Outer tracking law (a practical simplified controller, not the exact paper law):
    v_cmd     = v_d*cos(e_theta) + k_x*e_x
    omega_cmd = omega_d + k_y*v_d*e_y + k_theta*sin(e_theta)

The resulting (v_cmd, omega_cmd) is converted to follower wheel targets.
Layer 1 compares those targets to follower encoder velocities, applies bounded
PI corrections, and converts the corrected wheel pair back to Twist2DStamped.
"""

import math
import os
import threading
import time
from collections import deque

import rospy
from duckietown_msgs.msg import WheelEncoderStamped, Twist2DStamped


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_to_pi(angle):
    """Wrap angle to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def valid_positive(value, fallback):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) and value > 0.0 else fallback


class EncoderState:
    """Thread-safe cumulative encoder state and signed wheel-speed estimate."""

    def __init__(self, name, encoder_sign, default_resolution, wheel_radius, filter_alpha=0.65):
        self.name = name
        self.sign = float(encoder_sign)
        self.default_resolution = float(default_resolution)
        self.wheel_radius = float(wheel_radius)
        self.filter_alpha = clamp(float(filter_alpha), 0.0, 1.0)
        self.lock = threading.Lock()

        self.initialized = False
        self.count = 0
        self.resolution = self.default_resolution
        self.last_message_time = None

        self.last_velocity_count = None
        self.last_change_time = None
        self.filtered_velocity = 0.0

        self.last_odom_count = None

    def callback(self, msg):
        now = time.monotonic()
        count = int(msg.data)
        try:
            resolution = valid_positive(msg.resolution, self.default_resolution)
        except AttributeError:
            resolution = self.default_resolution

        with self.lock:
            self.last_message_time = now

            if not self.initialized:
                self.initialized = True
                self.count = count
                self.resolution = resolution
                self.last_velocity_count = count
                self.last_change_time = now
                self.filtered_velocity = 0.0
                return

            previous_resolution = self.resolution
            self.count = count
            self.resolution = resolution
            delta = count - self.last_velocity_count

            # Duplicate bridged messages are expected. They refresh message
            # freshness but must not refresh the last actual tick-change time.
            if delta == 0:
                return

            now_since_change = now - self.last_change_time if self.last_change_time is not None else 0.0
            self.last_velocity_count = count
            self.last_change_time = now

            if now_since_change <= 1.0e-6:
                return

            ticks_per_rev = valid_positive(previous_resolution, self.default_resolution)
            metres_per_tick = 2.0 * math.pi * self.wheel_radius / ticks_per_rev
            instantaneous = self.sign * delta * metres_per_tick / now_since_change

            # React immediately to a sign reversal; otherwise low-pass filter
            # quantized ticks to reduce velocity noise.
            alpha = self.filter_alpha
            if self.filtered_velocity * instantaneous < 0.0:
                self.filtered_velocity = instantaneous
            else:
                self.filtered_velocity = (
                    alpha * instantaneous + (1.0 - alpha) * self.filtered_velocity
                )

    def is_message_stale(self, now, timeout_s):
        with self.lock:
            return self.last_message_time is None or (now - self.last_message_time) > timeout_s

    def get_velocity(self, now, stop_timeout_s):
        with self.lock:
            if (not self.initialized or self.last_change_time is None or
                    now - self.last_change_time > stop_timeout_s):
                return 0.0
            return self.filtered_velocity

    def consume_tick_delta(self):
        """Return (new tick delta, resolution). Each cumulative delta is used once."""
        with self.lock:
            if not self.initialized:
                return None
            if self.last_odom_count is None:
                self.last_odom_count = self.count
                return (0, self.resolution)
            delta = self.count - self.last_odom_count
            self.last_odom_count = self.count
            return (delta, self.resolution)


class DifferentialDriveOdometry:
    """Integrate wheel tick deltas into a pose in the configured world frame."""

    def __init__(self, x0, y0, theta0, radius, wheel_base, left_sign, right_sign):
        self.x = float(x0)
        self.y = float(y0)
        self.theta = wrap_to_pi(float(theta0))
        self.radius = float(radius)
        self.wheel_base = float(wheel_base)
        self.left_sign = float(left_sign)
        self.right_sign = float(right_sign)
        if self.radius <= 0.0 or self.wheel_base <= 0.0:
            raise ValueError("Wheel radius and wheelbase must be positive")

    def integrate(self, left_delta, left_resolution, right_delta, right_resolution):
        left_resolution = valid_positive(left_resolution, 135.0)
        right_resolution = valid_positive(right_resolution, 135.0)

        ds_l = (self.left_sign * int(left_delta) * 2.0 * math.pi * self.radius
                / left_resolution)
        ds_r = (self.right_sign * int(right_delta) * 2.0 * math.pi * self.radius
                / right_resolution)

        ds = 0.5 * (ds_l + ds_r)
        dtheta = (ds_r - ds_l) / self.wheel_base

        # Exact integration of a constant-curvature differential-drive arc.
        if abs(dtheta) < 1.0e-8:
            self.x += ds * math.cos(self.theta)
            self.y += ds * math.sin(self.theta)
        else:
            radius_of_arc = ds / dtheta
            new_theta = self.theta + dtheta
            self.x += radius_of_arc * (math.sin(new_theta) - math.sin(self.theta))
            self.y -= radius_of_arc * (math.cos(new_theta) - math.cos(self.theta))

        self.theta = wrap_to_pi(self.theta + dtheta)

    def pose(self):
        return self.x, self.y, self.theta


class WheelSpeedController:
    """Bounded PI correction around each wheel's desired physical velocity."""

    def __init__(self, name, kp, ki, max_correction, max_wheel_speed,
                 deadband, max_integral):
        self.name = name
        self.kp = float(kp)
        self.ki = float(ki)
        self.max_correction = float(max_correction)
        self.max_wheel_speed = float(max_wheel_speed)
        self.deadband = float(deadband)
        self.max_integral = float(max_integral)
        self.integral = 0.0
        self.last_target = 0.0

    def reset(self):
        self.integral = 0.0
        self.last_target = 0.0

    def update(self, target, measured, dt):
        target = clamp(target, -self.max_wheel_speed, self.max_wheel_speed)
        dt = clamp(dt, 0.001, 0.2)

        if abs(target) < self.deadband:
            self.reset()
            return 0.0

        if self.last_target != 0.0 and target * self.last_target < 0.0:
            self.integral = 0.0
        self.last_target = target

        error = target - measured
        proposed_integral = clamp(
            self.integral + error * dt, -self.max_integral, self.max_integral
        )
        raw_correction = self.kp * error + self.ki * proposed_integral
        correction = clamp(raw_correction, -self.max_correction, self.max_correction)

        # Simple anti-windup: don't integrate further into a saturated correction.
        saturated_same_direction = (
            abs(raw_correction) > self.max_correction and error * raw_correction > 0.0
        )
        if not saturated_same_direction:
            self.integral = proposed_integral

        return clamp(target + correction, -self.max_wheel_speed, self.max_wheel_speed)


class TrajectorySample:
    __slots__ = ("t", "x", "y", "theta", "v", "omega")

    def __init__(self, t, x, y, theta, v, omega):
        self.t = float(t)
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
        self.v = float(v)
        self.omega = float(omega)


class FollowerController:
    def __init__(self):
        rospy.init_node("follower_controller", anonymous=False)

        self.bot_name = os.getenv("VEHICLE_NAME", "myrobot2")

        # Topics. Sensor bridge relays bot1 encoders to these follower-side topics.
        self.master_left_topic = rospy.get_param(
            "~master_left_topic", "/quackcomm/peer/bot1/left_encoder")
        self.master_right_topic = rospy.get_param(
            "~master_right_topic", "/quackcomm/peer/bot1/right_encoder")
        self.follower_left_topic = rospy.get_param(
            "~follower_left_topic",
            "/" + self.bot_name + "/left_wheel_encoder_driver_node/tick")
        self.follower_right_topic = rospy.get_param(
            "~follower_right_topic",
            "/" + self.bot_name + "/right_wheel_encoder_driver_node/tick")

        # Current Duckietown chassis-level topic. Override if your deployment
        # intentionally routes commands through a different controller source.
        self.command_topic = rospy.get_param(
            "~command_topic", "/" + self.bot_name + "/car_cmd_switch_node/cmd")

        default_resolution = float(rospy.get_param("~default_encoder_resolution", 135.0))
        velocity_filter_alpha = clamp(float(rospy.get_param("~velocity_filter_alpha", 0.65)), 0.0, 1.0)
        master_radius = float(rospy.get_param("~master_wheel_radius_m", 0.0318))
        follower_radius = float(rospy.get_param("~follower_wheel_radius_m", 0.0318))
        master_base = float(rospy.get_param("~master_wheel_base_m", 0.102))
        follower_base = float(rospy.get_param("~follower_wheel_base_m", 0.102))

        self.master_left_sign = float(rospy.get_param("~master_left_sign", 1.0))
        self.master_right_sign = float(rospy.get_param("~master_right_sign", 1.0))
        self.follower_left_sign = float(rospy.get_param("~follower_left_sign", 1.0))
        self.follower_right_sign = float(rospy.get_param("~follower_right_sign", 1.0))

        self.master_left = EncoderState(
            "master_left", self.master_left_sign, default_resolution, master_radius, velocity_filter_alpha)
        self.master_right = EncoderState(
            "master_right", self.master_right_sign, default_resolution, master_radius, velocity_filter_alpha)
        self.follower_left = EncoderState(
            "follower_left", self.follower_left_sign, default_resolution, follower_radius, velocity_filter_alpha)
        self.follower_right = EncoderState(
            "follower_right", self.follower_right_sign, default_resolution, follower_radius, velocity_filter_alpha)

        # Both robots must be initialized in one common world frame. Defaults
        # assume leader starts at (0,0,0), follower starts 0.30 m behind it.
        ml_x = float(rospy.get_param("~master_initial_x", 0.0))
        ml_y = float(rospy.get_param("~master_initial_y", 0.0))
        ml_theta = float(rospy.get_param("~master_initial_theta", 0.0))
        fl_x = float(rospy.get_param("~follower_initial_x", -0.30))
        fl_y = float(rospy.get_param("~follower_initial_y", 0.0))
        fl_theta = float(rospy.get_param("~follower_initial_theta", 0.0))

        self.master_odom = DifferentialDriveOdometry(
            ml_x, ml_y, ml_theta, master_radius, master_base,
            self.master_left_sign, self.master_right_sign)
        self.follower_odom = DifferentialDriveOdometry(
            fl_x, fl_y, fl_theta, follower_radius, follower_base,
            self.follower_left_sign, self.follower_right_sign)

        # Outer-layer parameters.
        self.trajectory_delay = max(0.0, float(rospy.get_param("~trajectory_delay_s", 2.0)))
        self.k_x = float(rospy.get_param("~k_x", 0.8))
        self.k_y = float(rospy.get_param("~k_y", 4.0))
        self.k_theta = float(rospy.get_param("~k_theta", 1.5))
        self.max_linear_speed = abs(float(rospy.get_param("~max_linear_speed_mps", 0.18)))
        self.max_angular_speed = abs(float(rospy.get_param("~max_angular_speed_rad_s", 1.5)))
        self.max_wheel_speed = abs(float(rospy.get_param("~max_wheel_speed_mps", 0.25)))
        self.max_tracking_error = abs(float(rospy.get_param("~max_tracking_error_m", 1.5)))

        # Encoder and control loop timing.
        self.control_rate_hz = max(5.0, float(rospy.get_param("~control_rate_hz", 30.0)))
        self.master_message_timeout = float(rospy.get_param("~master_message_timeout_s", 0.60))
        self.follower_message_timeout = float(rospy.get_param("~follower_message_timeout_s", 0.60))
        self.encoder_stop_timeout = float(rospy.get_param("~encoder_stop_timeout_s", 0.70))
        self.deadband = float(rospy.get_param("~wheel_velocity_deadband_mps", 0.002))
        self.enable_controller = bool(rospy.get_param("~enable_controller", True))

        kp = float(rospy.get_param("~wheel_kp", 0.60))
        ki = float(rospy.get_param("~wheel_ki", 0.40))
        max_correction = float(rospy.get_param("~max_feedback_correction_mps", 0.05))
        max_integral = float(rospy.get_param("~max_integral_error", 0.20))

        controller_args = (kp, ki, max_correction, self.max_wheel_speed,
                           self.deadband, max_integral)
        self.left_controller = WheelSpeedController("left", *controller_args)
        self.right_controller = WheelSpeedController("right", *controller_args)

        self.publisher = rospy.Publisher(self.command_topic, Twist2DStamped, queue_size=1)
        self.subscribers = [
            rospy.Subscriber(self.master_left_topic, WheelEncoderStamped,
                             self.master_left.callback, queue_size=20),
            rospy.Subscriber(self.master_right_topic, WheelEncoderStamped,
                             self.master_right.callback, queue_size=20),
            rospy.Subscriber(self.follower_left_topic, WheelEncoderStamped,
                             self.follower_left.callback, queue_size=20),
            rospy.Subscriber(self.follower_right_topic, WheelEncoderStamped,
                             self.follower_right.callback, queue_size=20),
        ]

        self.history = deque()
        self.history_initialized = False
        self.last_loop_time = time.monotonic()
        rospy.on_shutdown(self.stop_robot)

        rospy.loginfo("Follower controller initialized")
        rospy.loginfo("Master encoders: %s | %s", self.master_left_topic, self.master_right_topic)
        rospy.loginfo("Follower encoders: %s | %s", self.follower_left_topic, self.follower_right_topic)
        rospy.loginfo("Command topic: %s", self.command_topic)
        rospy.loginfo("Signs ML/MR/FL/FR = %.0f/%.0f/%.0f/%.0f",
                      self.master_left_sign, self.master_right_sign,
                      self.follower_left_sign, self.follower_right_sign)
        rospy.loginfo("Trajectory delay %.2f s; initial follower pose (%.2f, %.2f, %.2f)",
                      self.trajectory_delay, fl_x, fl_y, fl_theta)
        if not self.enable_controller:
            rospy.logwarn("CONTROLLER DISARMED. No motion commands will be published until ~enable_controller:=true")

    def publish_twist(self, v, omega):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = float(v)
        msg.omega = float(omega)
        self.publisher.publish(msg)

    def stop_robot(self):
        # Send repeated stops for shutdown safety.
        if not hasattr(self, "publisher"):
            return
        for _ in range(3):
            try:
                self.publish_twist(0.0, 0.0)
                time.sleep(0.03)
            except Exception:
                break

    def update_odometry(self):
        ml = self.master_left.consume_tick_delta()
        mr = self.master_right.consume_tick_delta()
        fl = self.follower_left.consume_tick_delta()
        fr = self.follower_right.consume_tick_delta()

        if ml is not None and mr is not None:
            self.master_odom.integrate(ml[0], ml[1], mr[0], mr[1])
        if fl is not None and fr is not None:
            self.follower_odom.integrate(fl[0], fl[1], fr[0], fr[1])

    @staticmethod
    def interpolate_sample(a, b, t):
        if b.t <= a.t:
            return TrajectorySample(t, a.x, a.y, a.theta, a.v, a.omega)
        ratio = clamp((t - a.t) / (b.t - a.t), 0.0, 1.0)
        theta_delta = wrap_to_pi(b.theta - a.theta)
        return TrajectorySample(
            t,
            a.x + ratio * (b.x - a.x),
            a.y + ratio * (b.y - a.y),
            wrap_to_pi(a.theta + ratio * theta_delta),
            a.v + ratio * (b.v - a.v),
            a.omega + ratio * (b.omega - a.omega),
        )

    def initialize_history(self, now, master_pose):
        x, y, theta = master_pose
        # The past before this node starts is unknown, so initialize it as a
        # stationary leader at its current estimated pose. Start node before moving.
        n = max(2, int(math.ceil(self.trajectory_delay * self.control_rate_hz)) + 1)
        start_t = now - self.trajectory_delay
        for i in range(n):
            t = start_t + (self.trajectory_delay * i / float(n - 1))
            self.history.append(TrajectorySample(t, x, y, theta, 0.0, 0.0))
        self.history_initialized = True

    def append_history(self, now, master_pose, v, omega):
        x, y, theta = master_pose
        self.history.append(TrajectorySample(now, x, y, theta, v, omega))
        keep_after = now - max(5.0, self.trajectory_delay + 3.0)
        while len(self.history) > 2 and self.history[1].t < keep_after:
            self.history.popleft()

    def reference_at(self, t):
        if not self.history or t < self.history[0].t or t > self.history[-1].t:
            return None
        previous = self.history[0]
        for current in list(self.history)[1:]:
            if current.t >= t:
                return self.interpolate_sample(previous, current, t)
            previous = current
        last = self.history[-1]
        return TrajectorySample(t, last.x, last.y, last.theta, last.v, last.omega)

    def publish_stop_if_armed(self, reason):
        self.left_controller.reset()
        self.right_controller.reset()
        self.publish_twist(0.0, 0.0)
        rospy.logwarn_throttle(2.0, "Follower stopped: %s", reason)

    def run(self):
        rate = rospy.Rate(self.control_rate_hz)

        while not rospy.is_shutdown():
            now = time.monotonic()
            dt = clamp(now - self.last_loop_time, 0.001, 0.2)
            self.last_loop_time = now

            self.update_odometry()

            if not self.enable_controller:
                # Deliberately do not publish in disarmed mode, so this node
                # does not contend with the normal joystick command publisher.
                rate.sleep()
                continue

            stale_sources = []
            for name, encoder, timeout in (
                ("master-left", self.master_left, self.master_message_timeout),
                ("master-right", self.master_right, self.master_message_timeout),
                ("follower-left", self.follower_left, self.follower_message_timeout),
                ("follower-right", self.follower_right, self.follower_message_timeout),
            ):
                if encoder.is_message_stale(now, timeout):
                    stale_sources.append(name)

            if stale_sources:
                self.publish_stop_if_armed("stale encoder message(s): " + ", ".join(stale_sources))
                rate.sleep()
                continue

            ml_v = self.master_left.get_velocity(now, self.encoder_stop_timeout)
            mr_v = self.master_right.get_velocity(now, self.encoder_stop_timeout)
            fl_v = self.follower_left.get_velocity(now, self.encoder_stop_timeout)
            fr_v = self.follower_right.get_velocity(now, self.encoder_stop_timeout)

            master_v = 0.5 * (ml_v + mr_v)
            master_omega = (mr_v - ml_v) / self.master_odom.wheel_base
            master_pose = self.master_odom.pose()
            follower_pose = self.follower_odom.pose()

            if not self.history_initialized:
                self.initialize_history(now, master_pose)
            else:
                self.append_history(now, master_pose, master_v, master_omega)

            reference = self.reference_at(now - self.trajectory_delay)
            if reference is None:
                self.publish_stop_if_armed("delayed leader reference is not available")
                rate.sleep()
                continue

            x_f, y_f, theta_f = follower_pose
            dx = reference.x - x_f
            dy = reference.y - y_f
            position_error = math.hypot(dx, dy)

            if position_error > self.max_tracking_error:
                self.publish_stop_if_armed(
                    "tracking error %.2f m exceeds configured limit %.2f m" %
                    (position_error, self.max_tracking_error))
                rate.sleep()
                continue

            # World-frame position difference transformed into follower frame.
            e_x = math.cos(theta_f) * dx + math.sin(theta_f) * dy
            e_y = -math.sin(theta_f) * dx + math.cos(theta_f) * dy
            e_theta = wrap_to_pi(reference.theta - theta_f)

            # Layer 2: outer trajectory-tracking control law.
            v_cmd = reference.v * math.cos(e_theta) + self.k_x * e_x
            omega_cmd = (
                reference.omega
                + self.k_y * reference.v * e_y
                + self.k_theta * math.sin(e_theta)
            )
            v_cmd = clamp(v_cmd, -self.max_linear_speed, self.max_linear_speed)
            omega_cmd = clamp(omega_cmd, -self.max_angular_speed, self.max_angular_speed)

            # Convert body command to desired wheel velocities (Layer 1 targets).
            left_target = v_cmd - 0.5 * self.follower_odom.wheel_base * omega_cmd
            right_target = v_cmd + 0.5 * self.follower_odom.wheel_base * omega_cmd

            # Pairwise saturation preserves the target wheel-speed ratio/curvature.
            peak = max(abs(left_target), abs(right_target))
            if peak > self.max_wheel_speed:
                scale = self.max_wheel_speed / peak
                left_target *= scale
                right_target *= scale

            # Layer 1: bounded PI correction using follower encoder feedback.
            corrected_left = self.left_controller.update(left_target, fl_v, dt)
            corrected_right = self.right_controller.update(right_target, fr_v, dt)

            # Safety saturation after PI correction; keep both wheel commands
            # scaled together if either exceeds the allowed wheel speed.
            peak = max(abs(corrected_left), abs(corrected_right))
            if peak > self.max_wheel_speed:
                scale = self.max_wheel_speed / peak
                corrected_left *= scale
                corrected_right *= scale

            v_out = 0.5 * (corrected_left + corrected_right)
            omega_out = (corrected_right - corrected_left) / self.follower_odom.wheel_base
            v_out = clamp(v_out, -self.max_linear_speed, self.max_linear_speed)
            omega_out = clamp(omega_out, -self.max_angular_speed, self.max_angular_speed)
            self.publish_twist(v_out, omega_out)

            rospy.loginfo_throttle(
                1.0,
                "REF=(%.2f, %.2f, %.2f) F=(%.2f, %.2f, %.2f) "
                "e=(%.2f, %.2f, %.2f) ref(v,w)=(%.3f, %.3f) "
                "cmd(v,w)=(%.3f, %.3f) wheels target=(%.3f, %.3f) measured=(%.3f, %.3f)" % (
                    reference.x, reference.y, reference.theta,
                    x_f, y_f, theta_f,
                    e_x, e_y, e_theta,
                    reference.v, reference.omega,
                    v_out, omega_out,
                    left_target, right_target, fl_v, fr_v))

            rate.sleep()


if __name__ == "__main__":
    try:
        FollowerController().run()
    except rospy.ROSInterruptException:
        pass
