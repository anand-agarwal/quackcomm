# Duckiebot Pure Pursuit lane follower

ROS1 `rospy` package for a DB21J-style Duckiebot. The node detects the white
RIGHT boundary of the lane, fits it in a metric bird's-eye view (BEV), offsets
the curve inward by half a lane width to estimate the lane centre, and uses
Pure Pursuit to steer toward a point at a measured arc length along that curve.

## Assumptions

- ROS1 with `rospy` and `duckietown_msgs` installed.
- Camera topic: `/<VEHICLE_NAME>/camera_node/image/compressed`.
- Chassis command topic: `/<VEHICLE_NAME>/car_cmd_switch_node/cmd` (`Twist2DStamped`).
- The white line being followed is the right boundary of the robot's current lane.
- The calibration rectangle is flat, level, and measured. Default dimensions are
  1.0 m wide x 1.5 m long, centred on the robot. Its near edge is 0.15 m ahead
  of the axle; the far edge is 1.65 m ahead. Change the YAML if you use other
  dimensions.

## 1. Copy/build

Copy this folder into your template's catkin workspace `src/` directory:

```bash
cd ~/catkin_ws/src
# copy duckiebot_pure_pursuit here
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

If your Duckietown template uses a different workspace path, use that path.

## 2. Calibrate the ground plane (required)

Put the robot on a flat surface. Mark a rectangle on the floor with the exact
width/length set in `config/controller.yaml` (default 1.0 m x 1.5 m), centred
on the robot, with its long axis aligned with the robot's forward axis. The
near edge must be `near_distance_m` ahead of the axle.

Run the click helper in the robot's ROS environment:

```bash
rosrun duckiebot_pure_pursuit calibrate_bev.py _output:=/tmp/bev_points.yaml
```

Click the four visible floor corners in this order:
1. far-left (robot-left)
2. far-right (robot-right)
3. near-right
4. near-left

Copy the resulting `src_points: [...]` line into
`duckiebot_pure_pursuit/config/controller.yaml`, replacing `src_points: []`.
Rebuild/re-source if required. The four clicks must correspond to the measured
rectangle corners; arbitrary clicks do not provide metric scale.

## 3. Run

Disable the stock lane controller / any other node publishing drive commands
while testing. Do not run two controllers on the same command topic.

```bash
roslaunch duckiebot_pure_pursuit pure_pursuit.launch veh:=YOUR_DUCKIEBOT_NAME
```

Debug BEV image:

```bash
rqt_image_view /YOUR_DUCKIEBOT_NAME/pure_pursuit_lane_follower/debug
```

Confidence:

```bash
rostopic echo /YOUR_DUCKIEBOT_NAME/pure_pursuit_lane_follower/confidence
```

## 4. First tuning sequence

1. Lift the wheels or use a stand and confirm the published command sign before
   putting the robot on the floor. A left target should produce positive omega.
2. Start with `max_speed: 0.08` and `lookahead_min_m: 0.30`.
3. Tune `white_hsv_lower/upper` until the BEV debug view shows only the white
   boundary as a continuous line. Reduce/raise V for lighting; adjust S for
   warm lighting or faded markings.
4. Adjust `lane_width_m` to the actual map. An incorrect lane width creates a
   persistent lateral offset in the reconstructed centreline.
5. Tune `max_speed`, `lookahead_min_m`, and `max_lateral_accel` only after the
   line fit is stable. Shorter lookahead follows tight bends more closely but
   can become twitchy; longer lookahead is smoother but cuts corners.

## Safety / limitations

- This is a starting implementation, not a validated autonomous-driving stack.
  Test in a closed, empty area at low speed with an accessible emergency stop.
- If the line estimate is missing/low confidence, the node commands zero speed.
- If the command topic is not accepted by your template's command mux, verify
  the active command source and topic names with `rostopic list` and the
  Duckietown launch configuration. Do not bypass the safety mux to make it move.
- White-only tracking assumes the white stripe is visible and is the lane's
  right boundary. It does not handle intersections, stop lines, obstacles,
  lane changes, or ambiguous multiple white markings.
- The polynomial is local (quadratic) and the BEV assumes a planar road.
- The camera perspective transform must be recalibrated if camera pose, mount,
  image resolution/crop, or calibration rectangle changes.
