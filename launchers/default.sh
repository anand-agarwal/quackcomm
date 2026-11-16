#!/bin/bash
source /environment.sh
dt-launchfile-init
VEH="${VEHICLE_NAME:-duckiebot}"
# Motion is on by default. The safety supervisor still holds the robot still
# until fsm_node/mode is LANE_FOLLOWING (keyboard_control: A = go, S = stop),
# for 2 s after start, and whenever ToF/YOLO see an obstacle.
# For bench testing, start with ALLOW_MOTION=false.
ALLOW="${ALLOW_MOTION:-true}"
dt-exec roslaunch purepursuit purepursuit_controller.launch veh:="${VEH}" allow_motion:="${ALLOW}"
dt-launchfile-join
