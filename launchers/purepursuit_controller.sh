#!/bin/bash
source /environment.sh
dt-launchfile-init
VEH="${VEHICLE_NAME:-duckiebot}"
ALLOW="${ALLOW_MOTION:-true}"
dt-exec roslaunch purepursuit purepursuit_controller.launch veh:="${VEH}" allow_motion:="${ALLOW}"
dt-launchfile-join
