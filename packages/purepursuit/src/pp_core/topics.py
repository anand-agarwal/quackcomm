"""Absolute topic names. The vehicle name comes from VEHICLE_NAME or a parameter.

The final chassis command is the lane-controller source that car_cmd_switch
already selects while the FSM is in LANE_FOLLOWING. Wheel commands stay with
the kinematics node inside dt-car-interface.
"""


def join(vehicle, suffix):
    vehicle = vehicle.strip("/")
    suffix = suffix.strip("/")
    return "/%s/%s" % (vehicle, suffix)


FINAL_COMMAND_SUFFIX = "lane_controller_node/car_cmd"
WHEELS_COMMAND_SUFFIX = "wheels_driver_node/wheels_cmd"
EMERGENCY_STOP_SUFFIX = "wheels_driver_node/emergency_stop"
DESIRED_COMMAND_SUFFIX = "purepursuit/desired_cmd"
LANE_CONTROLLER_SWITCH_SUFFIX = "lane_controller_node/switch"
OBSTACLE_AVOIDANCE_SWITCH_SUFFIX = "obstacle_avoidance_node/switch"
# The ente anti_instagram_node publishes only ~thresholds, not a corrected image.
CORRECTED_IMAGE_SUFFIX = "anti_instagram_node/corrected_image/compressed"
RAW_IMAGE_SUFFIX = "camera_node/image/compressed"
CAMERA_INFO_SUFFIX = "camera_node/camera_info"
GROUND_SEGMENTS_SUFFIX = "ground_projection_node/lineseglist_out"
FILTERED_SEGMENTS_SUFFIX = "lane_filter_node/seglist_filtered"
LANE_POSE_SUFFIX = "lane_filter_node/lane_pose"
TOF_RANGE_SUFFIX = "front_center_tof_driver_node/range"
FSM_MODE_SUFFIX = "fsm_node/mode"
DETECTION_ARRAY_SUFFIX = "purepursuit/detection_array"
RAW_DETECTIONS_SUFFIX = "purepursuit/detections_raw"
DEBUG_IMAGE_SUFFIX = "purepursuit/yolo_debug_image/compressed"
OBSTACLE_DISTANCE_SUFFIX = "purepursuit/obstacle_distance"
SAFETY_STATE_SUFFIX = "purepursuit/safety_state"
CONTROLLER_STATUS_SUFFIX = "purepursuit/controller_status"
LATENCY_SUFFIX = "purepursuit/latency"
LANE_CONFIDENCE_SUFFIX = "purepursuit/lane_confidence"
LANE_ESTIMATE_SUFFIX = "purepursuit/lane_estimate"
LANE_DEBUG_IMAGE_SUFFIX = "purepursuit/lane_debug_image/compressed"
VELOCITY_SUFFIX = "kinematics_node/velocity"
