cmake_minimum_required(VERSION 3.0.2)
project(purepursuit)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  diagnostic_msgs
  std_srvs
  duckietown_msgs
  message_generation
)

catkin_python_setup()

add_message_files(FILES
  Detection.msg
  DetectionArray.msg
  ObstacleStatus.msg
  SafetyState.msg
  ControllerStatus.msg
  LaneEstimate.msg
)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(CATKIN_DEPENDS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  diagnostic_msgs
  std_srvs
  duckietown_msgs
  message_runtime
)

catkin_install_python(PROGRAMS
  scripts/yolo_detector_node.py
  scripts/tof_obstacle_node.py
  scripts/object_localization_node.py
  scripts/purepursuit_controller_node.py
  scripts/lane_detector_node.py
  scripts/safety_supervisor_node.py
  scripts/benchmark.py
  scripts/validate_topics.py
  scripts/autopilot.py
  scripts/bag_summary.py
  scripts/build_tensorrt_engine.py
  scripts/export_custom_yolo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
