cmake_minimum_required(VERSION 3.0.2)
project(quackcomm)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
)

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/quackcomm_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)