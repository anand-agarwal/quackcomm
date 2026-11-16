cmake_minimum_required(VERSION 3.0.2)
project(duckiebot_pure_pursuit)
find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs duckietown_msgs)
catkin_package()
catkin_install_python(PROGRAMS
  scripts/pure_pursuit_node.py
  scripts/calibrate_bev.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
