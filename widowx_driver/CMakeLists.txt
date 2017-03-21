cmake_minimum_required(VERSION 2.8.3)
project(widowx_driver)

find_package(catkin REQUIRED COMPONENTS
rospy
roslib
message_generation
arbotix_python
std_msgs
)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Torque.msg
# )

# Generate services in the 'srv' folder
 add_service_files(
   FILES
   SecurityStop.srv
 )

 ## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
 )

catkin_package(CATKIN_DEPENDS message_runtime std_msgs)

