cmake_minimum_required(VERSION 2.8.3)
project(widowx_controller)

find_package(catkin REQUIRED COMPONENTS
rospy
roslib
message_generation
cv_bridge
std_msgs
widowx_driver
)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Torque.msg
# )

# Generate services in the 'srv' folder
 # add_service_files(
 #   FILES
 #   computeSkeleton.srv
 # )

 ## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
 )

catkin_package(CATKIN_DEPENDS message_runtime std_msgs)

