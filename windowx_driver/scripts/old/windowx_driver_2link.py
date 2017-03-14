#!/usr/bin/env python

"""
Start ROS node to pubblish torques for manuvering windowx arm through the arbotix controller.
All the messages need to be pubblished in the windowx_arm/torque topic.
"""

import rospy, roslib
import time
from math import pi
from arbotix_python.arbotix import ArbotiX
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from servos_parameters import *

class WindowxNode(ArbotiX):
    """Node to control in torque the dynamixel servos"""
    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        #Initialize arbotix comunications
        print"\nArbotix initialization, wait 10 seconds..."
        ArbotiX.__init__(self, port, baud)
        for x in xrange(1,21):
            time.sleep(0.5)
            print(str(x*0.5) + "/10s")

        print"Done."
        #Set inital torque limits
        print"Limiting torques"
        mx28_init_torque_limit = int(MX_TORQUE_STEPS/3)
        mx64_init_torque_limit = int(MX_TORQUE_STEPS/5)
        ax_init_torque_limit = int(AX_TORQUE_STEPS/5)

        self.setTorqueLimit(int(1), mx28_init_torque_limit)
        self.setTorqueLimit(int(2), mx64_init_torque_limit)
        self.setTorqueLimit(int(3), (mx64_init_torque_limit + 100)) #The 3rd servo needs more torque to contrast initial inertias
        self.setTorqueLimit(int(4), mx28_init_torque_limit)
        self.setTorqueLimit(int(5), ax_init_torque_limit)
        self.setTorqueLimit(int(6), ax_init_torque_limit)
        #Go to 0 position for each servo
        print"Going to initialization position..."
        self.setPosition(int(1), int(MX_POS_CENTER))
        self.setPosition(int(4), int(MX_POS_CENTER))
        self.setPosition(int(3), int(MX_POS_CENTER))
        self.setPosition(int(2), int(MX_POS_CENTER))
        self.setPosition(int(5), int(AX_POS_CENTER))
        self.setPosition(int(6), int(AX_POS_CENTER))

        print"Arm ready, setting up ROS topics..."

        #Setupr velocities and positions vectors and messages
        self.joints_poses = [0,0,0,0,0,0]
        self.joints_vels = [0,0,0,0,0]
        self.ee_closed = 0
        self.vels_to_pub = Float32MultiArray()
        self.poses_to_pub = Float32MultiArray()
        self.poses_layout = MultiArrayDimension('joints_poses', 6, 0)
        self.vels_layout = MultiArrayDimension('joints_vels', 5, 0)
        self.poses_to_pub.layout.dim = [self.poses_layout]
        self.poses_to_pub.layout.data_offset = 0
        self.vels_to_pub.layout.dim = [self.vels_layout]
        self.vels_to_pub.layout.data_offset = 0

        #ROS pubblisher for joint velocities and positions
        self.pos_pub = rospy.Publisher('/windowx_2links/joints_poses', Float32MultiArray, queue_size=1)
        self.vel_pub = rospy.Publisher('/windowx_2links/joints_vels', Float32MultiArray, queue_size=1)

        #ROS listener for control torues
        self.torque_sub = rospy.Subscriber('windowx_2links/torques', Float32MultiArray, self._torque_callback, queue_size=1)

        print"Windowx_2link node created, whaiting for messages in:"
        print"      windowx_2links/torque"
        print"Publishing joints' positions and velocities in:"
        print"      /windowx_2links/joints_poses"
        print"      /windowx_2links/joints_vels"

        self.publish()

    def _torque_callback(self, msg):
        """
        ROS callback
        """
        goal_torque = msg.data
        goal_torque_steps = [0,0]
        direction = [0,0]
        #Setup torque steps
        goal_torque_steps[0] = min(int(MX64_TORQUE_UNIT * abs(goal_torque[2])), MX64_MAX_TORQUE_LIMIT)
        goal_torque_steps[1] = min(int(MX28_TORQUE_UNIT * abs(goal_torque[3])), MX28_MAX_TORQUE_LIMIT)
        # print("joints_poses")
        # print(self.joints_poses)
        # print("goal_torque:")
        # print(goal_torque)
        # print("goal_torque_steps:")
        # print(goal_torque_steps)
        #Setup directions------FOR ID 2 THE DIRECTION IS INVERTED!!!!!!
        for j in xrange(0,2):
            if goal_torque[2+j] >= 0:
                direction[j] = 1 #CCW
            else:
                direction[j] = 0 #CW
        # print("Direction:")
        # print(direction)
        # print("\n")


        self._set_torque(goal_torque_steps, direction)

    def _set_torque(self, goal_torque, direction):
        #Set the torques
        self.setTorque(3, goal_torque[0], direction[0])
        self.setTorque(4, goal_torque[1], direction[1])

    def publish(self):
        while not rospy.is_shutdown():
            #MX-* servos poses
            self.joints_poses[0] = MX_POS_UNIT * (self.getPosition(1) - MX_POS_CENTER)
            self.joints_poses[1] = MX_POS_UNIT * (int(MX_POS_CENTER + MX_POS_CENTER/2) - self.getPosition(2))
            self.joints_poses[2] = MX_POS_UNIT * (self.getPosition(3) - int(MX_POS_CENTER + MX_POS_CENTER/2))
            self.joints_poses[3] = MX_POS_UNIT * (self.getPosition(4) - MX_POS_CENTER)
            #AX 12 servos poses
            self.joints_poses[4] = AX_POS_UNIT * (self.getPosition(5) - AX_POS_CENTER)
            self.joints_poses[5] = self.ee_closed

            self.poses_to_pub.data = self.joints_poses

            #MX-* servos vels
            for j in xrange(0,4):
                actualmx_step_speed = self.getSpeed(j+1)
                if actualmx_step_speed < MX_VEL_CENTER:
                    self.joints_vels[j] = (pi/30) * MX_VEL_UNIT * actualmx_step_speed
                    # if self.joints_vels[j] > 5:
                    #     print("Speeds CCW:")
                    #     print(actualmx_step_speed)
                    #     print(self.joints_vels[j])
                else:
                    self.joints_vels[j] = (pi/30) * MX_VEL_UNIT * (MX_VEL_CENTER - actualmx_step_speed)
                    # if self.joints_vels[j] > 5:
                    #     print("Speeds CW:")
                    #     print(actualmx_step_speed)
                    #     print(self.joints_vels[j])
            #Invert second joint velocity sign
            self.joints_vels[1] = -1*self.joints_vels[1]
            #AX 12 servos vels
            actualax_step_speed = self.getSpeed(5)
            if actualax_step_speed < AX_VEL_CENTER:
                self.joints_vels[4] = (pi/30) * AX_VEL_UNIT * actualax_step_speed
            else:
                self.joints_vels[4] = (pi/30) * AX_VEL_UNIT * (AX_VEL_CENTER - actualax_step_speed)

            self.vels_to_pub.data = self.joints_vels
            self.pos_pub.publish(self.poses_to_pub)
            self.vel_pub.publish(self.vels_to_pub)


def tourn_off_arm():
    ar = ArbotiX()
    print "Disabling servos please wait..."
    #Set all servos torques limits to 0
    for j in xrange(1,7):
        ar.setTorqueLimit(j,0)

    print("Servos disabled. Windowx_2link node closed.")


if __name__ == '__main__':
    #Iitialize the node
    rospy.init_node('windowx_2links')
    rospy.on_shutdown(tourn_off_arm)
    #Create windowx arm object
    wn = WindowxNode()
    rospy.spin()
