#!/usr/bin/env python

"""
Start ROS node to set the torques and publish speed and velocities
for manuvering the 2nd, 3rd and 4th joints of the windowx arm through the arbotix controller.
"""

import rospy, roslib
import operator
import time
from math import pi
from arbotix_python.arbotix import ArbotiX
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Bool
from servos_parameters import *

class WindowxNode(ArbotiX):
    """Node to control in torque the dynamixel servos"""
    def __init__(self):
        #Initialize arbotix comunications
        print"\nArbotix initialization for r1, wait 10 seconds..."
        ArbotiX.__init__(self, port="/dev/ttyUSB0")
        for x in xrange(1,21):
            time.sleep(0.5)
            print(str(x*0.5) + "/10s for r1")
            if rospy.is_shutdown():
                break
        print"Done."

        #Set inital torque limits
        print"Limiting torques"
        mx28_init_torque_limit = int(MX_TORQUE_STEPS/3)
        mx64_init_torque_limit = int(MX_TORQUE_STEPS/5)
        ax_init_torque_limit = int(AX_TORQUE_STEPS/2)

        self.setTorqueLimit(int(1), mx28_init_torque_limit)
        self.setTorqueLimit(int(2), mx64_init_torque_limit)
        self.setTorqueLimit(int(3), (mx64_init_torque_limit + 100)) #The 3rd servo needs more torque to contrast initial inertias
        self.setTorqueLimit(int(4), mx28_init_torque_limit)
        self.setTorqueLimit(int(5), ax_init_torque_limit)
        self.setTorqueLimit(int(6), ax_init_torque_limit)

        #Go to 0 position for each servo
        print"Going to initialization position..."
        self.setPosition(int(1), int(MX_POS_CENTER))
        self.setPosition(int(4), int(2170))
        self.setPosition(int(3), int(1577))
        self.setPosition(int(2), int(1670))
        self.setPosition(int(5), int(AX_POS_CENTER))
        self.setPosition(int(6), int(AX_POS_CENTER))
        time.sleep(3)
        print("Closing gruppers")
        self.setPosition(int(6), 10)

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
        self.pos_pub = rospy.Publisher('/windowx_3links_r1/joints_poses', Float32MultiArray, queue_size=1)
        self.vel_pub = rospy.Publisher('/windowx_3links_r1/joints_vels', Float32MultiArray, queue_size=1)
        self.pub_rate = rospy.Rate(160)

        #ROS listener for control torues
        self.torque_sub = rospy.Subscriber('windowx_3links_r1/torques', Float32MultiArray, self._torque_callback, queue_size=1)
        self.gripper_sub = rospy.Subscriber('windowx_3links_r1/gripper', Bool, self._gripper_callback, queue_size=1)

        print"\nWindowx_3link_r1 node created, whaiting for messages in:"
        print"      windowx_3links_r1/torque"
        print"Publishing joints' positions and velocities in:"
        print"      /windowx_3links_r1/joints_poses"
        print"      /windowx_3links_r1/joints_vels"
        #Start publisher
        self.publish()

    def _torque_callback(self, msg):
        """
        ROS callback
        """
        goal_torque = msg.data
        goal_torque_steps = [0,0,0]
        direction = [0,0,0]
        #Setup torque steps
        max1 = MX_TORQUE_STEPS/1.5
        max2 = MX_TORQUE_STEPS/1.5
        max3 = MX_TORQUE_STEPS
        goal_torque_steps[0] = min(int(MX64_TORQUE_UNIT * abs(goal_torque[1])), int(max1))
        goal_torque_steps[1] = min(int(MX64_TORQUE_UNIT * abs(goal_torque[2])), int(max2))
        goal_torque_steps[2] = min(int(MX28_TORQUE_UNIT * abs(goal_torque[3])), int(max3))

        if goal_torque_steps[0] == int(max1) or goal_torque_steps[1] == int(max2) or goal_torque_steps[2] == int(max3):
            print("\nWARNING, R1 MAX TORQUE LIMIT REACHED FOR ID: ")
            if goal_torque_steps[0] == int(max1):
                print("2")
            if goal_torque_steps[1] == int(max2):
                print("3")
            if goal_torque_steps[2] == int(max3):
                print("4")
            print("goal_torque:")
            print(goal_torque)
            print("goal_torque_steps:")
            print(goal_torque_steps)

        # print("joints_poses")
        # print(self.joints_poses)
        # print("goal_torque:")
        # print(goal_torque)
        # print("goal_torque_steps:")
        # print(goal_torque_steps)
        #Setup directions------FOR ID 2 THE DIRECTION IS INVERTED!!!!!!
        #ID 3 and 4
        for j in xrange(1,3):
            if goal_torque[1+j] >= 0:
                direction[j] = 1*MX_POS_STEPS #CCW
            else:
                direction[j] = 0*MX_POS_STEPS #CW
        # ID 2
        if goal_torque[1] >= 0:
            direction[0] = 0*MX_POS_STEPS #CCW
        else:
            direction[0] = 1*MX_POS_STEPS #CW
        # print("Direction:")
        # print(direction)
        # print("\n")
        #self._set_torque(goal_torque_steps, direction)
        torque_msg = [[2, goal_torque_steps[0]], [3, goal_torque_steps[1]], [4, goal_torque_steps[2]]]
        direction_msg = [[2, direction[0]], [3, direction[1]], [4, direction[2]]]
        self.syncSetTorque(torque_msg, direction_msg)
        #read present loads:
        # present_load = [0,0,0]
        # for ID in xrange(2,5):
        #     load = self.getLoad(ID)
        #     if load > 1024:
        #         load = load-1024
        #     present_load[ID-2] = load

        # print("\nSetted torque: ")
        # print(goal_torque_steps)
        # print("Present torque: ")
        # print(present_load)
        # print("error: ")
        # print(map(operator.sub, goal_torque_steps, present_load))
    #def _set_torque(self, goal_torque, direction):
        #Set the torques

    def _gripper_callback(self, msg):
        """
        ROS callback
        """
        if msg.data:
            self.setPosition(int(6), 0)
        else:
            self.setPosition(int(6), AX_POS_CENTER)


    def publish(self):
        rad_mx_step = (pi/30) * MX_VEL_UNIT
        #rad_ax_step = (pi/30) * AX_VEL_UNIT
        while not rospy.is_shutdown():
            #MX-* servos poses
            #self.joints_poses[0] = MX_POS_UNIT * (self.getPosition(1) - MX_POS_CENTER)
            present_positions = self.syncGetPos([2, 3, 4])
            present_vels = self.syncGetVel([2,3,4])
            #Check if got good values for position and vels otherwise repeat the reading
            if not -1 in present_vels and not -1 in present_positions:
                self.joints_poses[1] = MX_POS_UNIT * (int(MX_POS_CENTER + MX_POS_CENTER/2) - present_positions[0])
                self.joints_poses[2] = MX_POS_UNIT * (present_positions[1] - int(MX_POS_CENTER + MX_POS_CENTER/2))
                if self.joints_poses[2] > -0.45:
                    rospy.logerr("Joint 2 near jacobian singularity. Shutting Down. Actual position: %frad, singularity in: -0.325rad", self.joints_poses[2])
                    rospy.signal_shutdown("Joint 2 near jacobian singularity.")
                elif self.joints_poses[2] > -0.55: #I'm near the Jacobian sigularity => send warning
                    rospy.logwarn("Joint 2 is approaching the jacobian singularity (actual position: %frad, singularity in: -0.325rad): Move away from here.", self.joints_poses[2])

                self.joints_poses[3] = MX_POS_UNIT * (present_positions[2] - MX_POS_CENTER)
                #AX 12 servos poses
                #self.joints_poses[4] = AX_POS_UNIT * (self.getPosition(5) - AX_POS_CENTER)
                #self.joints_poses[5] = self.ee_closed

                #MX-* servos vels
                for j in xrange(1,4):
                    if present_vels[j-1] < MX_VEL_CENTER:
                        self.joints_vels[j] = rad_mx_step * present_vels[j-1]
                    else:
                        self.joints_vels[j] = rad_mx_step * (MX_VEL_CENTER - present_vels[j-1])
                        if self.joints_vels[j] < -5:
                            print(self.joints_vels[j])
                            print(present_vels[j-1])

                #Invert second joint velocity sign
                self.joints_vels[1] = -1*self.joints_vels[1]
                # #AX 12 servos vels
                # actualax_step_speed = self.getSpeed(5)
                # if actualax_step_speed < AX_VEL_CENTER:
                #     self.joints_vels[4] = rad_ax_step * actualax_step_speed
                # else:
                #     self.joints_vels[4] = rad_ax_step * (AX_VEL_CENTER - actualax_step_speed)

                self.poses_to_pub.data = self.joints_poses
                self.vels_to_pub.data = self.joints_vels
                self.pos_pub.publish(self.poses_to_pub)
                self.vel_pub.publish(self.vels_to_pub)
                self.pub_rate.sleep()
            else:
                rospy.logwarn("Lost packet.")

    def tourn_off_arm(self):
        """
        Disable all servos.
        """
        print "Disabling servos please wait..."
        self.setPosition(int(6), int(AX_POS_CENTER))
        for j in xrange(1,6):
            self.setTorqueLimit(j,0)
        print"Servos disabled. Windowx_3link node closed."



if __name__ == '__main__':
    #Iitialize the node
    rospy.init_node('windowx_3links_driver_r1')
    #Create windowx arm object
    wn = WindowxNode()
    #Handle shutdown
    rospy.on_shutdown(wn.tourn_off_arm)
    rospy.spin()
