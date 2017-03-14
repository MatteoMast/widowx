#!/usr/bin/env python

"""
Start ROS node to set the torques and publish speed and velocities
for manuvering the 2nd, 3rd and 4th joints of the windowx arm through the arbotix controller.
"""

import rospy, roslib
import operator
import time
from math import pi
import numpy as np
from arbotix_python.arbotix import ArbotiX
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Bool
from servos_parameters import *
from windowx_driver.srv import *

class WindowxNode(ArbotiX):
    """Node to control in torque the dynamixel servos"""
    def __init__(self, serial_port, robot_name):
        #Initialize arbotix comunications
        print"\nArbotix initialization for " + robot_name + ", wait 10 seconds..."
        ArbotiX.__init__(self, port=serial_port)
        for x in xrange(1,21):
            time.sleep(0.5)
            print(str(x*0.5) + "/10s for " + robot_name)
            if rospy.is_shutdown():
                break
        print robot_name + " Done."

        #reset vel limit
        print"Reset max vels for " + robot_name
        max_rpm = 10.0
        max_rad_s = max_rpm * 2*pi/60
        max_speed_steps = int(max_rpm/MX_VEL_UNIT)
        self.setSpeed(2, max_speed_steps)
        self.setSpeed(3, max_speed_steps)
        self.setSpeed(4, max_speed_steps)

        #reset PID values
        internal_PID = [32, 0, 0]
        self.setPID(2, internal_PID[0], internal_PID[1], internal_PID[2])
        self.setPID(3, internal_PID[0], internal_PID[1], internal_PID[2])
        self.setPID(4, internal_PID[0], internal_PID[1], internal_PID[2])

        #Set inital torque limits
        print"Limiting torques for " + robot_name
        mx28_init_torque_limit = int(MX_TORQUE_STEPS/3)
        mx64_init_torque_limit = int(MX_TORQUE_STEPS/5)
        ax_init_torque_limit = int(AX_TORQUE_STEPS/4)

        max_torque_msg = [[1, mx28_init_torque_limit], [2, mx64_init_torque_limit], [3, mx64_init_torque_limit + 100], [4, mx28_init_torque_limit], [5, ax_init_torque_limit], [6,ax_init_torque_limit]]
        pos_msg = [[1, int(MX_POS_CENTER)], [2, int(1710)], [3, int(1577)], [4, int(2170)], [5, int(AX_POS_CENTER)], [6,int(AX_POS_CENTER)]]

        self.syncSetTorque(max_torque_msg, pos_msg)
        time.sleep(3)

        print(robot_name + ": Closing gruppers")
        self.setPosition(int(6), 50)

        #Limit joints velocities
        max_rpm = 1.5
        max_rad_s = max_rpm * 2*pi/60
        print(robot_name + ": Limiting joints velocities at: "+ str(max_rpm) +"rpm = "+ str(max_rad_s) +"rad/s")
        max_speed_steps = int(max_rpm/MX_VEL_UNIT)
        self.setSpeed(2, max_speed_steps)
        self.setSpeed(3, max_speed_steps)
        self.setSpeed(4, max_speed_steps)

        #Setup PID parameters
        internal_PID = [8, 0, 0]
        self.setPID(2, internal_PID[0], internal_PID[1], internal_PID[2])
        self.setPID(3, internal_PID[0], internal_PID[1], internal_PID[2])
        self.setPID(4, internal_PID[0], internal_PID[1], internal_PID[2])


        print robot_name + " ready, setting up ROS topics..."

        #Setup velocities and positions vectors and messages
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
        self.pos_pub = rospy.Publisher('/windowx_3links_'+ robot_name +'/joints_poses', Float32MultiArray, queue_size=1)
        self.vel_pub = rospy.Publisher('/windowx_3links_'+ robot_name +'/joints_vels', Float32MultiArray, queue_size=1)
        self.pub_rate = rospy.Rate(150)

        #ROS listener for control torues
        self.torque_sub = rospy.Subscriber('windowx_3links_'+ robot_name +'/torques', Float32MultiArray, self._torque_callback, queue_size=1)
        self.gripper_sub = rospy.Subscriber('windowx_3links_'+ robot_name +'/gripper', Bool, self._gripper_callback, queue_size=1)

        #Topic for checkings
        self.check_pub = rospy.Publisher('/torque_check', Float32MultiArray, queue_size=1)
        #Initialize check message
        self.check = Float32MultiArray()
        self.check_layout = MultiArrayDimension('torque_check', 6, 0)
        self.check.layout.dim = [self.check_layout]
        self.check.layout.data_offset = 0

        #ROS service for security stop
        self.sec_stop_server = rospy.Service('windowx_3links_' + robot_name + '/security_stop', SecurityStop, self._sec_stop)

        #Frequency estimation for written torques
        self.cycle_count = 1
        self.freq_sum = 0
        self.iter_time = rospy.get_rostime()
        self.old_time = self.iter_time
        self.first_torque = True

        print"\nWindowx_3link_" + robot_name + " node created, whaiting for messages in:"
        print"      windowx_3links_" + robot_name + "/torque"
        print"Publishing joints' positions and velocities in:"
        print"      /windowx_3links_" + robot_name + "/joints_poses"
        print"      /windowx_3links_" + robot_name + "/joints_vels"
        print"Scurity stop server running: windowx_3links_" + robot_name + "/security_stop"
        #Start publisher
        self.publish()

    def _compute_currents(self, torques, directions):
        """
        Compute currents from torques
        """
        currents = [np.sign(torques[0])*0.0338*(torques[0]**2) + 0.7592*torques[0], \
                    np.sign(torques[1])*0.0338*(torques[1]**2) + 0.7592*torques[1], \
                    np.sign(torques[2])*0.2258*(torques[2]**2) + 0.4850*torques[2] + self._gain(np.sign(directions[2]),np.sign(torques[2]))*1.5*0.1514]
        return currents

    def _gain(self, direc, tau):
        """
        Compute the gain for the costant term in current computation, arguments must be the signs of direction and torque
        """
        if direc == tau:
            return direc
        else:
            return 2*direc

    def _torque_callback(self, msg):
        """
        ROS callback
        """

        #Initialize freqency estimation
        if self.first_torque:
            #Enable torque control in mx64 servos
            self.enableTorqueControl(2)
            self.enableTorqueControl(3)
            #set torque limits
            self.setTorqueLimit((2), int(MX_TORQUE_STEPS/2) + 1)
            self.setTorqueLimit((3), int(MX_TORQUE_STEPS/2) + 1)
            old_time = rospy.get_rostime()
            self.first_torque = False

        goal_current = self._compute_currents(msg.data[0:3], msg.data[3:6])
        goal_current_steps = [0,0,0]
        direction = [0,0,0]
        # print("\nchecks:")
        # print(goal_current)
        # print(msg.data[0:3])
        # print(msg.data[3:6])
        #Setup torque steps
        max1 = MX_TORQUE_STEPS/2
        max2 = MX_TORQUE_STEPS/2
        max3 = MX_TORQUE_STEPS/2
        goal_current_steps[0] = min(int(MX64_CURRENT_UNIT * abs(goal_current[0])), int(max1))
        goal_current_steps[1] = min(int(MX64_CURRENT_UNIT * abs(goal_current[1])), int(max2))
        goal_current_steps[2] = min(int(MX28_CURRENT_UNIT * abs(goal_current[2])), int(max3))

        if goal_current_steps[0] == int(max1) or goal_current_steps[1] == int(max2) or goal_current_steps[2] == int(max3):
            print("\nWARNING, "+ robot_name +" MAX TORQUE LIMIT REACHED FOR ID: ")
            if goal_current_steps[0] == int(max1):
                print("2")
            if goal_current_steps[1] == int(max2):
                print("3")
            if goal_current_steps[2] == int(max3):
                print("4")
            print("goal_current:")
            print(goal_current)
            print("goal_current_steps:")
            print(goal_current_steps)

        #Setup directions------FOR ID 2 THE DIRECTION IS INVERTED!!!!!!
        #ID 4
        if goal_current[2] >= 0:
            direction[2] = MX_POS_STEPS - 10 #CCW
        else:
            direction[2] = 10 #CW
        
        # ID 1 and 2
        if goal_current[0] >= 0:
            goal_current_steps[0] = 1024 + goal_current_steps[0] #CW

        if goal_current[1] < 0:
            goal_current_steps[1] = 1024 + goal_current_steps[1] #CW

        #MX 28 torque message
        torque_msg = [[4, goal_current_steps[2]]]
        direction_msg = [[4, direction[2]]]
        self.syncSetTorque(torque_msg, direction_msg)
        self.syncSetTorqueGoal([[2, goal_current_steps[0]], [3, goal_current_steps[1]]])

    def _gripper_callback(self, msg):
        """
        ROS callback
        """
        if msg.data:
            self.setPosition(int(6), 50)
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
                    rospy.logerr(robot_name + ": Joint 2 near jacobian singularity. Shutting Down. Actual position: %frad, singularity in: -0.325rad", self.joints_poses[2])
                    rospy.signal_shutdown(robot_name + ": Joint 2 near jacobian singularity.")
                elif self.joints_poses[2] > -0.55: #I'm near the Jacobian sigularity => send warning
                    rospy.logwarn(robot_name + ": Joint 2 is approaching the jacobian singularity (actual position: %frad, singularity in: -0.325rad): Move away from here.", self.joints_poses[2])

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
                rospy.logwarn(robot_name + ": Lost packet at %fs", rospy.get_rostime().to_sec()) # If getting lost packets check return delay of servos or reduce publish rate for torques and/or joints vels and poses

    def _sec_stop(self, req):
        rospy.logerr(req.reason)
        rospy.signal_shutdown(req.reason)

    def tourn_off_arm(self):
        """
        Disable all servos.
        """
        print robot_name + ": Disabling servos please wait..."
        self.disableTorqueControl(2)
        self.disableTorqueControl(3)
        max_torque_msg = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, int(AX_TORQUE_STEPS/4)]]
        pos_msg = [[1, int(MX_POS_CENTER)], [2, int(1710)], [3, int(1577)], [4, int(2170)], [5, int(AX_POS_CENTER)], [6, int(AX_POS_CENTER)]]
        self.syncSetTorque(max_torque_msg, pos_msg)

        print robot_name + ": Servos disabled. Driver node closed."


if __name__ == '__main__':
    #Iitialize the node
    rospy.init_node("windowx_3links")
    robot_name = rospy.get_param(rospy.get_name() + "/robot_name", "r1")
    serial_port = rospy.get_param(rospy.get_name() + "/serial_port", "/dev/ttyUSB0")
    #Create windowx arm object
    wn = WindowxNode(serial_port, robot_name)
    #Handle shutdown
    rospy.on_shutdown(wn.tourn_off_arm)
    rospy.spin()
