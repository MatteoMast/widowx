#!/usr/bin/env python

"""
Start ROS node to publish speed and positions of the robotic arm,
torques are disabled in order to test the controller
"""

import rospy, roslib
import time
from math import pi
from arbotix_python.arbotix import ArbotiX
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from servos_parameters import *

class WindowxNode(ArbotiX):
    """Node to control in torque the dynamixel servos"""
    def __init__(self):
        #Initialize arbotix comunications
        print"\nArbotix initialization, wait 10 seconds..."
        ArbotiX.__init__(self)
        for x in xrange(1,21):
            time.sleep(0.5)
            print(str(x*0.5) + "/10s")



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
        self.pos_pub = rospy.Publisher('/windowx_3links/joints_poses', Float32MultiArray, queue_size=1)
        self.vel_pub = rospy.Publisher('/windowx_3links/joints_vels', Float32MultiArray, queue_size=1)

        print"Windowx_3link node created, whaiting for messages in:"
        print"      windowx_2links/torque"
        print"Publishing joints' positions and velocities in:"
        print"      /windowx_3links/joints_poses"
        print"      /windowx_3links/joints_vels"
        #Start publisher
        self.publish()

    def publish(self):
        rad_mx_step = (pi/30) * MX_VEL_UNIT
        #rad_ax_step = (pi/30) * AX_VEL_UNIT
        while not rospy.is_shutdown():
            #MX-* servos poses
            #self.joints_poses[0] = MX_POS_UNIT * (self.getPosition(1) - MX_POS_CENTER)
            self.joints_poses[1] = MX_POS_UNIT * (int(MX_POS_CENTER + MX_POS_CENTER/2) - self.getPosition(2))
            self.joints_poses[2] = MX_POS_UNIT * (self.getPosition(3) - int(MX_POS_CENTER + MX_POS_CENTER/2))
            if self.joints_poses[2] > -0.45:
                rospy.logerr("Joint 2 near jacobian singularity. Shutting Down. Actual position: %frad, singularity in: -0.325rad", self.joints_poses[2])
                rospy.signal_shutdown("Joint 2 near jacobian singularity.")
            elif self.joints_poses[2] > -0.55: #I'm near the Jacobian sigularity => send warning
                rospy.logwarn("Joint 2 is approaching the jacobian singularity (actual position: %frad, singularity in: -0.325rad): Move away from here.", self.joints_poses[2])

            self.joints_poses[3] = MX_POS_UNIT * (self.getPosition(4) - MX_POS_CENTER)
            #AX 12 servos poses
            #self.joints_poses[4] = AX_POS_UNIT * (self.getPosition(5) - AX_POS_CENTER)
            #elf.joints_poses[5] = self.ee_closed

            #MX-* servos vels
            for j in xrange(1,3):
                actualmx_step_speed = self.getSpeed(j+1)
                if actualmx_step_speed < MX_VEL_CENTER:
                    self.joints_vels[j] = rad_mx_step * actualmx_step_speed
                else:
                    self.joints_vels[j] = rad_mx_step * (MX_VEL_CENTER - actualmx_step_speed)

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


def tourn_off_arm():
    ar = ArbotiX()
    print "Disabling servos please wait..."
    #Set all servos torque limits to 0
    for j in xrange(1,7):
        ar.setTorqueLimit(j,0)

    print("Servos disabled. Windowx_3link node closed.")


if __name__ == '__main__':
    #Iitialize the node
    rospy.init_node('windowx_3links_driver')
    rospy.on_shutdown(tourn_off_arm)
    #Create windowx arm object
    wn = WindowxNode()
    rospy.spin()
