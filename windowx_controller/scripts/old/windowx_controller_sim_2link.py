#!/usr/bin/env python

"""
Start ROS node to pubblish torques for manuvering windowx arm through the v-rep simulator.
"""

import cv2
import rospy, roslib
import math
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np

class WindowxController():
    """
    Class to compute and pubblish joints torques
    """
    def __init__(self):
        #initialize pose, velocity listeners and torques publisher
        self.pose_sub = rospy.Subscriber('/joints_poses', Float32MultiArray, self._pose_callback, queue_size=1)
        self.vel_sub = rospy.Subscriber('/joints_vels', Float32MultiArray, self._vel_callback, queue_size=1)
        self.target_sub = rospy.Subscriber('/target_pose', Float32MultiArray, self._target_callback, queue_size=10)
        self.torque_pub = rospy.Publisher('/torques', Float32MultiArray, queue_size=1)
        #Initial pose, all joints will move to 0 position, and initialization of pose and vels vectors
        self.target_pose = np.array([[0,0]]).T
        self.target_vel = np.array([[0,0]]).T
        self.torques = Float32MultiArray()
        self.torques_layout = MultiArrayDimension('control_torques', 6, 0)
        self.joints_poses = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joints_vels =  [0.0, 0.0, 0.0, 0.0, 0.0]
        #Control Kd and Kp
        self.KD = np.matrix([[40, 0], [0, 40]])
        self.KP = np.matrix([[550, 0], [0, 450]])
        #Challer check
        self.pose_call = False
        self.vel_call = False


    def _pose_callback(self, msg):
        """
        ROS callback to get the joint poses
        """
        self.joints_poses = msg.data
        self.compute_torques('pose')

    def _vel_callback(self, msg):
        """
        ROS callback to get the joint velocities
        """
        self.joints_vels = msg.data
        self.compute_torques('vel')

    def _target_callback(self, msg):
        """
        ROS callback to get the target position
        """
        self.target_pose = np.asarray(msg.data)[np.newaxis].T
        print("going to:")
        print(self.target_pose)

    def compute_torques(self, caller):
        """
        Compute and pubblish torques values for 3rd and 4th joints
        """
        if caller == 'pose':
            self.pose_call = True
        if caller == 'vel':
            self.vel_call = True
        #If both vels and poses has called compute torques
        if self.pose_call and self.vel_call:
            #Reset checkers
            self.pose_call = False
            self.vel_call = False
            #Vels and poses
            # print "Heard:"
            # print "    ".join(str(n) for n in self.joints_vels)
            # print "    ".join(str(n) for n in self.joints_poses)
            #Compute B g and C matrices
            array_vels = np.asarray(self.joints_vels)[np.newaxis].T
            array_poses = np.asarray(self.joints_poses)[np.newaxis].T
            # print("array_vels")
            # print(array_vels[2:4])
            # print("array_poses")
            # print(array_poses[2:4])
            err_vels = array_vels[2:4] - self.target_vel
            err_poses = array_poses[2:4] - self.target_pose
            # print("velocity error:")
            # print(err_vels)
            # print("position error:")
            # print(err_poses)
            B = np.matrix([[0.00788496260627 + 0.0040085638208*math.cos(self.joints_poses[2]),0.00264010717227 + 0.0020042819104*math.cos(self.joints_poses[2])],\
                [ 0.00264010717227+ 0.0020042819104*math.cos(self.joints_poses[2]),0.00264010717227]])
            C = np.matrix([[-0.0020042819104*math.sin(self.joints_poses[2])*self.joints_vels[2],-0.0020042819104*math.sin(self.joints_poses[2])*(self.joints_vels[1]+self.joints_vels[2])],\
                [0.0020042819104*math.sin(self.joints_poses[2])*self.joints_vels[1],0]])
            g = np.array([[0.34543627608*math.cos(self.joints_poses[1])+0.13839324576*math.cos(self.joints_poses[1]+self.joints_poses[2])],\
                [0.13839324576*math.cos(self.joints_poses[1]+self.joints_poses[2])]])
            #Compute control torque
            control_from_errors = -np.dot(self.KD, err_vels) - np.dot(self.KP, err_poses)
            #print(control_from_errors)
            control_torque = np.dot(C, array_vels[2:4]) + g + np.dot(B, control_from_errors)
            print("Torques: ")
            print(control_torque)
            #Create ROS message
            self.torques.layout.dim = [self.torques_layout]
            # self.torques.layout.dim.size = 6
            # self.torques.layout.dim.stride = 1
            self.torques.layout.data_offset = 0
            self.torques.data = [0.0, 0.0, control_torque[0], control_torque[1], 0.0, 0.0]
            self.torque_pub.publish(self.torques)



if __name__ == '__main__':
    #Iitialize the node
    rospy.init_node('windowx_controller')
    #Create windowx controller object
    wc = WindowxController()

    try:
        print"WindowX controller node created"
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS WindowX controller node"
