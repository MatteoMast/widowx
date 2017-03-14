#!/usr/bin/env python

"""
Start ROS node to pubblish torques for manuvering windowx arm through the v-rep simulator.
"""

import cv2
import rospy, roslib
from math import sin, cos
from windowx_msgs.msg import TargetConfiguration
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
        self.target_sub = rospy.Subscriber('/windowx_3links/target_conf', TargetConfiguration, self._target_callback, queue_size=10)
        self.torque_pub = rospy.Publisher('/torques', Float32MultiArray, queue_size=1)
        #Initial pose, all joints will move to 0 position, and initialization of pose and vels vectors
        self.target_pose = np.array([[1.5708,-1.5708,0]]).T
        self.target_vel = np.array([[0,0,0]]).T
        self.target_acc = np.array([[0,0,0]]).T
        self.torques = Float32MultiArray()
        self.torques_layout = MultiArrayDimension('control_torques', 6, 0)
        self.joints_poses = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joints_vels =  [0.0, 0.0, 0.0, 0.0, 0.0]
        #Control Kd and Kp
        self.KD = np.matrix([[25, 0, 0], [0, 25, 0], [0, 0, 20]])
        self.KP = np.matrix([[380, 0, 0],[0, 380, 0], [0, 0, 250]])
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
        self.target_pose = np.asarray(msg.pos)[np.newaxis].T
        self.target_vel = np.asarray(msg.vel)[np.newaxis].T
        self.target_acc = np.asarray(msg.acc)[np.newaxis].T

        print("\nGoing to:")
        print("Pos: \n" + str(self.target_pose))
        print("Vel: \n" + str(self.target_vel))
        print("Acc: \n" + str(self.target_acc))

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
            err_vels = array_vels[1:4] - self.target_vel
            err_poses = array_poses[1:4] - self.target_pose
            print("velocity error:")
            print(err_vels)
            print("position error:")
            print(err_poses)
            B = np.matrix([[0.0040055721446399998476906034738931*cos(self.joints_poses[2] + self.joints_poses[3]) - 0.0013481452371199999142570291610355*sin(self.joints_poses[2] + self.joints_poses[3]) + 0.011671172651879999466092491395841*cos(self.joints_poses[2]) + 0.0040085638208*cos(self.joints_poses[3]) - 0.0039281369187899997198368480111341*sin(self.joints_poses[2]) + 0.042812399753418998939427354098797,\
                0.0020027860723199999238453017369466*cos(self.joints_poses[2] + self.joints_poses[3]) - 0.00067407261855999995712851458051773*sin(self.joints_poses[2] + self.joints_poses[3]) + 0.0058355863259399997330462456979205*cos(self.joints_poses[2]) + 0.0040085638208*cos(self.joints_poses[3]) - 0.0019640684593949998599184240055671*sin(self.joints_poses[2]) + 0.01625959562072499985284632093574,\
                0.0020027860723199999238453017369466*cos(self.joints_poses[2] + self.joints_poses[3]) - 0.00067407261855999995712851458051773*sin(self.joints_poses[2] + self.joints_poses[3]) + 0.0020042819104*cos(self.joints_poses[3]) + 0.0026794854106086355903769417993171],\
                [0.0020027860723199999238453017369466*cos(self.joints_poses[2] + self.joints_poses[3]) - 0.00067407261855999995712851458051773*sin(self.joints_poses[2] + self.joints_poses[3]) + 0.0058355863259399997330462456979205*cos(self.joints_poses[2]) + 0.0040085638208*cos(self.joints_poses[3]) - 0.0019640684593949998599184240055671*sin(self.joints_poses[2]) + 0.01625959562072499985284632093574,\
                0.0040085638208*cos(self.joints_poses[3]) + 0.01618298062072499985284632093574,\
                0.0020042819104*cos(self.joints_poses[3]) + 0.0026794854106086355903769417993171],
                [0.0020027860723199999238453017369466*cos(self.joints_poses[2] + self.joints_poses[3]) - 0.00067407261855999995712851458051773*sin(self.joints_poses[2] + self.joints_poses[3]) + 0.0020042819104*cos(self.joints_poses[3]) + 0.0026794854106086355903769417993171,\
                0.0020042819104*cos(self.joints_poses[3]) + 0.0026794854106086355903769417993171,\
                0.0026403112045896820614231443819367]])

            C = np.matrix([[- 0.176*self.joints_vels[3]*(0.0038299580599999997564120146620326*cos(self.joints_poses[2] + self.joints_poses[3]) + 0.011379466319999999567302850778105*sin(self.joints_poses[2] + self.joints_poses[3]) + 0.0113879654*sin(self.joints_poses[3])) - 1.0*self.joints_vels[2]*(0.00067407261855999995712851458051773*cos(self.joints_poses[2] + self.joints_poses[3]) + 0.0020027860723199999238453017369466*sin(self.joints_poses[2] + self.joints_poses[3]) + 0.0019640684593949998599184240055671*cos(self.joints_poses[2]) + 0.0058355863259399997330462456979205*sin(self.joints_poses[2])),\
                - 0.176*self.joints_vels[3]*(0.0038299580599999997564120146620326*cos(self.joints_poses[2] + self.joints_poses[3]) + 0.011379466319999999567302850778105*sin(self.joints_poses[2] + self.joints_poses[3]) + 0.0113879654*sin(self.joints_poses[3])) - 1.0*self.joints_vels[1]*(0.00067407261855999995712851458051773*cos(self.joints_poses[2] + self.joints_poses[3]) + 0.0020027860723199999238453017369466*sin(self.joints_poses[2] + self.joints_poses[3]) + 0.0019640684593949998599184240055671*cos(self.joints_poses[2]) + 0.0058355863259399997330462456979205*sin(self.joints_poses[2])) - 1.0*self.joints_vels[2]*(0.00067407261855999995712851458051773*cos(self.joints_poses[2] + self.joints_poses[3]) + 0.0020027860723199999238453017369466*sin(self.joints_poses[2] + self.joints_poses[3]) + 0.0019640684593949998599184240055671*cos(self.joints_poses[2]) + 0.0058355863259399997330462456979205*sin(self.joints_poses[2])),\
                -0.176*(self.joints_vels[1] + self.joints_vels[2] + self.joints_vels[3])*(0.0038299580599999997564120146620326*cos(self.joints_poses[2] + self.joints_poses[3]) + 0.011379466319999999567302850778105*sin(self.joints_poses[2] + self.joints_poses[3]) + 0.0113879654*sin(self.joints_poses[3]))],\
                [self.joints_vels[1]*(0.00067407261855999995712851458051773*cos(self.joints_poses[2] + self.joints_poses[3]) + 0.0020027860723199999238453017369466*sin(self.joints_poses[2] + self.joints_poses[3]) + 0.0019640684593949998599184240055671*cos(self.joints_poses[2]) + 0.0058355863259399997330462456979205*sin(self.joints_poses[2])) - 0.0020042819104*self.joints_vels[3]*sin(self.joints_poses[3]),\
                -0.0020042819104*self.joints_vels[3]*sin(self.joints_poses[3]),\
                -0.0020042819104*sin(self.joints_poses[3])*(self.joints_vels[1] + self.joints_vels[2] + self.joints_vels[3])],\
                [0.0020042819104*self.joints_vels[2]*sin(self.joints_poses[3]) + 0.176*self.joints_vels[1]*(0.0038299580599999997564120146620326*cos(self.joints_poses[2] + self.joints_poses[3]) + 0.011379466319999999567302850778105*sin(self.joints_poses[2] + self.joints_poses[3]) + 0.0113879654*sin(self.joints_poses[3])),\
                0.0020042819104*sin(self.joints_poses[3])*(self.joints_vels[1] + self.joints_vels[2]),0]])

            g = np.array([[0.69474494555999997358275432901564*cos(self.joints_poses[1]) + 0.21649055273999998623105089912144*sin(self.joints_poses[1]) + 0.40336448984999999688544018994207*cos(self.joints_poses[1])*cos(self.joints_poses[2]) - 0.40336448984999999688544018994207*sin(self.joints_poses[1])*sin(self.joints_poses[2]) + 0.1384355808*cos(self.joints_poses[1])*cos(self.joints_poses[2])*cos(self.joints_poses[3]) - 0.1384355808*cos(self.joints_poses[1])*sin(self.joints_poses[2])*sin(self.joints_poses[3]) - 0.1384355808*cos(self.joints_poses[2])*sin(self.joints_poses[1])*sin(self.joints_poses[3]) - 0.1384355808*cos(self.joints_poses[3])*sin(self.joints_poses[1])*sin(self.joints_poses[2])],\
                [0.1384355808*cos(self.joints_poses[1] + self.joints_poses[2] + self.joints_poses[3]) + 0.40336448984999999688544018994207*cos(self.joints_poses[1] + self.joints_poses[2])],\
                [ 0.1384355808*cos(self.joints_poses[1] + self.joints_poses[2] + self.joints_poses[3])]])
            #Compute control torque
            control_from_errors = self.target_acc -np.dot(self.KD, err_vels) - np.dot(self.KP, err_poses)
            print("Derivative contribution: ")
            print(np.dot(self.KD, err_vels))
            print("proportional contribution: ")
            print(np.dot(self.KP, err_poses))
            control_torque = np.dot(C, self.target_vel) + g + np.dot(B, control_from_errors)
            print("Torques: ")
            print(control_torque)
            #Create ROS message
            self.torques.layout.dim = [self.torques_layout]
            # self.torques.layout.dim.size = 6
            # self.torques.layout.dim.stride = 1
            self.torques.layout.data_offset = 0
            self.torques.data = [0.0, control_torque[0], control_torque[1], control_torque[2], 0.0, 0.0]
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
