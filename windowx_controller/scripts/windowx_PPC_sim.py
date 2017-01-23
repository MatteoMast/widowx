#!/usr/bin/env python

"""
Start PPC controller for manuvering windowx arms through the v-rep simulator.
"""

import cv2
import rospy, roslib
from math import sin, cos, pi, sqrt
from windowx_msgs.msg import TargetConfiguration
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np
from numpy.linalg import inv
from windowx_arm import *

class WindowxController():
    """Class to compute and pubblish joints torques"""
    def __init__(self):
        #Object parameters
        self.m_obj = 0.2 #Kg
        self.Co = np.matrix([[0,0,0],[0,0,0],[0,0,0]])
        self.go = np.array([[0], [self.m_obj*9.81], [0]])
        i_obj = 0.0067
        self.Io = np.matrix([[i_obj, 0, 0],[0, i_obj, 0],[0,0, i_obj]])
        #Load share coefficients
        self.c1 = 0.5
        self.c2 = 0.5
        #initialize pose, velocity listeners and torques publisher
        #Robot1
        self.r1_pose_sub = rospy.Subscriber('/robot1/joints_poses', Float32MultiArray, self._r1_pose_callback, queue_size=1)
        self.r1_vel_sub = rospy.Subscriber('/robot1/joints_vels', Float32MultiArray, self._r1_vel_callback, queue_size=1)
        self.r1_torque_pub = rospy.Publisher('/robot1/torques', Float32MultiArray, queue_size=1)
        #Robot2
        self.r2_pose_sub = rospy.Subscriber('/robot2/joints_poses', Float32MultiArray, self._r2_pose_callback, queue_size=1)
        self.r2_vel_sub = rospy.Subscriber('/robot2/joints_vels', Float32MultiArray, self._r2_vel_callback, queue_size=1)
        self.r2_torque_pub = rospy.Publisher('/robot2/torques', Float32MultiArray, queue_size=1)
        #Object
        self.obj_pose_sub = rospy.Subscriber('/object_position', Float32MultiArray, self._obj_pose_callback, queue_size=1)
        self.obj_vel_sub = rospy.Subscriber('/object_vel', Float32MultiArray, self._obj_vel_callback, queue_size=1)
        #Trajectory listener
        self.target_sub = rospy.Subscriber('/object/target_conf', TargetConfiguration, self._target_callback, queue_size=1)

        #Initial pose, all joints will move to the initial target position, and initialization of pose and vels vectors
        #Here the target configuration is x_e = [x,y,orientation] x_e_dot x_e_ddot of the end effector wrt the inertial frame of the robot
        self.target_pose = np.array([[0.385,0.13,0.0]]).T
        self.target_vel = np.array([[0.0,0.0,0.0]]).T
        self.target_acc = np.array([[0.0,0.0,0.0]]).T
        #Robot1
        self.r1_joints_poses = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.r1_joints_vels =  [0.0, 0.0, 0.0, 0.0, 0.0]
        self.r1_close_gripper = 1
        #Robot2
        self.r2_joints_poses = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.r2_joints_vels =  [0.0, 0.0, 0.0, 0.0, 0.0]
        self.r2_close_gripper = 1
        #Obj
        self.obj_pose = [0.0, 0.0, 0.0]
        self.obj_vel =  [0.0, 0.0, 0.0]
        #Control Kd and Kp
        self.Kv = np.matrix([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
        self.K = np.matrix([[50, 0, 0],[0, 50, 0], [0, 0, 50]])
        #Initialize torque message
        self.torques1 = Float32MultiArray()
        self.torques2 = Float32MultiArray()
        self.torques_layout = MultiArrayDimension('control_torques', 6, 0)
        self.torques1.layout.dim = [self.torques_layout]
        self.torques1.layout.data_offset = 0
        self.torques2.layout.dim = [self.torques_layout]
        self.torques2.layout.data_offset = 0

        #Initialize timers
        self.start = rospy.get_rostime()
        self.actual_time = rospy.get_rostime()

        self.pose_call1 = False
        self.vel_call1 = False
        self.pose_call2 = False
        self.vel_call2 = False
        self.obj_pose_call = False
        self.obj_vel_call = False
        self.first_iteration = True
        print("\nWindowX controller node created")
        print("\nWaiting for target position, velocity and acceleration...")

    #SENSING CALLBACKS
    def _r1_pose_callback(self, msg):
        """
        ROS callback to get the joint poses
        """
        self.r1_joints_poses = msg.data
        self.compute_torques('pose1')

    def _r2_pose_callback(self, msg):
        """
        ROS callback to get the joint poses
        """
        self.r2_joints_poses = msg.data
        self.compute_torques('pose2')

    def _obj_pose_callback(self, msg):
        """
        ROS callback to get the joint poses
        """
        self.obj_pose = [msg.data[0] + 0.65, msg.data[1] - 0.125, msg.data[2]]
        self.compute_torques('obj_pose')

    def _r1_vel_callback(self, msg):
        """
        ROS callback to get the joint velocities
        """
        self.r1_joints_vels = msg.data
        self.compute_torques('vel1')

    def _r2_vel_callback(self, msg):
        """
        ROS callback to get the joint velocities
        """
        self.r2_joints_vels = msg.data
        self.compute_torques('vel2')

    def _obj_vel_callback(self, msg):
        """
        ROS callback to get the joint velocities
        """
        self.obj_vel = [msg.data[0], msg.data[1], msg.data[2]]
        self.compute_torques('obj_vel')

    def _target_callback(self, msg):
        """
        ROS callback to get the target position
        """
        self.target_pose = np.asarray(msg.pos)[np.newaxis].T
        self.target_vel = np.asarray(msg.vel)[np.newaxis].T
        self.target_acc = np.asarray(msg.acc)[np.newaxis].T

    #CONTROLLER
    def compute_torques(self,caller):
        """
        Compute and pubblish torques values for 3rd and 4th joints
        """
        if caller == 'pose1':
            self.pose_call1 = True
        if caller == 'vel1':
            self.vel_call1 = True
        if caller == 'pose2':
            self.pose_call2 = True
        if caller == 'vel2':
            self.vel_call2 = True
        if caller == 'obj_pose':
            self.obj_pose_call = True
        if caller == 'obj_vel':
            self.obj_vel_call = True

        if self.pose_call1 and self.vel_call1 and self.pose_call2 and self.vel_call2 and self.obj_pose_call and self.obj_vel_call:

            self.pose_call1 = False
            self.vel_call1 = False
            self.pose_call2 = False
            self.vel_call2 = False
            self.obj_pose_call = False
            self.obj_vel_call = False

            #Start timer
            if self.first_iteration:
                self.start = rospy.get_rostime()
                self.first_iteration = False
            #Update simulation time
            self.actual_time = rospy.get_rostime() - self.start

            r1_array_vels = np.asarray(self.r1_joints_vels)[np.newaxis].T
            r1_array_poses = np.asarray(self.r1_joints_poses)[np.newaxis].T
            r2_array_vels = np.asarray(self.r2_joints_vels)[np.newaxis].T
            r2_array_poses = np.asarray(self.r2_joints_poses)[np.newaxis].T
            obj_array_vel = np.asarray(self.obj_vel)[np.newaxis].T
            obj_array_pose = np.asarray(self.obj_pose)[np.newaxis].T

            # Compute jacobians and ee position from joints_poses
            r1_x_e = np.array([[L1_X*cos(r1_array_poses[1]) - L1_Y*sin(r1_array_poses[1]) + L2*cos(r1_array_poses[1]+r1_array_poses[2]) + L3*cos(r1_array_poses[1]+r1_array_poses[2]+r1_array_poses[3])],\
                            [L1_X*sin(r1_array_poses[1]) + L1_Y*cos(r1_array_poses[1]) + L2*sin(r1_array_poses[1]+r1_array_poses[2]) + L3*sin(r1_array_poses[1]+r1_array_poses[2]+r1_array_poses[3])],\
                            [r1_array_poses[1] + r1_array_poses[2] + r1_array_poses[3]]])
            # Compute ee velocities from joints_vels
            r1_J_e = np.matrix([[ 0.047766999999999996961985715415722*cos(r1_array_poses[1]) - 0.14203*sin(r1_array_poses[1] + r1_array_poses[2]) - 0.16036*sin(r1_array_poses[1] + r1_array_poses[2] + r1_array_poses[3]) - 0.14192399999999999460342792190204*sin(r1_array_poses[1]),\
                             - 0.16036*sin(r1_array_poses[1] + r1_array_poses[2] + r1_array_poses[3]) - 0.14203*sin(r1_array_poses[1] + r1_array_poses[2]), -0.16036*sin(r1_array_poses[1] + r1_array_poses[2] + r1_array_poses[3])],\
                            [ 0.16036*cos(r1_array_poses[1] + r1_array_poses[2] + r1_array_poses[3]) + 0.14203*cos(r1_array_poses[1] + r1_array_poses[2]) + 0.14192399999999999460342792190204*cos(r1_array_poses[1]) + 0.047766999999999996961985715415722*sin(r1_array_poses[1]),\
                              0.16036*cos(r1_array_poses[1] + r1_array_poses[2] + r1_array_poses[3]) + 0.14203*cos(r1_array_poses[1] + r1_array_poses[2]),  0.16036*cos(r1_array_poses[1] + r1_array_poses[2] + r1_array_poses[3])],\
                            [1.0,1.0,1.0]])
            r1_v_e = np.dot(r1_J_e, r1_array_vels[1:4])

            r2_x_e = np.array([[L1_X*cos(r2_array_poses[1]) - L1_Y*sin(r2_array_poses[1]) + L2*cos(r2_array_poses[1]+r2_array_poses[2]) + L3*cos(r2_array_poses[1]+r2_array_poses[2]+r2_array_poses[3])],\
                            [L1_X*sin(r2_array_poses[1]) + L1_Y*cos(r2_array_poses[1]) + L2*sin(r2_array_poses[1]+r2_array_poses[2]) + L3*sin(r2_array_poses[1]+r2_array_poses[2]+r2_array_poses[3])],\
                            [r2_array_poses[1] + r2_array_poses[2] + r2_array_poses[3]]])
            # Compute ee velocities from joints_vels
            r2_J_e = np.matrix([[ 0.16036*sin(r2_array_poses[1,0] + r2_array_poses[2,0] + r2_array_poses[3,0]) + 0.14203*sin(r2_array_poses[1,0] + r2_array_poses[2,0]) - 0.047766999999999996961985715415722*cos(r2_array_poses[1,0]) + 0.14192399999999999460342792190204*sin(r2_array_poses[1,0]), 0.16036*sin(r2_array_poses[1,0] + r2_array_poses[2,0] + r2_array_poses[3,0]) + 0.14203*sin(r2_array_poses[1,0] + r2_array_poses[2,0]), 0.16036*sin(r2_array_poses[1,0] + r2_array_poses[2,0] + r2_array_poses[3,0])],\
                                [ 0.16036*cos(r2_array_poses[1,0] + r2_array_poses[2,0] + r2_array_poses[3,0]) + 0.14203*cos(r2_array_poses[1,0] + r2_array_poses[2,0]) + 0.14192399999999999460342792190204*cos(r2_array_poses[1,0]) + 0.047766999999999996961985715415722*sin(r2_array_poses[1,0]), 0.16036*cos(r2_array_poses[1,0] + r2_array_poses[2,0] + r2_array_poses[3,0]) + 0.14203*cos(r2_array_poses[1,0] + r2_array_poses[2,0]), 0.16036*cos(r2_array_poses[1,0] + r2_array_poses[2,0] + r2_array_poses[3,0])],\
                                [-1.0,-1.0,-1.0]])

            r2_v_e = np.dot(r2_J_e, r2_array_vels[1:4])
            r2_x_e = np.array([[0.77 - r2_x_e[0,0]],[r2_x_e[1,0]],[-r2_x_e[2,0]]])

            control_torque_r1 = np.dot(r1_J_e.T, u_r1)
            control_torque_r2 = np.dot(r2_J_e.T, u_r2)
            print("Torques: ")
            print(control_torque_r1)
            print(control_torque_r2)
            #Create ROS message
            self.torques1.data = [0.0, control_torque_r1[0,0], control_torque_r1[1,0], control_torque_r1[2,0], 0.0, self.r1_close_gripper]
            self.torques2.data = [0.0, control_torque_r2[0,0], control_torque_r2[1,0], control_torque_r2[2,0], 0.0, self.r2_close_gripper]
            self.r1_torque_pub.publish(self.torques1)
            self.r2_torque_pub.publish(self.torques2)



if __name__ == '__main__':
    #Iitialize the node
    rospy.init_node('windowx_controller')
    #Create windowx controller object
    wc = WindowxController()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS WindowX controller node"
