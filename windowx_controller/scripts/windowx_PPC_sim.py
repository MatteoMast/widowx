#!/usr/bin/env python

"""
Start PPC controller for manuvering windowx arms through the v-rep simulator.
"""

import cv2
import rospy, roslib
from math import sin, cos, pi, sqrt, exp, log, fabs
from windowx_msgs.msg import TargetConfiguration
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np
from numpy.linalg import inv, det, norm
from windowx_arm import *

class WindowxController():
    """Class to compute and pubblish joints torques"""
    def __init__(self):
        #Load-share coefficients
        self.c1 = 0.5
        self.c2 = 0.5

        #Object pose in EEs frames
        self.p1o_in_e1 = np.array([[-0.0755],[0],[0]])
        self.p2o_in_e2 = np.array([[-0.0755],[0],[0]])

        #Control parameters
        self.gs = 0.5
        self.gv = 50.0

        #Performance functions paramenters

        #position
        self.ro_s_0_x = 0.1;
        self.ro_s_0_y = 0.1;
        self.ro_s_0_theta = 0.5;

        self.ro_s_inf_x = 0.03;
        self.ro_s_inf_y = 0.03;
        self.ro_s_inf_theta = 0.25;

        self.l_s_x = 0.1;
        self.l_s_y = 0.1;
        self.l_s_theta = 0.1;

        #Velocity
        self.ro_v_0_x = 13.0;
        self.ro_v_0_y = 13.0;
        self.ro_v_0_theta = 15.0;

        self.ro_v_inf_x = 7.0;
        self.ro_v_inf_y = 7.0;
        self.ro_v_inf_theta = 10.0;

        self.l_v_x = 0.1;
        self.l_v_y = 0.1;
        self.l_v_theta = 0.1;

        #Initialize performance functions
        self.ro_s = np.matrix([[self.ro_s_0_x,0,0],[0,self.ro_s_0_y,0],[0,0, self.ro_s_0_theta]])
        self.ro_v = np.matrix([[self.ro_v_0_x,0,0],[0,self.ro_v_0_y,0],[0,0, self.ro_v_0_theta]])

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
        #Signal check publisher
        self.errors_pub = rospy.Publisher('/errors', Float32MultiArray, queue_size=1)

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

        #Initialize torque message
        self.torques1 = Float32MultiArray()
        self.torques2 = Float32MultiArray()
        self.torques_layout = MultiArrayDimension('control_torques', 6, 0)
        self.torques1.layout.dim = [self.torques_layout]
        self.torques1.layout.data_offset = 0
        self.torques2.layout.dim = [self.torques_layout]
        self.torques2.layout.data_offset = 0

        #Initialize signal check message
        self.errors = Float32MultiArray()
        self.errors_layout = MultiArrayDimension('errors', 6, 0)
        self.errors.layout.dim = [self.errors_layout]
        self.errors.layout.data_offset = 0

        #Initialize timers
        self.start = rospy.get_rostime()
        self.actual_time = rospy.get_rostime()
        #Check variabe to start timer
        self.first_iteration = True
        #Setup check and delay variables for tragectory tracking
        self.first_traj_msg = True
        self.performance_start_step = rospy.Duration.from_sec(4.0)

        #Needed to wait for vrep to update all joints variables
        self.pose_call1 = False
        self.vel_call1 = False
        self.pose_call2 = False
        self.vel_call2 = False
        self.obj_pose_call = False
        self.obj_vel_call = False

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
        #Restart performance functions to respect errors bounds
        if self.first_traj_msg:
                self.start = rospy.get_rostime() - self.performance_start_step
                self.first_traj_msg = False

        self.target_pose = np.asarray(msg.pos)[np.newaxis].T
        self.target_vel = np.asarray(msg.vel)[np.newaxis].T
        self.target_acc = np.asarray(msg.acc)[np.newaxis].T

    #PPC CONTROLLER
    def compute_torques(self,caller):
        """
        Compute and pubblish torques values for 2nd 3rd and 4th joints
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

            #Compute obj position and vel from ee positions and vel
            r1_p_ee = np.array([[r1_x_e[0,0]],[r1_x_e[1,0]],[0]])
            r2_p_ee = np.array([[r2_x_e[0,0]],[r2_x_e[1,0]],[0]])
            Re1  = np.matrix([[cos(r1_x_e[2,0]), -sin(r1_x_e[2,0]), 0], [sin(r1_x_e[2,0]), cos(r1_x_e[2,0]), 0], [0,0,1]])
            Re2_y = np.matrix([[-1, 0, 0],[0,1,0],[0,0,-1]])
            Re2_z  = np.matrix([[cos(r2_x_e[2,0]), -sin(r2_x_e[2,0]), 0], [sin(r2_x_e[2,0]), cos(r2_x_e[2,0]), 0], [0,0,1]])
            Re2 = np.dot(Re2_z, Re2_y)
            self.obj_pose1 = r1_p_ee - np.dot(Re1, self.p1o_in_e1)
            self.obj_pose1[2,0] = r1_x_e[2,0]
            self.obj_pose2 = r2_p_ee - np.dot(Re2, self.p2o_in_e2)
            self.obj_pose2[2,0] = (r2_x_e[2,0])
            p_o1 = self.obj_pose1[0:2] - r1_x_e[0:2]
            p_o2 = self.obj_pose2[0:2] - r2_x_e[0:2]
            # print("Po1:")
            # print(p_o1)
            # print("Po2:")
            # print(p_o2)
            J_1o = np.matrix([[1,0,-p_o1[1,0]],[0,1,p_o1[0,0]],[0,0,1]])
            self.obj_vel1 = np.dot(J_1o, r1_v_e)
            J_2o = np.matrix([[1,0,-p_o2[1,0]],[0,1,p_o2[0,0]],[0,0,1]])
            self.obj_vel2 = np.dot(J_2o, r2_v_e) #[(r1_v_e[0,0] + r2_v_e[0,0])/2, (r1_v_e[1,0] + r2_v_e[1,0])/2, r1_v_e[2,0]]

            #Object-EE jacobians
            J_o1_inv = np.matrix([[1,0,-p_o1[1,0]],[0,1,p_o1[0,0]],[0,0,1]])
            J_o2_inv = np.matrix([[1,0,-p_o2[1,0]],[0,1,p_o2[0,0]],[0,0,1]])

            #Update performance functions
            #if first iteration reset the timer
            if self.first_iteration:
                self.start = rospy.get_rostime()
                self.first_iteration = False
            #Compute elapsed time
            self.actual_time = rospy.get_rostime() - self.start
            #ro s
            self.ro_s[0,0] = (self.ro_s_0_x - self.ro_s_inf_x) * exp(-self.l_s_x * (self.actual_time.to_sec())) + self.ro_s_inf_x
            self.ro_s[1,1] = (self.ro_s_0_y - self.ro_s_inf_y) * exp(-self.l_s_y * (self.actual_time.to_sec())) + self.ro_s_inf_y
            self.ro_s[2,2] = (self.ro_s_0_theta - self.ro_s_inf_theta) * exp(-self.l_s_theta * (self.actual_time.to_sec())) + self.ro_s_inf_theta
            #ro v
            self.ro_v[0,0] = (self.ro_v_0_x - self.ro_v_inf_x) * exp(-self.l_v_x * (self.actual_time.to_sec())) + self.ro_v_inf_x
            self.ro_v[1,1] = (self.ro_v_0_y - self.ro_v_inf_y) * exp(-self.l_v_y * (self.actual_time.to_sec())) + self.ro_v_inf_y
            self.ro_v[2,2] = (self.ro_v_0_theta - self.ro_v_inf_theta) * exp(-self.l_v_theta * (self.actual_time.to_sec())) + self.ro_v_inf_theta

            #Compute errors and derived signals
            #position errors
            e_s = self.obj_pose1 - self.target_pose
            csi_s = np.dot(inv(self.ro_s), e_s)
            csi_s[0,0] = np.sign(csi_s[0,0]) * min(0.99, fabs(csi_s[0,0]))
            csi_s[1,0] = np.sign(csi_s[1,0]) * min(0.99, fabs(csi_s[1,0]))
            csi_s[2,0] = np.sign(csi_s[2,0]) * min(0.99, fabs(csi_s[2,0]))
            eps_s = np.array([[log((1 + csi_s[0,0])/(1 - csi_s[0,0])), log((1 + csi_s[1,0])/(1 - csi_s[1,0])), log((1 + csi_s[2,0])/(1 - csi_s[2,0]))]]).T
            r_s = np.matrix([[2/(1 - csi_s[0,0]**2),0,0],[0,2/(1 - csi_s[1,0]**2),0],[0,0, 2/(1 - csi_s[2,0]**2)]])

            #Compute reference velocity
            v_o_des = - self.gs * np.dot(np.dot(inv(self.ro_s), r_s), eps_s)

            #Velocity errors
            e_v = self.obj_vel1 - v_o_des
            csi_v = np.dot(inv(self.ro_v), e_v)
            csi_v[0,0] = np.sign(csi_v[0,0]) * min(0.9, fabs(csi_v[0,0]))
            csi_v[1,0] = np.sign(csi_v[1,0]) * min(0.9, fabs(csi_v[1,0]))
            csi_v[2,0] = np.sign(csi_v[2,0]) * min(0.9, fabs(csi_v[2,0]))
            eps_v = np.array([[log((1 + csi_v[0,0])/(1 - csi_v[0,0])), log((1 + csi_v[1,0])/(1 - csi_v[1,0])), log((1 + csi_v[2,0])/(1 - csi_v[2,0]))]]).T
            r_v = np.matrix([[2/(1 - csi_v[0,0]**2),0,0],[0,2/(1 - csi_v[1,0]**2),0],[0,0, 2/(1 - csi_v[2,0]**2)]])

            if fabs(max(csi_s)) >0.899999 or fabs(max(csi_v))>0.89999 :
                print("\n csi_s_v:")
                print(csi_s)
                print(csi_v)


            #Compute inputs
            tmp = np.dot(J_o1_inv, inv(self.ro_v))
            tmp = np.dot(tmp, r_v)
            tmp = np.dot(tmp, eps_v)
            u_r1 = - self.c1 * self.gv * tmp

            tmp = np.dot(J_o2_inv, inv(self.ro_v))
            tmp = np.dot(tmp, r_v)
            tmp = np.dot(tmp, eps_v)
            u_r2 = - self.c2 * self.gv * tmp

            control_torque_r1 = np.dot(r1_J_e.T, u_r1)
            control_torque_r2 = np.dot(r2_J_e.T, u_r2)
            print("\n Torques: ")
            print(control_torque_r1)
            print(control_torque_r2)
            #Create ROS message
            self.torques1.data = [0.0, control_torque_r1[0,0], control_torque_r1[1,0], control_torque_r1[2,0], 0.0, self.r1_close_gripper]
            self.torques2.data = [0.0, control_torque_r2[0,0], control_torque_r2[1,0], control_torque_r2[2,0], 0.0, self.r2_close_gripper]
            self.r1_torque_pub.publish(self.torques1)
            self.r2_torque_pub.publish(self.torques2)
            #self.errors.data = [self.obj_pose1[0,0], self.obj_pose1[1,0], self.obj_pose1[2,0], self.target_pose[0,0], self.target_pose[1,0], self.target_pose[2,0]]
            self.errors.data = [r1_array_vels[1,0], r1_array_vels[2,0], r1_array_vels[3,0]]
            self.errors_pub.publish(self.errors)



if __name__ == '__main__':
    #Iitialize the node
    rospy.init_node('windowx_controller')
    #Create windowx controller object
    wc = WindowxController()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS WindowX controller node"
