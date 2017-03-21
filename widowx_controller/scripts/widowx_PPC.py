#!/usr/bin/env python

"""
Start PPC controller for manuvering widowx arms through the ArbotiX board simulator.
"""

#Ros handlers services and messages
import rospy, roslib
from widowx_msgs.msg import TargetConfiguration
from widowx_driver.srv import *
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
#Math imports
from math import sin, cos, pi, sqrt, exp, log, fabs
import numpy as np
from numpy.linalg import inv, det, norm, pinv
#Arm parameters
#from widowx_arm import *
#widowx dynamics and kinematics class
from widowx_compute_dynamics import WidowxDynamics
#For sleeps
import time

class WidowxController():
    """Class to compute and pubblish joints torques"""
    def __init__(self):
        #Load-share coefficients
        self.c1 = 0.3
        self.c2 = 0.7

        #Object pose in EEs frames
        self.p1o_in_e1 = np.array([[-0.044],[0],[0]])
        self.p2o_in_e2 = np.array([[-0.044],[0],[0]])

        #Robots offsets
        self.first_iter = True
        self.x_off = 0.603 #m
        self.ees_y_off = 0
        self.omega_off1 = 0
        self.omega_off2 = 0

        #Identity matrix
        self.I = np.matrix([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])

        #Control parameters
        self.gs = 0.05
        self.gv = 6.8
        self.C = np.matrix([[self.c1,0,0,0,0,0],[0,self.c1,0,0,0,0],[0,0,self.c1,0,0,0],[0,0,0,self.c2,0,0],[0,0,0,0,self.c2,0],[0,0,0,0,0,self.c2]])

        #Performance functions paramenters
        #position
        self.ro_s_0_x = 0.05;
        self.ro_s_0_y = 0.05;
        self.ro_s_0_theta = 0.4;

        self.ro_s_inf_x = 0.02;
        self.ro_s_inf_y = 0.02;
        self.ro_s_inf_theta = 0.2;

        self.l_s_x = 0.2;
        self.l_s_y = 0.2;
        self.l_s_theta = 0.2;

        #Velocity
        self.ro_v_0_x = 10.0;
        self.ro_v_0_y = 15.0;
        self.ro_v_0_theta = 7.0;

        self.ro_v_inf_x = 5.0;
        self.ro_v_inf_y = 10.0;
        self.ro_v_inf_theta = 3.0;

        self.l_v_x = 0.2;
        self.l_v_y = 0.2;
        self.l_v_theta = 0.2;

        #Init widowx dynamics and kinematics handler
        self.wd = WidowxDynamics()

        #Initialize performance functions matrices
        self.ro_s = np.matrix([[self.ro_s_0_x,0,0],[0,self.ro_s_0_y,0],[0,0, self.ro_s_0_theta]])
        self.ro_v = np.matrix([[self.ro_v_0_x,0,0],[0,self.ro_v_0_y,0],[0,0, self.ro_v_0_theta]])

        #initialize pose, velocity listeners and torques publishers
        #Robot1
        self.r1_pose_sub = rospy.Subscriber('/widowx_3links_r1/joints_poses', Float32MultiArray, self._r1_pose_callback, queue_size=1)
        self.r1_vel_sub = rospy.Subscriber('/widowx_3links_r1/joints_vels', Float32MultiArray, self._r1_vel_callback, queue_size=1)
        self.r1_torque_pub = rospy.Publisher('/widowx_3links_r1/torques', Float32MultiArray, queue_size=1)
        #Robot2
        self.r2_pose_sub = rospy.Subscriber('/widowx_3links_r2/joints_poses', Float32MultiArray, self._r2_pose_callback, queue_size=1)
        self.r2_vel_sub = rospy.Subscriber('/widowx_3links_r2/joints_vels', Float32MultiArray, self._r2_vel_callback, queue_size=1)
        self.r2_torque_pub = rospy.Publisher('/widowx_3links_r2/torques', Float32MultiArray, queue_size=1)
        #Trajectory listener
        self.target_sub = rospy.Subscriber('/object/target_conf', TargetConfiguration, self._target_callback, queue_size=1)
        #Signal check publisher
        self.errors_pub = rospy.Publisher('/control_signals', Float32MultiArray, queue_size=1)
        #Torque pubblish rate
        self.pub_rate = rospy.Rate(120) #max 120, higher values generetes reads errors

        #Security signal service
        print("\nChecking security-stop service availability ... ...")
        #rospy.wait_for_service('/widowx_3links_r1/security_stop')
        print("r1: security-stop ok ...")
        rospy.wait_for_service('/widowx_3links_r2/security_stop')
        rospy.wait_for_service('/widowx_3links_r1/security_stop')
        print("r2: security-stop ok.")
        self.r1_sec_stop = rospy.ServiceProxy('/widowx_3links_r1/security_stop', SecurityStop)
        self.r2_sec_stop = rospy.ServiceProxy('/widowx_3links_r2/security_stop', SecurityStop)

        #Initial pose, all joints will move to the initial target position, and initialization of pose and vels vectors
        #Here the target configuration is x_e = [x,y,orientation] x_e_dot x_e_ddot of the end effector wrt the inertial frame of the robot
        self.target_pose = np.array([[0.301,0.11,0.0]]).T
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

        #Torque compensation
        self.tau_comp1 = np.matrix([[0.4], [0.7], [0.2]])
        self.tau_comp2 = np.matrix([[0.3], [0.6], [0.15]])
        self.tau_old = np.array([[0,0,0]]).T

        #Initialize control_signals message
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
        self.performance_start_step = rospy.Duration.from_sec(2.0)

        time.sleep(1)
        print("\nWidowX controller node created")
        print("\nWaiting for target position, velocity and acceleration...")
        self.compute_torques()

    #SENSING CALLBACKS
    def _r1_pose_callback(self, msg):
        """
        ROS callback to get the joint poses
        """
        self.r1_joints_poses = msg.data

    def _r2_pose_callback(self, msg):
        """
        ROS callback to get the joint poses
        """
        self.r2_joints_poses = msg.data

    def _r1_vel_callback(self, msg):
        """
        ROS callback to get the joint velocities
        """
        self.r1_joints_vels = msg.data
    def _r2_vel_callback(self, msg):
        """
        ROS callback to get the joint velocities
        """
        self.r2_joints_vels = msg.data

    #DESIRED TRAJECTORY CALLBACK
    def _target_callback(self, msg):
        """
        ROS callback to get the target configuration
        """
        #Restart performance functions to respect errors bounds
        if self.first_traj_msg:
                self.start = rospy.get_rostime()
                self.first_traj_msg = False
        self.target_pose = np.asarray(msg.pos)[np.newaxis].T
        self.target_vel = np.asarray(msg.vel)[np.newaxis].T
        self.target_acc = np.asarray(msg.acc)[np.newaxis].T

    #PPC CONTROLLER
    def compute_torques(self):
        """
        Compute and pubblish torques values for 2nd, 3rd and 4th joints
        """

        while not rospy.is_shutdown():

            r1_array_vels = np.asarray(self.r1_joints_vels)[np.newaxis].T
            r1_array_poses = np.asarray(self.r1_joints_poses)[np.newaxis].T
            r2_array_vels = np.asarray(self.r2_joints_vels)[np.newaxis].T
            r2_array_poses = np.asarray(self.r2_joints_poses)[np.newaxis].T
            obj_target_pose = self.target_pose
            obj_target_vel = self.target_vel
            obj_target_acc = self.target_acc

            #Compute ee position from joints_poses
            r1_x_e = self.wd.compute_ee_pos1(r1_array_poses)
            # Compute ee velocities from joints_vels
            r1_J_e = self.wd.compute_jacobian1(r1_array_poses)
            r1_v_e = np.dot(r1_J_e, r1_array_vels[1:4])
            # Compute ee position from joints_poses
            r2_x_e = self.wd.compute_ee_pos2(r2_array_poses, "exp")
            #Compute ee velocities from joints_vels
            r2_J_e = self.wd.compute_jacobian2(r2_array_poses)
            r2_v_e = np.dot(r2_J_e, r2_array_vels[1:4])
            #Invert the Jacobians
            r1_J_e_inv = inv(r1_J_e)
            r2_J_e_inv = inv(r2_J_e)

            #Setup offsets
            if self.first_iter:
                # self.ees_y_off = r1_x_e[1,0] - r2_x_e[1,0]
                #self.omega_off1 = r1_x_e[2,0]
                #self.omega_off2 = r2_x_e[2,0]
                self.first_iter = False

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
            self.obj_pose2[2,0] = r2_x_e[2,0]
            p_1o = r1_x_e[0:2] - self.obj_pose1[0:2]
            p_2o = r2_x_e[0:2] - self.obj_pose2[0:2]
            # print("Po1:")
            # print(p_o1)
            # print("Po2:")
            # print(p_o2)
            J_1o = np.matrix([[1,0,p_1o[1,0]],[0,1,-p_1o[0,0]],[0,0,1]])
            self.obj_vel1 = np.dot(J_1o, r1_v_e)
            J_2o = np.matrix([[1,0,p_2o[1,0]],[0,1,-p_2o[0,0]],[0,0,1]])
            self.obj_vel2 = np.dot(J_2o, r2_v_e) #[(r1_v_e[0,0] + r2_v_e[0,0])/2, (r1_v_e[1,0] + r2_v_e[1,0])/2, r1_v_e[2,0]]
            G = np.matrix([[1,0,-p_1o[1,0]],[0,1,p_1o[0,0]],[0,0,1],[1,0,-p_2o[1,0]],[0,1,p_2o[0,0]],[0,0,1]])
            G_star = np.matrix([[1,0,p_1o[1,0],1,0,p_2o[1,0]],[0,1,-p_1o[0,0],0,1,-p_2o[0,0]],[0,0,1,0,0,1]]).T

            #GRASP MATRIX
            G1 = np.matrix([[1,0,-p_1o[1,0]],[0,1,p_1o[0,0]],[0,0,1],[1,0,p_1o[1,0]],[0,1,-p_1o[0,0]],[0,0,1]])
            G2 = np.matrix([[1,0,p_2o[1,0]],[0,1,-p_2o[0,0]],[0,0,1],[1,0,-p_2o[1,0]],[0,1,p_2o[0,0]],[0,0,1]])
            G1_star = np.matrix([[1,0,p_1o[1,0],1,0,-p_1o[1,0]],[0,1,-p_1o[0,0],0,1,p_1o[0,0]],[0,0,1,0,0,1]]).T
            G2_star = np.matrix([[1,0,-p_2o[1,0],1,0,p_2o[1,0]],[0,1,p_2o[0,0],0,1,-p_2o[0,0]],[0,0,1,0,0,1]]).T
            J_2o_1 = np.matrix([[1,0,-p_1o[1,0]],[0,1,p_1o[0,0]],[0,0,1]])
            J_1o_2 = np.matrix([[1,0,-p_2o[1,0]],[0,1,p_2o[0,0]],[0,0,1]])
            # print("\n obj_vels")
            # print(self.obj_vel2)
            # print(r2_array_vels)
            # print(self.obj_pose2)
            # print(self.obj_pose1)
            # print(r1_x_e)
            # print(r2_x_e)
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
            #r1
            #position errors
            e_s = self.obj_pose1 - self.target_pose
            e_s1 = e_s
            csi_s = np.dot(inv(self.ro_s), e_s)
            csi_s[0,0] = np.sign(csi_s[0,0]) * min(0.9999, fabs(csi_s[0,0]))
            csi_s[1,0] = np.sign(csi_s[1,0]) * min(0.9999, fabs(csi_s[1,0]))
            csi_s[2,0] = np.sign(csi_s[2,0]) * min(0.9999, fabs(csi_s[2,0]))
            eps_s = np.array([[log((1 + csi_s[0,0])/(1 - csi_s[0,0])), log((1 + csi_s[1,0])/(1 - csi_s[1,0])), log((1 + csi_s[2,0])/(1 - csi_s[2,0]))]]).T
            r_s = np.matrix([[2/(1 - csi_s[0,0]**2),0,0],[0,2/(1 - csi_s[1,0]**2),0],[0,0, 2/(1 - csi_s[2,0]**2)]])

            #Compute moving direction for joints from position error
            v_1_des = np.dot(inv(J_1o), -e_s)
            q1_dot_des = np.sign(np.dot(r1_J_e_inv, v_1_des))
            q1_dot_des = np.matrix([[q1_dot_des[0,0],0,0],[0,q1_dot_des[1,0],0],[0,0,q1_dot_des[2,0]]])

            #Compute reference velocity
            tmp = np.dot(inv(self.ro_s), r_s)
            tmp = np.dot(tmp, eps_s)
            v_o_des = - self.gs * tmp

            #Velocity errors
            e_v = self.obj_vel1 - v_o_des
            e_v1 = e_v
            csi_v = np.dot(inv(self.ro_v), e_v)
            csi_v[0,0] = np.sign(csi_v[0,0]) * min(0.99, fabs(csi_v[0,0]))
            csi_v[1,0] = np.sign(csi_v[1,0]) * min(0.99, fabs(csi_v[1,0]))
            csi_v[2,0] = np.sign(csi_v[2,0]) * min(0.99, fabs(csi_v[2,0]))
            eps_v1 = np.array([[log((1 + csi_v[0,0])/(1 - csi_v[0,0])), log((1 + csi_v[1,0])/(1 - csi_v[1,0])), log((1 + csi_v[2,0])/(1 - csi_v[2,0]))]]).T
            r_v1 = np.matrix([[2/(1 - csi_v[0,0]**2),0,0],[0,2/(1 - csi_v[1,0]**2),0],[0,0, 2/(1 - csi_v[2,0]**2)]])

            if max(fabs(min(csi_s)), max(csi_s)) >0.9998 or max(fabs(min(csi_v)), max(csi_v))>0.98 :
                print("\n r1: \ncsi_s:")
                print(csi_s)
                print("csi_v")
                print(csi_v)
                print("e_v")
                print(e_v)
                print("ro_v")
                print(self.ro_v)
                print("e_s")
                print(e_s)
                print("ro_s")
                print(self.ro_s)
                print("Obj_vel")
                print(self.obj_vel1)
                print("referenc vel")
                print(v_o_des)

            #r2
            e_s = self.obj_pose2 - self.target_pose
            e_s2 = e_s
            csi_s = np.dot(inv(self.ro_s), e_s)
            csi_s[0,0] = np.sign(csi_s[0,0]) * min(0.9999, fabs(csi_s[0,0]))
            csi_s[1,0] = np.sign(csi_s[1,0]) * min(0.9999, fabs(csi_s[1,0]))
            csi_s[2,0] = np.sign(csi_s[2,0]) * min(0.9999, fabs(csi_s[2,0]))
            eps_s = np.array([[log((1 + csi_s[0,0])/(1 - csi_s[0,0])), log((1 + csi_s[1,0])/(1 - csi_s[1,0])), log((1 + csi_s[2,0])/(1 - csi_s[2,0]))]]).T
            r_s = np.matrix([[2/(1 - csi_s[0,0]**2),0,0],[0,2/(1 - csi_s[1,0]**2),0],[0,0, 2/(1 - csi_s[2,0]**2)]])

            #Compute moving direction for joints from position error
            v_2_des = np.dot(inv(J_2o), -e_s)
            q2_dot_des = np.sign(np.dot(r2_J_e_inv, v_2_des))
            q2_dot_des = np.matrix([[q2_dot_des[0,0],0,0],[0,q2_dot_des[1,0],0],[0,0,q2_dot_des[2,0]]])

            #Compute reference velocity
            tmp = np.dot(inv(self.ro_s), r_s)
            tmp = np.dot(tmp, eps_s)
            v_o_des = - self.gs * tmp

            #Velocity errors
            e_v = self.obj_vel2 - v_o_des
            e_v2 = e_v
            csi_v = np.dot(inv(self.ro_v), e_v)
            csi_v[0,0] = np.sign(csi_v[0,0]) * min(0.99, fabs(csi_v[0,0]))
            csi_v[1,0] = np.sign(csi_v[1,0]) * min(0.99, fabs(csi_v[1,0]))
            csi_v[2,0] = np.sign(csi_v[2,0]) * min(0.99, fabs(csi_v[2,0]))
            eps_v2 = np.array([[log((1 + csi_v[0,0])/(1 - csi_v[0,0])), log((1 + csi_v[1,0])/(1 - csi_v[1,0])), log((1 + csi_v[2,0])/(1 - csi_v[2,0]))]]).T
            r_v2 = np.matrix([[2/(1 - csi_v[0,0]**2),0,0],[0,2/(1 - csi_v[1,0]**2),0],[0,0, 2/(1 - csi_v[2,0]**2)]])





            if max(fabs(min(csi_s)), max(csi_s)) >0.9998 or max(fabs(min(csi_v)), max(csi_v))>0.98 :
                print("\n r2: \ncsi_s:")
                print(csi_s)
                print("csi_v")
                print(csi_v)
                print("e_v")
                print(e_v)
                print("ro_v")
                print(self.ro_v)
                print("e_s")
                print(e_s)
                print("ro_s")
                print(self.ro_s)
                print("Obj_vel")
                print(self.obj_vel2)
                print("referenc vel")
                print(v_o_des)


            #Compute inputs
            #Object center of mass input
            u_o1 = - self.gv * np.dot(np.dot(inv(self.ro_v), r_v1), eps_v1)
            u_o2 = - self.gv * np.dot(np.dot(inv(self.ro_v), r_v2), eps_v2)

            u_r1 = self.c1 * np.dot(J_1o.T, u_o1)
            u_r2 = self.c2 * np.dot(J_2o.T, u_o2)

            u = np.array([[u_r1[0,0]],[u_r1[1,0]],[u_r1[2,0]],[u_r2[0,0]],[u_r2[1,0]],[u_r2[2,0]]])
            u_i = np.dot((self.I-0.5*np.dot(G_star,G.T)), u)

            u_m = u - u_i

            u_i_new = np.dot((self.I-0.5*np.dot(G_star,G.T)), u_m)

            #Distributed
            u_r1_2 = self.c2 * np.dot(J_1o_2.T, u_o2)
            u_r2_1 = self.c1 * np.dot(J_2o_1.T, u_o1)

            u1 = np.array([[u_r1[0,0]],[u_r1[1,0]],[u_r1[2,0]],[u_r2_1[0,0]],[u_r2_1[1,0]],[u_r2_1[2,0]]])
            u2 = np.array([[u_r1_2[0,0]],[u_r1_2[1,0]],[u_r1_2[2,0]],[u_r2[0,0]],[u_r2[1,0]],[u_r2[2,0]]])
            u_i_1 = np.dot((self.I-0.5*np.dot(G1_star,G1.T)), u1)
            u_i_2 = np.dot((self.I-0.5*np.dot(G2_star,G2.T)), u2)

            print("\ninternal forces distr:")
            print(u_i_1)
            print(u_i_2)

            control_torque_r1 = np.dot(r1_J_e.T, u_m[0:3])
            control_torque_r2 = np.dot(r2_J_e.T, u_m[3:6])
            # control_torque_r1 = control_torque_r1 + np.dot(q1_dot_des, self.tau_comp1)
            # control_torque_r2 = control_torque_r2 + np.dot(q2_dot_des, self.tau_comp2)

            if  norm(control_torque_r2) < 10 and norm(control_torque_r1) < 10:
                #Create ROS message
                self.torques1.data = [control_torque_r1[0,0], control_torque_r1[1,0], control_torque_r1[2,0], q1_dot_des[0,0], q1_dot_des[1,1], q1_dot_des[2,2]]
                self.torques2.data = [control_torque_r2[0,0], control_torque_r2[1,0], control_torque_r2[2,0], q2_dot_des[0,0], q2_dot_des[1,1], q2_dot_des[2,2]]
                self.r1_torque_pub.publish(self.torques1)
                self.r2_torque_pub.publish(self.torques2)
            else:
                #There's a problem with the torques
                print("\n Torques: ")
                print(control_torque_r1)
                print(norm(control_torque_r1))
                print(control_torque_r2)
                print(norm(control_torque_r2))
                print("Inputs r1, r2, obj")
                print(u_r1)
                print(u_r2)
                print(-self.gv*u_o1)
                print(-self.gv*u_o2)
                print("Jacobians")
                print(r1_J_e.T)
                print(r2_J_e.T)
                print("q1,q2")
                print(r1_array_poses)
                print(r2_array_poses)
                rospy.logerr("Torques limit reached, shutting down driver and controller")
                try:
                    self.r1_sec_stop('Torques limit reached')
                except:
                    print("r1 stopped")
                try:
                    self.r2_sec_stop('Torques limit reached')
                except:
                    print("r2 stopped")

                rospy.signal_shutdown("Torques limit reached")


            #self.errors.data = [self.obj_pose1[0,0], self.obj_pose1[1,0], self.obj_pose1[2,0], self.target_pose[0,0], self.target_pose[1,0], self.target_pose[2,0]]
            #self.errors.data = [self.ro_v[0,0], e_v[0,0], self.ro_v[1,1], e_v[1,0], self.ro_v[2,2], e_v[2,0], self.ro_s[0,0], e_s[0,0], self.ro_s[1,1], e_s[1,0], self.ro_s[2,2], e_s[2,0]]
            #self.errors.data = [self.obj_pose1[0,0], self.obj_pose1[1,0], self.obj_pose1[2,0], r1_x_e[0,0], r1_x_e[1,0], r1_x_e[2,0], r2_x_e[0,0], r2_x_e[1,0], r2_x_e[2,0], self.obj_vel1[0,0], self.obj_vel1[1,0], self.obj_vel1[2,0]]
            #self.errors.data = [r1_v_e[0,0], r1_v_e[1,0], r1_v_e[2,0], r2_v_e[0,0], r2_v_e[1,0], r2_v_e[2,0]]
            #self.errors.data = [r1_array_vels[1,0], r1_array_vels[2,0], r1_array_vels[3,0], r2_array_vels[1,0], r2_array_vels[2,0], r2_array_vels[3,0], e_v[0,0], e_v[1,0], e_v[2,0], e_s1[0,0], e_s1[1,0], e_s1[2,0], e_s2[0,0], e_s2[1,0], e_s2[2,0]]
            self.errors.data = [e_s1[0,0], e_s1[1,0], e_s1[2,0], e_v1[0,0], e_v1[1,0], e_v1[2,0],  \
                                e_s2[0,0], e_s2[1,0], e_s2[2,0], e_v2[0,0], e_v2[1,0], e_v2[2,0], \
                                self.ro_s[0,0], self.ro_s[1,1], self.ro_s[2,2], \
                                self.ro_v[0,0], self.ro_v[1,1], self.ro_v[2,2], \
                                r1_array_vels[1,0], r1_array_vels[2,0], r1_array_vels[3,0], r2_array_vels[1,0], r2_array_vels[2,0], r2_array_vels[3,0],\
                                self.obj_pose1[0,0], self.obj_pose1[1,0], self.obj_pose1[2,0], self.obj_pose2[0,0], self.obj_pose2[1,0], self.obj_pose2[2,0],\
                                self.target_pose[0,0], self.target_pose[1,0], self.target_pose[2,0],\
                                control_torque_r1[0,0], control_torque_r1[1,0], control_torque_r1[2,0],\
                                control_torque_r2[0,0], control_torque_r2[1,0], control_torque_r2[2,0],\
                                u_o1[0,0], u_o1[1,0], u_o1[2,0], u_r1[0,0], u_r1[1,0], u_r1[2,0], u_m[0,0], u_m[1,0], u_m[2,0],\
                                u_i_new[0,0], u_i_new[1,0], u_i_new[2,0], u_i[0,0], u_i[1,0], u_i[2,0],\
                                u_o2[0,0], u_o2[1,0], u_o2[2,0], u_r2[0,0], u_r2[1,0], u_r2[2,0], u_m[3,0], u_m[4,0], u_m[5,0],\
                                u_i_new[3,0], u_i_new[4,0], u_i_new[5,0], u_i[3,0], u_i[4,0], u_i[5,0]\
                                ]

            self.errors_pub.publish(self.errors)
            self.pub_rate.sleep()



if __name__ == '__main__':
    #Iitialize the node
    rospy.init_node('widowx_controller')
    #Create widowx controller object
    wc = WidowxController()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS WidowX controller node"
