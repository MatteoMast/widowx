#!/usr/bin/env python

"""
Start ROS node to pubblish torques for manuvering windowx arm using a Robust Quaternion-based Cooperative
controller without Force/Torque Information on its non adaptive version.
"""

#Ros handlers services and messages
import rospy, roslib
from windowx_msgs.msg import TargetConfiguration
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from windowx_driver.srv import *
#Math imports
from math import sin, cos, atan2, pi, sqrt
from numpy.linalg import inv, det, norm, pinv
import numpy as np
#Arm parameters
#from windowx_arm import *
#widowx dynamics and kinematics class
from widowx_compute_dynamics import WidowxDynamics


class WindowxController():
    """Class to compute and pubblish joints torques"""
    def __init__(self):
        #Object parameters
        self.m_obj = 0.062 #0.2 #Kg
        self.go = np.array([[0], [self.m_obj*9.81], [0]])
        i_obj = 0.00169*self.m_obj
        self.Io = np.matrix([[i_obj, 0, 0],[0, i_obj, 0],[0,0, i_obj]])
        #Object to ee vectors
        self.p1o_in_e1 = np.array([[-0.045],[0],[0]])
        self.p2o_in_e2 = np.array([[-0.045],[0],[0]])
        #Load share coefficients
        self.c1 = 0.3
        self.c2 = 0.7
        #Robots offsets
        self.pose1 = False
        self.pose2 = False
        self.vel1 = False
        self.vel2 = False
        self.first_iter = False
        self.x_off = 0.603
        self.ees_y_off = 0
        self.omega_off1 = 0
        self.omega_off2 = 0

        #Init widowx dynamics and kinematics handler
        self.wd = WidowxDynamics()

        #Control coefficients matrices
        #e_v weight in input computation
        self.Kv = np.matrix([[3.5, 0, 0],  [0, 0.5, 0], [0, 0, 0.5]])
        #position error weight in v_o^r
        self.K_ref = np.matrix([[50, 0, 0],[0, 50, 0], [0, 0, 80]])
        #derivative of position error weight in dot{v_o^r}
        self.K_ref_dot = np.matrix([[10, 0, 0],  [0, 10, 0], [0, 0, 3]])
        #Internal forces regulation
        self.fd_int = np.matrix([[0.6, -0.05, -0.45, -0.6, 0.05, 0.45]]).T


        
        #ROS SETUP
        #initialize pose, velocity listeners and torques publisher
        #Robot1
        self.r1_pose_sub = rospy.Subscriber('/windowx_3links_r1/joints_poses', Float32MultiArray, self._r1_pose_callback, queue_size=1)
        self.r1_vel_sub = rospy.Subscriber('/windowx_3links_r1/joints_vels', Float32MultiArray, self._r1_vel_callback, queue_size=1)
        self.r1_torque_pub = rospy.Publisher('/windowx_3links_r1/torques', Float32MultiArray, queue_size=1)
        #Robot2
        self.r2_pose_sub = rospy.Subscriber('/windowx_3links_r2/joints_poses', Float32MultiArray, self._r2_pose_callback, queue_size=1)
        self.r2_vel_sub = rospy.Subscriber('/windowx_3links_r2/joints_vels', Float32MultiArray, self._r2_vel_callback, queue_size=1)
        self.r2_torque_pub = rospy.Publisher('/windowx_3links_r2/torques', Float32MultiArray, queue_size=1)
        #Trajectory listener
        self.target_sub = rospy.Subscriber('/object/target_conf', TargetConfiguration, self._target_callback, queue_size=1)
        self.errors_pub = rospy.Publisher('/errors', Float32MultiArray, queue_size=1)
        #Publishing rate
        rate = 120
        self.period = 1.0/rate
        self.pub_rate = rospy.Rate(rate)

        #Initialize torque message
        self.torques1 = Float32MultiArray()
        self.torques2 = Float32MultiArray()
        self.torques_layout = MultiArrayDimension('control_torques', 6, 0)
        self.torques1.layout.dim = [self.torques_layout]
        self.torques1.layout.data_offset = 0
        self.torques2.layout.dim = [self.torques_layout]
        self.torques2.layout.data_offset = 0
        #Initiaze signals check message
        self.errors = Float32MultiArray()
        self.errors_layout = MultiArrayDimension('errors', 6, 0)
        self.errors.layout.dim = [self.errors_layout]
        self.errors.layout.data_offset = 0

        #Initialize signals
        #Initial object's desired position, vel and acc
        self.target_pose = np.array([[0.301,0.11,0.0]]).T
        self.target_vel = np.array([[0,0.0,0.0]]).T
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
        self.obj_pose1 = [0.0, 0.0, 0.0]
        self.obj_vel1 =  [0.0, 0.0, 0.0]
        self.obj_pose2 = [0.0, 0.0, 0.0]
        self.obj_vel2 =  [0.0, 0.0, 0.0]

        #Servo's frictions
        self.Fs = np.matrix([[0.0843,0,0],[0,0.0843,0],[0,0, 0.0078]])
        self.Fv = np.matrix([[0.0347,0,0],[0,0.0347,0],[0,0, 0.0362]])

        #Identity matrix 6x6
        self.I = np.matrix([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])

        #Security signal services
        print("\nChecking security-stop service availability ... ...")
        rospy.wait_for_service('/windowx_3links_r1/security_stop')
        print("r1: security-stop ok ...")
        rospy.wait_for_service('/windowx_3links_r2/security_stop')
        print("r2: security-stop ok.")
        self.r1_sec_stop = rospy.ServiceProxy('/windowx_3links_r1/security_stop', SecurityStop)
        self.r2_sec_stop = rospy.ServiceProxy('/windowx_3links_r2/security_stop', SecurityStop)

        print("\nWindowX controller node created")
        print("\nWaiting for target position, velocity and acceleration...")
        self.compute_torques()

    #SENSING CALLBACKS
    def _r1_pose_callback(self, msg):
        """
        ROS callback to get the joint poses
        """
        self.r1_joints_poses = msg.data
        if self.first_iter:
            self.pose1 = True

    def _r2_pose_callback(self, msg):
        """
        ROS callback to get the joint poses
        """
        self.r2_joints_poses = msg.data
        if self.first_iter:
            self.pose2 = True

    def _r1_vel_callback(self, msg):
        """
        ROS callback to get the joint velocities
        """
        self.r1_joints_vels = msg.data
        if self.first_iter:
            self.vel1 = True

    def _r2_vel_callback(self, msg):
        """
        ROS callback to get the joint velocities
        """
        self.r2_joints_vels = msg.data
        if self.first_iter:
            self.vel2 = True

    def _target_callback(self, msg):
        """
        ROS callback to get the target configuration
        """
        self.target_pose = np.asarray(msg.pos)[np.newaxis].T
        self.target_vel = np.asarray(msg.vel)[np.newaxis].T
        self.target_acc = np.asarray(msg.acc)[np.newaxis].T

    #CONTROLLER
    def compute_torques(self):
        """
        Compute and pubblish torques values for 2nd 3rd and 4th joints
        """

        while not rospy.is_shutdown():
            #Store joints state and desired position for the object
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

            #Uncomment to setup offsets or do initialization stuff
            # if self.first_iter and self.pose1 and self.pose2 and self.vel1 and self.vel2:
            #     # self.ees_y_off = r1_x_e[1,0] - r2_x_e[1,0]
            #     # self.omega_off1 = r1_x_e[2,0]
            #     # self.omega_off2 = r2_x_e[2,0]
            #     self.first_iter = False

            #Compute obj position and vel from ee positions and vel
            #ee positions
            r1_p_ee = np.array([[r1_x_e[0,0]],[r1_x_e[1,0]],[0]])
            r2_p_ee = np.array([[r2_x_e[0,0]],[r2_x_e[1,0]],[0]])
            #Rotation matrice
            Re1  = np.matrix([[cos(r1_x_e[2,0] - self.omega_off1), -sin(r1_x_e[2,0] - self.omega_off1), 0], [sin(r1_x_e[2,0] - self.omega_off1), cos(r1_x_e[2,0] - self.omega_off1), 0], [0,0,1]])
            Re2_y = np.matrix([[-1, 0, 0],[0,1,0],[0,0,-1]]) #pi rotation about y axis
            Re2_z  = np.matrix([[cos(r2_x_e[2,0] - self.omega_off2), -sin(r2_x_e[2,0] - self.omega_off2), 0], [sin(r2_x_e[2,0] - self.omega_off2), cos(r2_x_e[2,0] - self.omega_off2), 0], [0,0,1]])
            Re2 = np.dot(Re2_z, Re2_y)
            #Object poses wrt the 2 agents
            self.obj_pose1 = r1_p_ee - np.dot(Re1, self.p1o_in_e1) 
            self.obj_pose1[2,0] = r1_x_e[2,0] - self.omega_off1
            obj_array_pose1 = self.obj_pose1
            self.obj_pose2 = r2_p_ee - np.dot(Re2, self.p2o_in_e2)
            self.obj_pose2[2,0] = (r2_x_e[2,0] - self.omega_off2)
            obj_array_pose2 = self.obj_pose2
            p_o1 = obj_array_pose1[0:2] - r1_x_e[0:2]
            p_o2 = obj_array_pose2[0:2] - r2_x_e[0:2]
            p_1o = -p_o1
            p_2o = -p_o2
            #Agent to objectjacobians
            J_1o = np.matrix([[1,0,-p_o1[1,0]],[0,1,p_o1[0,0]],[0,0,1]])
            J_2o = np.matrix([[1,0,-p_o2[1,0]],[0,1,p_o2[0,0]],[0,0,1]])
            #Object velocity
            self.obj_vel1 = np.dot(J_1o, r1_v_e)
            obj_array_vel1 = self.obj_vel1            
            self.obj_vel2 = np.dot(J_2o, r2_v_e) 
            obj_array_vel2 = self.obj_vel2
            #Object-EE jacobians and its derivative
            p_o1_dot = obj_array_vel1[0:2] - r1_v_e[0:2]
            p_o2_dot = obj_array_vel2[0:2] - r2_v_e[0:2]
            J_o1 = np.matrix([[1,0,p_o1[1,0]],[0,1,-p_o1[0,0]],[0,0,1]])
            J_o2 = np.matrix([[1,0,p_o2[1,0]],[0,1,-p_o2[0,0]],[0,0,1]])
            J_o1_dot = np.matrix([[1,0,p_o1_dot[1,0]],[0,1,-p_o1_dot[0,0]],[0,0,1]])
            J_o2_dot = np.matrix([[1,0,p_o2_dot[1,0]],[0,1,-p_o2_dot[0,0]],[0,0,1]])

            #GRASP MATRIX
            #Real grasp matrix
            G = np.matrix([[1,0,-p_1o[1,0]],[0,1,p_1o[0,0]],[0,0,1],[1,0,-p_2o[1,0]],[0,1,p_2o[0,0]],[0,0,1]])
            G_star = np.matrix([[1,0,p_1o[1,0],1,0,p_2o[1,0]],[0,1,-p_1o[0,0],0,1,-p_2o[0,0]],[0,0,1,0,0,1]]).T
            #Grasp matrices computed offline by the 2 agents
            G1 = np.matrix([[1,0,-p_1o[1,0]],[0,1,p_1o[0,0]],[0,0,1],[1,0,p_1o[1,0]],[0,1,-p_1o[0,0]],[0,0,1]])
            G2 = np.matrix([[1,0,p_2o[1,0]],[0,1,-p_2o[0,0]],[0,0,1],[1,0,-p_2o[1,0]],[0,1,p_2o[0,0]],[0,0,1]])
            G1_star = np.matrix([[1,0,p_1o[1,0],1,0,-p_1o[1,0]],[0,1,-p_1o[0,0],0,1,p_1o[0,0]],[0,0,1,0,0,1]]).T
            G2_star = np.matrix([[1,0,-p_2o[1,0],1,0,p_2o[1,0]],[0,1,p_2o[0,0],0,1,-p_2o[0,0]],[0,0,1,0,0,1]]).T
            #Agent to object jiacobians computed from the decentralize perspective of each agent
            J_2o_1 = np.matrix([[1,0,-p_1o[1,0]],[0,1,p_1o[0,0]],[0,0,1]])
            J_1o_2 = np.matrix([[1,0,-p_2o[1,0]],[0,1,p_2o[0,0]],[0,0,1]])

            #Robot 1 and 2 dynamics
            M1 = self.wd.compute_M1(r1_array_poses)
            C1 = self.wd.compute_C1(r1_array_poses, r1_array_vels)
            g1 = self.wd.compute_g1(r1_array_poses)

            M2 = self.wd.compute_M2(r2_array_poses)
            C2 = self.wd.compute_C2(r2_array_poses, r2_array_vels)
            g2 = self.wd.compute_g2(r2_array_poses)

            #Object dynamics 3rd component of Co is always 0 and we have rotations only about 3rd axis
            Ro1 = np.matrix([[cos(self.obj_pose1[2]), -sin(self.obj_pose1[2]), 0], [sin(self.obj_pose1[2]), cos(self.obj_pose1[2]), 0], [0,0,1]])
            Mo1 = self.wd.compute_Mo(Ro1)

            Ro2 = np.matrix([[cos(self.obj_pose2[2]), -sin(self.obj_pose2[2]), 0], [sin(self.obj_pose2[2]), cos(self.obj_pose2[2]), 0], [0,0,1]])
            Mo2 = self.wd.compute_Mo(Ro2)

            #Quaternions
            eta_o1 = cos(obj_array_pose1[2,0]/2)
            eps_o1 = sin(obj_array_pose1[2,0]/2) * np.array([[0],[0],[1]])
            eta_o2 = cos(obj_array_pose2[2,0]/2)
            eps_o2 = sin(obj_array_pose2[2,0]/2) * np.array([[0],[0],[1]])
            Rod = np.matrix([[cos(obj_target_pose[2,0]), -sin(obj_target_pose[2,0]), 0], [sin(obj_target_pose[2,0]), cos(obj_target_pose[2,0]), 0], [0,0,1]])
            ksi_den = 2*sqrt(Rod[0,0]+Rod[1,1]+Rod[2,2]+1)
            ksi_1 = (Rod[0,0]+Rod[1,1]+Rod[2,2]+1)/ksi_den
            ksi_2 = (Rod[2,1] - Rod[1,2])/ksi_den
            ksi_3 = (Rod[0,2] - Rod[2,0])/ksi_den
            ksi_4 = (Rod[1,0] - Rod[0,1])/ksi_den
            eta_od = ksi_1 #cos(obj_target_pose[2,0]/2)
            eps_od = sin(obj_target_pose[2,0]/2) * np.array([[0],[0],[1]])#np.array([[ksi_2],[ksi_3],[ksi_4]]) #sin(obj_target_pose[2,0]/2) * np.array([[0],[0],[1]])
            # print("epsilon")
            # print(eps_o)
            # print("epsilon des:")
            # print(eps_od)
            
            #Errors agent 1
            #Position
            e_p1 = obj_array_pose1[0:2] - obj_target_pose[0:2]
            e_eta1 = eta_o1*eta_od + sin(obj_array_pose1[2,0]/2)*sin(obj_target_pose[2,0]/2)
            S_eps1 = np.matrix([[0, -eps_o1[2,0], eps_o1[1,0]],[eps_o1[2,0], 0, -eps_o1[0,0]],[-eps_o1[1,0], eps_o1[0,0], 0]])
            e_eps1 = eta_o1*eps_od - eta_od*eps_o1 + np.dot(S_eps1, eps_od)
            #Velocity
            e_p_dot1 = obj_array_vel1[0:2] - obj_target_vel[0:2]
            e_omega1 = np.array([[0], [0], [obj_array_vel1[2,0] - obj_target_vel[2,0]]])
            S_e_eps1 = np.matrix([[0, -e_eps1[2,0], e_eps1[1,0]],[e_eps1[2,0], 0, -e_eps1[0,0]],[-e_eps1[1,0], e_eps1[0,0], 0]])
            e_eta_dot1 = 0.5*np.dot(e_eps1.T, e_omega1)
            e_eps_dot1 = -0.5*np.dot((np.identity(3)*e_eta1 + S_e_eps1), e_omega1) - np.dot(S_e_eps1, np.array([[0], [0], [obj_target_vel[2,0]]]))
            
            #Reference signals agent 1
            e1 = np.array([[e_p1[0,0]], [e_p1[1,0]], [-e_eps1[2,0]]])
            e_dot1 = np.array([[e_p_dot1[0,0]],[e_p_dot1[1,0]],[-e_eps_dot1[2,0]]])
            v_o_r1 = obj_target_vel - np.dot(self.K_ref, e1)
            q1_dot_des = np.sign(np.dot(r1_J_e_inv, v_o_r1))
            v_o_r_dot1 = obj_target_acc - np.dot(self.K_ref_dot, e_dot1)
            e_v1 = obj_array_vel1 - v_o_r1

            #Errors agent 1
            #Position
            e_p2 = obj_array_pose2[0:2] - obj_target_pose[0:2]
            e_eta2 = eta_o2*eta_od + sin(obj_array_pose2[2,0]/2)*sin(obj_target_pose[2,0]/2)
            S_eps2 = np.matrix([[0, -eps_o2[2,0], eps_o2[1,0]],[eps_o2[2,0], 0, -eps_o2[0,0]],[-eps_o2[1,0], eps_o2[0,0], 0]])
            e_eps2 = eta_o2*eps_od - eta_od*eps_o2 + np.dot(S_eps2, eps_od)
            #velocity
            e_p_dot2 = obj_array_vel2[0:2] - obj_target_vel[0:2]
            e_omega2 = np.array([[0], [0], [obj_array_vel2[2,0] - obj_target_vel[2,0]]])
            S_e_eps2 = np.matrix([[0, -e_eps2[2,0], e_eps2[1,0]],[e_eps2[2,0], 0, -e_eps2[0,0]],[-e_eps2[1,0], e_eps2[0,0], 0]])
            e_eta_dot2 = 0.5*np.dot(e_eps2.T, e_omega2)
            e_eps_dot2 = -0.5*np.dot((np.identity(3)*e_eta2 + S_e_eps2), e_omega2) - np.dot(S_e_eps2, np.array([[0], [0], [obj_target_vel[2,0]]]))
            #Reference signals
            e2 = np.array([[e_p2[0,0]], [e_p2[1,0]], [-e_eps2[2,0]]]) 
            e_dot2 = np.array([[e_p_dot2[0,0]],[e_p_dot2[1,0]],[-e_eps_dot2[2,0]]])
            v_o_r2 = obj_target_vel - np.dot(self.K_ref, e2)
            q2_dot_des = np.sign(np.dot(r2_J_e_inv, v_o_r2))
            v_o_r_dot2 = obj_target_acc - np.dot(self.K_ref_dot, e_dot2)
            e_v2 = obj_array_vel2 - v_o_r2

            #Control Inputs computation
            #robot1
            trm1 = np.dot(C1, J_o1) + np.dot(M1, J_o1_dot)
            trm1 = np.dot(trm1, v_o_r1)
            trm2 = np.dot(M1, J_o1)
            trm2 = np.dot(trm2, v_o_r_dot1)
            J_o1_t = J_o1.T
            J_o1_t_inv = np.linalg.inv(J_o1_t)
            errors_trm = np.dot(self.Kv, e_v1) + self.c1*e1
            trm3 = np.dot(J_o1_t_inv, errors_trm)
            ref_term = np.dot(Mo1, v_o_r_dot1) + self.go
            lambda1 = self.c1*np.dot(J_o1_t_inv, ref_term)
            u_r1 = g1 + trm1 + trm2 - trm3 + lambda1
            #robot2
            trm1 = np.dot(C2, J_o2) + np.dot(M2, J_o2_dot)
            trm1 = np.dot(trm1, v_o_r2)
            trm2 = np.dot(M2, J_o2)
            trm2 = np.dot(trm2, v_o_r_dot2)
            J_o2_t = J_o2.T
            J_o2_t_inv = np.linalg.inv(J_o2_t)
            errors_trm = np.dot(self.Kv, e_v2) + self.c2*e2
            trm3 = np.dot(J_o2_t_inv, errors_trm)
            ref_term = np.dot(Mo2, v_o_r_dot2) + self.go
            lambda2 = self.c2*np.dot(J_o2_t_inv, ref_term)
            u_r2 = g2 + trm1 + trm2 - trm3 + lambda2

            #Internal Forces evaluation and control
            u = np.array([[u_r1[0,0]],[u_r1[1,0]],[u_r1[2,0]],[u_r2[0,0]],[u_r2[1,0]],[u_r2[2,0]]])
            u_i = np.dot((self.I-0.5*np.dot(G_star,G.T)), u)

            u_m = u - u_i
            u_m1 = u_r1 - u_m[0:3]
            u_m2 = u_r2 - u_m[3:6]

            f_int1 = np.dot((self.I-0.5*np.dot(G1_star,G1.T)), self.fd_int)
            f_int2 = np.dot((self.I-0.5*np.dot(G2_star,G2.T)), self.fd_int)
            ud_r1 = u_r1 - f_int1[0:3]
            ud_r2 = u_r2 - f_int2[3:6]

            #new internal forces
            u_i_new = np.dot((self.I-0.5*np.dot(G_star,G.T)), u_m)

            control_torque_r1 = np.dot(r1_J_e.T, u_m1)
            control_torque_r2 = np.dot(r2_J_e.T, u_m2)

            #Create ROS message
            if  norm(control_torque_r2) < 10 and norm(control_torque_r1) < 10:
                #Create ROS message
                self.torques1.data = [control_torque_r1[0,0], control_torque_r1[1,0], control_torque_r1[2,0], q1_dot_des[0,0], q1_dot_des[1,0], q1_dot_des[2,0]]
                self.torques2.data = [control_torque_r2[0,0], control_torque_r2[1,0], control_torque_r2[2,0], q2_dot_des[0,0], q2_dot_des[1,0], q2_dot_des[2,0]]
                self.r1_torque_pub.publish(self.torques1)
                self.r2_torque_pub.publish(self.torques2)
            else:
                #There's a problem with the torques
                print("\n Torques: ")
                print(control_torque_r1)
                print(norm(control_torque_r1))
                print(control_torque_r2)
                print(norm(control_torque_r2))
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
            
            #Fill and publish signal check message
            self.errors.data = [e1[0,0], e1[1,0], e1[2,0], e_eta1, e2[0,0], e2[1,0], e2[2,0], e_eta2,\
                                e_dot1[0,0], e_dot1[1,0], e_dot1[2,0], e_dot2[0,0], e_dot2[1,0], e_dot2[2,0],\
                                e_v1[0,0], e_v1[1,0], e_v1[2,0], e_v2[0,0], e_v2[1,0], e_v2[2,0],
                                v_o_r1[0,0], v_o_r1[1,0], v_o_r1[2,0], v_o_r2[0,0], v_o_r2[1,0], v_o_r2[2,0],\
                                v_o_r_dot1[0,0], v_o_r_dot1[1,0], v_o_r_dot1[2,0], v_o_r_dot2[0,0], v_o_r_dot2[1,0], v_o_r_dot2[2,0],\
                                r1_array_vels[1,0], r1_array_vels[2,0], r1_array_vels[3,0], r2_array_vels[1,0], r2_array_vels[2,0], r2_array_vels[3,0],\
                                self.obj_pose1[0,0], self.obj_pose1[1,0], self.obj_pose1[2,0], self.obj_pose2[0,0], self.obj_pose2[1,0], self.obj_pose2[2,0],\
                                self.target_pose[0,0], self.target_pose[1,0], self.target_pose[2,0],\
                                control_torque_r1[0,0], control_torque_r1[1,0], control_torque_r1[2,0],\
                                control_torque_r2[0,0], control_torque_r2[1,0], control_torque_r2[2,0],\
                                u_r1[0,0], u_r1[1,0], u_r1[2,0], u_m[0,0], u_m[1,0], u_m[2,0],\
                                u_i_new[0,0], u_i_new[1,0], u_i_new[2,0], u_i[0,0], u_i[1,0], u_i[2,0],\
                                u_r2[0,0], u_r2[1,0], u_r2[2,0], u_m[3,0], u_m[4,0], u_m[5,0],\
                                u_i_new[3,0], u_i_new[4,0], u_i_new[5,0], u_i[3,0], u_i[4,0], u_i[5,0]\
                                ]

            self.errors_pub.publish(self.errors)
            self.pub_rate.sleep()



if __name__ == '__main__':
    #Iitialize the node
    rospy.init_node('windowx_coop_controller')
    #Create windowx controller object
    wc = WindowxController()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS WindowX controller node"
