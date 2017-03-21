#!/usr/bin/env python

"""
Start ROS node to pubblish torques for manuvering windowx arm using a Robust Quaternion-based Cooperative
controller without Force/Torque Information on its adaptive version.
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
from windowx_arm import *
#widowx dynamics and kinematics class
from widowx_compute_dynamics import WidowxDynamics


class WindowxController():
    """Class to compute and pubblish joints torques"""
    def __init__(self):
        #Object to ee vectors in ee frame
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

        #Control coefficients matrices
        #e_v weight in input computation
        self.Kv = np.matrix([[3.5, 0, 0],  [0, 0.5, 0], [0, 0, 0.5]])
        #position error weight in v_o^r
        self.K_ref = np.matrix([[50, 0, 0],[0, 50, 0], [0, 0, 80]])
        #derivative of position error weight in dot{v_o^r}
        self.K_ref_dot = np.matrix([[10, 0, 0],  [0, 10, 0], [0, 0, 3]])
        #Internal forces regulation constant vector
        self.fd_int = np.matrix([[0.6, -0.05, -0.45, -0.6, 0.05, 0.45]]).T

        #Dynamical parameters vectors
        self.theta_i_est =np.matrix([[M1*L1_CX**2, M1*L1_CX, M1*L1_CY**2, M1*L1_CY, I1, IR1,\
                                    M2*L2_CX**2, M2*L2_CX, M2*L2_CY**2, M2*L2_CY, M2, MR2, I2, IR2, \
                                    M3*L3_CX**3, M3*L3_CX, M3*L3_CY**3, M3*L3_CY, M3, MR3, I3, IR3]]).T
        self.theta_o_est = np.matrix([[MO, IO]]).T
        self.theta_1 =  0.001*np.random.rand(22,1) #random values between 0 and 0.01
        self.theta_2 = 0.001*np.random.rand(22,1) #random values between 0 and 0.01
        self.theta_o1 = 0.001*np.random.rand(2,1) #random values between 0 and 0.01
        self.theta_o2 = 0.001*np.random.rand(2,1) #random values between 0 and 0.01
        #Dynamical parameters update constants
        self.gamma_1 = 0.0005
        self.gamma_2 = 0.0005
        self.gamma_o = 0.001

        #Init widowx dynamics and kinematics handler
        self.wd = WidowxDynamics()

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

        #Security signal service
        print("\nChecking security-stop service availability ... ...")
        rospy.wait_for_service('/windowx_3links_r1/security_stop')
        print("r1: security-stop ok ...")
        rospy.wait_for_service('/windowx_3links_r2/security_stop')
        print("r2: security-stop ok.")
        #Setup security stop services
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
            #Rotation matrices
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
            #Agent to object jacobians
            J_1o = np.matrix([[1,0,-p_o1[1,0]],[0,1,p_o1[0,0]],[0,0,1]])
            J_2o = np.matrix([[1,0,-p_o2[1,0]],[0,1,p_o2[0,0]],[0,0,1]])
            #Object velocity
            self.obj_vel1 = np.dot(J_1o, r1_v_e)
            obj_array_vel1 = self.obj_vel1
            self.obj_vel2 = np.dot(J_2o, r2_v_e)
            obj_array_vel2 = self.obj_vel2
            # Object to angent Jacobians
            J_o1 = np.matrix([[1,0,p_o1[1,0]],[0,1,-p_o1[0,0]],[0,0,1]])
            J_o2 = np.matrix([[1,0,p_o2[1,0]],[0,1,-p_o2[0,0]],[0,0,1]])

            #Gasp matrices
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
            e_p1 = obj_array_pose1[0:2] - obj_target_pose[0:2]
            e_eta1 = eta_o1*eta_od + sin(obj_array_pose1[2,0]/2)*sin(obj_target_pose[2,0]/2)
            S_eps1 = np.matrix([[0, -eps_o1[2,0], eps_o1[1,0]],[eps_o1[2,0], 0, -eps_o1[0,0]],[-eps_o1[1,0], eps_o1[0,0], 0]])
            e_eps1 = eta_o1*eps_od - eta_od*eps_o1 + np.dot(S_eps1, eps_od)

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

            #Errors agent 2
            e_p2 = obj_array_pose2[0:2] - obj_target_pose[0:2]
            e_eta2 = eta_o2*eta_od + sin(obj_array_pose2[2,0]/2)*sin(obj_target_pose[2,0]/2)
            S_eps2 = np.matrix([[0, -eps_o2[2,0], eps_o2[1,0]],[eps_o2[2,0], 0, -eps_o2[0,0]],[-eps_o2[1,0], eps_o2[0,0], 0]])
            e_eps2 = eta_o2*eps_od - eta_od*eps_o2 + np.dot(S_eps2, eps_od)

            e_p_dot2 = obj_array_vel2[0:2] - obj_target_vel[0:2]
            e_omega2 = np.array([[0], [0], [obj_array_vel2[2,0] - obj_target_vel[2,0]]])
            S_e_eps2 = np.matrix([[0, -e_eps2[2,0], e_eps2[1,0]],[e_eps2[2,0], 0, -e_eps2[0,0]],[-e_eps2[1,0], e_eps2[0,0], 0]])
            e_eta_dot2 = 0.5*np.dot(e_eps2.T, e_omega2)
            e_eps_dot2 = -0.5*np.dot((np.identity(3)*e_eta2 + S_e_eps2), e_omega2) - np.dot(S_e_eps2, np.array([[0], [0], [obj_target_vel[2,0]]]))

            #Reference signals agent 2
            e2 = np.array([[e_p2[0,0]], [e_p2[1,0]], [-e_eps2[2,0]]]) # np.array([[e_p2[0,0]], [e_p2[1,0]], [obj_array_pose2[2,0] - obj_target_pose[2,0]]])
            e_dot2 = np.array([[e_p_dot2[0,0]],[e_p_dot2[1,0]],[-e_eps_dot2[2,0]]]) # np.array([[e_p_dot2[0,0]],[e_p_dot2[1,0]],[obj_array_vel2[2,0] - obj_target_vel[2,0]]])
            v_o_r2 = obj_target_vel - np.dot(self.K_ref, e2)
            q2_dot_des = np.sign(np.dot(r2_J_e_inv, v_o_r2))
            v_o_r_dot2 = obj_target_acc - np.dot(self.K_ref_dot, e_dot2)
            e_v2 = obj_array_vel2 - v_o_r2


            #Regression matrices
            Yo1 = self.wd.compute_Yo(v_o_r_dot1)
            Yo2 = self.wd.compute_Yo(v_o_r_dot2)
            Y1 = self.wd.compute_Y1(r1_array_poses, r1_array_vels, v_o_r1, v_o_r_dot1)
            Y2 = self.wd.compute_Y2(r2_array_poses, r2_array_vels, v_o_r2, v_o_r_dot2)

            # Update dynamic parameters estimation
            self.theta_1 = self.theta_1 - self.period*self.gamma_2*np.dot(Y1.T, e_v1)
            self.theta_2 = self.theta_2 - self.period*self.gamma_1*np.dot(Y2.T, e_v2)
            self.theta_o1 = self.theta_o1 - self.period*self.gamma_o*self.c1*np.dot(Yo1.T, e_v1)
            self.theta_o2 = self.theta_o2 - self.period*self.gamma_o*self.c2*np.dot(Yo2.T, e_v2)
            
            #Control inputs
            u_r1 = np.dot(J_1o.T, (np.dot(Y1, self.theta_1) - self.c1*e1 - np.dot(self.Kv, e_v1) + self.c1*np.dot(Yo1, self.theta_o1)))

            u_r2 = np.dot(J_2o.T, (np.dot(Y2, self.theta_2) - self.c2*e2 - np.dot(self.Kv, e_v2) + self.c2*np.dot(Yo2, self.theta_o2)))

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

            #Input torques
            control_torque_r1 = np.dot(r1_J_e.T, u_m1)
            control_torque_r2 = np.dot(r2_J_e.T, u_m2)

            print("\nForces: ")
            print(u_m1)
            print(u_m2)
            print("Torques: ")
            print(control_torque_r1)
            print(control_torque_r2)

            # print(control_torque_r1 - control_torque_r2)
            #Create ROS message
            if norm(control_torque_r2) < 10 and norm(control_torque_r1) < 10:
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
            self.errors.data = [np.linalg.norm(self.theta_1 - self.theta_i_est), np.linalg.norm(self.theta_2 - self.theta_i_est), np.linalg.norm(self.theta_o1 - self.theta_o_est), np.linalg.norm(self.theta_o2 - self.theta_o_est), np.linalg.norm(u_i_new)]
            
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
                                u_i_new[3,0], u_i_new[4,0], u_i_new[5,0], u_i[3,0], u_i[4,0], u_i[5,0],\
                                norm(self.theta_1), norm(self.theta_2), norm(self.theta_o1), norm(self.theta_o2),\
                                norm(self.theta_i_est), norm(self.theta_o_est)\
                                ] + np.matrix.tolist(self.theta_1.T)[0] + np.matrix.tolist(self.theta_2.T)[0] + np.matrix.tolist(self.theta_o1.T)[0] + np.matrix.tolist(self.theta_o2.T)[0]

            self.errors_pub.publish(self.errors)

            #ROS sleep
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
