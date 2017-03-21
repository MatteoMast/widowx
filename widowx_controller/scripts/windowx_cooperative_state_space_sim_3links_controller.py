#!/usr/bin/env python

"""
Start ROS node to pubblish torques for manuvering windowx arm in V-REP using a Robust Quaternion-based Cooperative
controller without Force/Torque Information on its non adaptive version.
"""

#Ros handlers services and messages
import rospy, roslib
from windowx_msgs.msg import TargetConfiguration
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
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
        self.m_obj = 0.062 #Kg
        self.Co = np.matrix([[0,0,0],[0,0,0],[0,0,0]])
        self.go = np.array([[0], [self.m_obj*9.81], [0]])
        i_obj = 0.00169*self.m_obj
        self.Io = np.matrix([[i_obj, 0, 0],[0, i_obj, 0],[0,0, i_obj]])
        #Load share coefficients
        self.c1 = 0.5
        self.c2 = 0.5
        #Control coefficients matrices
        #e_v weight in input computation
        self.Kv = np.matrix([[3, 0, 0], [0, 3, 0], [0, 0, 1]])
        #position error and it's derivative weight in v_o^r and dot{v_o^r} respectively
        self.K = np.matrix([[50, 0, 0],[0, 50, 0], [0, 0, 20]])

        #Init widowx dynamics and kinematics handler
        self.wd = WidowxDynamics()

        #ROS SETUP
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

        #Initialize torque message
        self.torques1 = Float32MultiArray()
        self.torques2 = Float32MultiArray()
        self.torques_layout = MultiArrayDimension('control_torques', 6, 0)
        self.torques1.layout.dim = [self.torques_layout]
        self.torques1.layout.data_offset = 0
        self.torques2.layout.dim = [self.torques_layout]
        self.torques2.layout.data_offset = 0

        #Initialize signals
        #Initial object's desired position, vel and acc
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

        #V-REP sincronization handlesrs
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
        ROS callback to get the object poses
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
        ROS callback to get the object velocities
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
        Compute and pubblish torques values for 2nd, 3rd and 4th joints
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

            #Compute ee position from joints_poses
            r1_x_e = self.wd.compute_ee_pos1(r1_array_poses)
            # Compute ee velocities from joints_vels
            r1_J_e = self.wd.compute_jacobian1(r1_array_poses)
            r1_v_e = np.dot(r1_J_e, r1_array_vels[1:4])
            # Compute ee position from joints_poses
            r2_x_e = self.wd.compute_ee_pos2(r2_array_poses, "sim")
            #Compute ee velocities from joints_vels
            r2_J_e = self.wd.compute_jacobian2(r2_array_poses)
            r2_v_e = np.dot(r2_J_e, r2_array_vels[1:4])

            #Robot 1 and 2 dynamics
            M1 = self.wd.compute_M1(r1_array_poses)
            C1 = self.wd.compute_C1(r1_array_poses, r1_array_vels)
            g1 = self.wd.compute_g1(r1_array_poses)

            M2 = self.wd.compute_M2(r2_array_poses)
            C2 = self.wd.compute_C2(r2_array_poses, r2_array_vels)
            g2 = self.wd.compute_g2(r2_array_poses)

            #Object-EE jacobians (2D)
            p_o1 = obj_array_pose[0:2] - r1_x_e[0:2]
            p_o2 = obj_array_pose[0:2] - r2_x_e[0:2]
            p_o1_dot = obj_array_vel[0:2] - r1_v_e[0:2]
            p_o2_dot = obj_array_vel[0:2] - r2_v_e[0:2]
            J_o1 = np.matrix([[1,0,p_o1[1,0]],[0,1,-p_o1[0,0]],[0,0,1]])
            J_o2 = np.matrix([[1,0,p_o2[1,0]],[0,1,-p_o2[0,0]],[0,0,1]])
            J_o1_dot = np.matrix([[1,0,p_o1_dot[1,0]],[0,1,-p_o1_dot[0,0]],[0,0,1]])
            J_o2_dot = np.matrix([[1,0,p_o2_dot[1,0]],[0,1,-p_o2_dot[0,0]],[0,0,1]])
            #Object dynamics
            Ro = np.matrix([[cos(self.obj_pose[2]), -sin(self.obj_pose[2]), 0], [sin(self.obj_pose[2]), cos(self.obj_pose[2]), 0], [0,0,1]])
            Mo = self.wd.compute_Mo(Ro)

            #Quaternions
            eta_o = cos(obj_array_pose[2,0]/2)
            eps_o = sin(obj_array_pose[2,0]/2) * np.array([[0],[0],[1]])
            eta_od = cos(self.target_pose[2,0]/2)
            eps_od = sin(self.target_pose[2,0]/2) * np.array([[0],[0],[1]])
            #Errors
            e_p = obj_array_pose[0:2] - self.target_pose[0:2]
            e_eta = eta_o*eta_od + sin(obj_array_pose[2,0]/2)*sin(self.target_pose[2,0]/2)
            S_eps = np.matrix([[0, -eps_o[2,0], eps_o[1,0]],[eps_o[2,0], 0, -eps_o[0,0]],[-eps_o[1,0], eps_o[0,0], 0]])
            e_eps = eta_o*eps_od - eta_od*eps_o + np.dot(S_eps, eps_od)

            e_p_dot = obj_array_vel[0:2] - self.target_vel[0:2]
            e_omega = np.array([[0], [0], [obj_array_vel[2,0] - self.target_vel[2,0]]])
            S_e_eps = np.matrix([[0, -e_eps[2,0], e_eps[1,0]],[e_eps[2,0], 0, -e_eps[0,0]],[-e_eps[1,0], e_eps[0,0], 0]])
            e_eta_dot = 0.5*np.dot(e_eps.T, e_omega)
            e_eps_dot = -0.5*np.dot((np.identity(3)*e_eta + S_e_eps), e_omega) - np.dot(S_e_eps, np.array([[0], [0], [self.target_vel[2,0]]]))
            
            #Reference signals
            e = np.array([[e_p[0,0]], [e_p[1,0]], [-e_eps[2,0]]])
            e_dot = np.array([[e_p_dot[0,0]],[e_p_dot[1,0]],[-e_eps_dot[2,0]]])
            v_o_r = self.target_vel - np.dot(self.K, e)
            v_o_r_dot = self.target_acc - np.dot(self.K, e_dot)
            e_v = obj_array_vel - v_o_r

            #Control inputs
            #robot1
            trm1 = np.dot(C1, J_o1) + np.dot(M1, J_o1_dot)
            trm1 = np.dot(trm1, v_o_r)
            trm2 = np.dot(M1, J_o1)
            trm2 = np.dot(trm2, v_o_r_dot)
            J_o1_t = J_o1.T
            J_o1_t_inv = np.linalg.inv(J_o1_t)
            errors_trm = np.dot(self.Kv, e_v) + self.c1*e
            trm3 = np.dot(J_o1_t_inv, errors_trm)
            ref_term = np.dot(Mo, v_o_r_dot) + np.dot(self.Co, v_o_r) + self.go
            lambda1 = self.c1*np.dot(J_o1_t_inv, ref_term)
            u_r1 = g1 + trm1 + trm2 - trm3 + lambda1
            #robot2
            trm1 = np.dot(C2, J_o2) + np.dot(M2, J_o2_dot)
            trm1 = np.dot(trm1, v_o_r)
            trm2 = np.dot(M2, J_o2)
            trm2 = np.dot(trm2, v_o_r_dot)
            J_o2_t = J_o2.T
            J_o2_t_inv = np.linalg.inv(J_o2_t)
            errors_trm = np.dot(self.Kv, e_v) + self.c2*e
            trm3 = np.dot(J_o2_t_inv, errors_trm)
            ref_term = np.dot(Mo, v_o_r_dot) + np.dot(self.Co, v_o_r) + self.go
            lambda2 = self.c2*np.dot(J_o2_t_inv, ref_term)
            u_r2 = g2 + trm1 + trm2 - trm3 + lambda2

            #Compute torques from forces
            control_torque_r1 = np.dot(r1_J_e.T, u_r1)
            control_torque_r2 = np.dot(r2_J_e.T, u_r2)
            
            #Limit torques
            control_torque_r1[0,0] = np.sign(control_torque_r1[0,0])*min(3, abs(control_torque_r1[0,0]))
            control_torque_r1[1,0] = np.sign(control_torque_r1[1,0])*min(3, abs(control_torque_r1[1,0]))
            control_torque_r1[2,0] = np.sign(control_torque_r1[2,0])*min(1.25, abs(control_torque_r1[2,0]))

            control_torque_r2[0,0] = np.sign(control_torque_r2[0,0])*min(3, abs(control_torque_r2[0,0]))
            control_torque_r2[1,0] = np.sign(control_torque_r2[1,0])*min(3, abs(control_torque_r2[1,0]))
            control_torque_r2[2,0] = np.sign(control_torque_r2[2,0])*min(1.25, abs(control_torque_r2[2,0]))
            
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
