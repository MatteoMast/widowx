#!/usr/bin/env python

"""
Start ROS node to pubblish target sine configuration.
"""

import rospy, roslib
from math import sin, cos, pi
from windowx_msgs.msg import TargetConfiguration
import numpy as np

if __name__ == '__main__':
    period = 35 #s
    omega = (2*pi)/period
    target_pose = TargetConfiguration()
    acc = [0,0,0]
    vel = [0,0,0]
    pos_0 = [0.301,0.11,0]#[0.385,0.10,0]#
    pos = pos_0
    amplitude_x = 0.05
    amplitude_y = 0.05
    amplitude_theta = pi/30
    #Iitialize the node
    print("Initializing node...")
    rospy.init_node('windowx_trajectory')
    target_pub = rospy.Publisher('/object/target_conf', TargetConfiguration, queue_size=1)
    #Start timer
    rate = rospy.Rate(150)
    start = rospy.get_rostime()
    print("Done. Publishing in: \n      /object/target_conf")
    while not rospy.is_shutdown():
        t = rospy.get_rostime() - start
        #circle with orient
        pos =[ pos_0[0] + amplitude_x*sin(omega*t.to_sec()), (pos_0[1] + amplitude_y) - amplitude_y*cos(omega*t.to_sec()), -amplitude_theta*sin(2*omega*t.to_sec())]
        vel =[amplitude_x*omega*cos(omega*t.to_sec()),amplitude_y*omega*sin(omega*t.to_sec()), -2*(amplitude_theta)*omega*cos(2*omega*t.to_sec())]
        acc =[-amplitude_x*(omega**2)*sin(omega*t.to_sec()),amplitude_y*(omega**2)*cos(omega*t.to_sec()), 4*(amplitude_theta)*(omega**2)*sin(2*omega*t.to_sec())]
        #Circle no orient
        # pos =[ pos_0[0] + amplitude_x*sin(omega*t.to_sec()), (pos_0[1] + amplitude_y) - amplitude_y*cos(omega*t.to_sec()), 0]
        # vel =[amplitude_x*omega*cos(omega*t.to_sec()),amplitude_y*omega*sin(omega*t.to_sec()), 0]
        # acc =[-amplitude_x*(omega**2)*sin(omega*t.to_sec()),amplitude_y*(omega**2)*cos(omega*t.to_sec()), 0]
        # #x
        # pos =[ 0.3 + amplitude_x*sin(omega*t.to_sec()), 0.12, 0]
        # vel =[amplitude_x*omega*cos(omega*t.to_sec()),0,0]
        # acc =[-amplitude_x*(omega**2)*sin(omega*t.to_sec()),0,0]
        #x and theta
        # pos =[ 0.3 + amplitude_x*sin(omega*t.to_sec()), 0.12, -amplitude_theta*sin(omega*t.to_sec())]
        # vel =[amplitude_x*omega*cos(omega*t.to_sec()),0,-(amplitude_theta)*omega*cos(omega*t.to_sec())]
        # acc =[-amplitude_x*(omega**2)*sin(omega*t.to_sec()),0,(amplitude_theta)*(omega**2)*sin(omega*t.to_sec())]
        #y
        # pos =[ 0.3 , 0.12 + amplitude_y*sin(omega*t.to_sec()), 0]
        # vel =[0,amplitude_y*omega*cos(omega*t.to_sec()),0]
        # acc =[0,-amplitude_y*(omega**2)*sin(omega*t.to_sec()),0]
        #theta
        # pos =[0.3, 0.12, -amplitude_theta*sin(2*omega*t.to_sec())]
        # vel =[0,0, -2*(amplitude_theta)*omega*cos(2*omega*t.to_sec())]
        # acc =[0,0, 4*(amplitude_theta)*(omega**2)*sin(2*omega*t.to_sec())]

        target_pose.pos = pos
        target_pose.vel = vel
        target_pose.acc = acc
        target_pub.publish(target_pose)
        rate.sleep()




