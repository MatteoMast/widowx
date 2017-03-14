#!/usr/bin/env python

"""
Start ROS node to pubblish target sine configuration.
"""

import rospy, roslib
from math import sin, cos, pi
from windowx_msgs.msg import TargetConfiguration
import numpy as np

if __name__ == '__main__':
    period = 15 #s
    omega = (2*pi)/period
    target_pose = TargetConfiguration()
    acc = [0,0,0]
    vel = [0,0,0]
    pos = [0.3,0.1,0]
    amplitude = 0.05
    #Iitialize the node
    print("Initializing node...")
    rospy.init_node('windowx_trajectory')
    target_pub = rospy.Publisher('/windowx_3links_r1/target_conf', TargetConfiguration, queue_size=1)
    #Start timer
    rate = rospy.Rate(150)
    start = rospy.get_rostime()
    print("Done. Publishing in: \n      /windowx_3links/target_conf")
    while not rospy.is_shutdown():
        t = rospy.get_rostime() - start
        pos =[ 0.3 + amplitude*sin(omega*t.to_sec()), (0.1-amplitude) + amplitude*cos(omega*t.to_sec()),0]
        vel =[amplitude*omega*cos(omega*t.to_sec()), -amplitude*omega*sin(omega*t.to_sec()),0]
        acc =[-amplitude*(omega**2)*sin(omega*t.to_sec()), -amplitude*(omega**2)*cos(omega*t.to_sec()), 0]
        target_pose.pos = pos
        target_pose.vel = vel
        target_pose.acc = acc
        target_pub.publish(target_pose)
        rate.sleep()




