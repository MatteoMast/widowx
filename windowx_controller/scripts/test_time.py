#!/usr/bin/env python

"""
Start PPC controller for manuvering windowx arms through the ArbotiX board simulator.
"""

import cv2
import rospy, roslib
from math import sin, cos, pi, sqrt, exp, log, fabs
from windowx_msgs.msg import TargetConfiguration
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Header
import numpy as np
from numpy.linalg import inv, det, norm, pinv
from windowx_arm import *
from windowx_driver.srv import *
import time


if __name__ == '__main__':
    #Iitialize the node
    print(np.max(np.array([[1],[2],[3]]),np.array([[4],[5],[6]])))


