#!/usr/bin/env python

"""
Start ROS node to pubblish torques for manuvering windowx arm through the v-rep simulator.
"""

import numpy as np
import math
from numpy.linalg import inv

if __name__ == '__main__':
    #Costants
    a1 = 0.14203 #m
    a2 = 0.16036 #m
    l1 = a1/2 #m
    l2 = a2/2 #m
    ml1 = 0.144 #kg
    ml2 = 0.176 #kg
    mm1 = 0
    mm2 = 0
    Im1 = 0
    Im2 = 0
    kr1 = 200
    kr2 = 193
    Il1 = (ml1*(a1**2))/3 #kg m2
    Il2 = (ml2*(a2**2))/3 #kg m2
    g = 9.807#m/s2

    #Compute matrices elements as string
    b11 = str(Il1 + (kr1**2)*Im1 + Im2 + mm2*(a1**2) + ml1*(l1**2) + Il2 + ml2*(a1**2) \
        + ml2*(l2**2)) + " + " + str(ml2*2*a1*l2) + "*cos( theta2 )"

    b12 = str( Il2 + ml2*(l2**2) + kr2*Im2) + " + " + str(ml2*a1*l2) + "*cos( theta2 )"

    b22 = str( Il2 + ml2*(l2**2) + (kr2**2)*Im2)

    h = str(ml2*a1*l2) + "*sin( theta2 )"

    g1 = str((ml1*l1 + mm2*a1 + ml2*a1)*g) + " cos( theta1 )" + " + " + str(ml2*l2*g) + "*cos( theta1 + theta2 )"
    g2 = str(ml2*l2*g) + "*cos( theta1 + theta2 )"

    B = np.matrix([[b11, b12],[b12,b22]])
    C = np.matrix([["-"+h+"*omega2", "-"+h + "(omega1 + omega2)"], [h + "*omega2", str(0)]])
    g = np.array([[g1],[g2]])

    b11_num = Il1 + (kr1**2)*Im1 + Im2 + mm2*(a1**2) + ml1*(l1**2) + Il2 + ml2*(a1**2) \
                    + ml2*(l2**2) + ml2*2*a1*l2
    b12_num = Il2 + ml2*(l2**2) + kr2*Im2 + ml2*a1*l2
    b22_num = Il2 + ml2*(l2**2) + (kr2**2)*Im2

    B_max = inv(np.matrix([[b11_num, b12_num],[b12_num,b22_num]]))

    KP = np.dot(B_max, np.matrix([[6],[2.5]]))


print "B:"
print (B)
print("\n")
print "C:"
print (C)
print("\n")
print "g:"
print (g)
print("\n")
print "KP:"
print (KP)

