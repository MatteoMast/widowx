#!/usr/bin/env python

L1_X = 0.141924
L1_Y = -0.047767
L2 = 0.14203
L3 = 0.15036

#Links' center of mass
L1_CX = L1_X/2
L1_CY = 0
L2_CX = L2/2
L2_CY = 0
L3_CX = L3/2
L3_CY = 0
#Link mass
M1 = 0.074
M2 = 0.061
M3 = 0.176
#Rotor mass
MR1 = 0.142
MR2 = 0.142
MR3 = 0.083
#Links Inertias
I1 = M1*(L1_X**2)/12
I2 = M2*(L2**2)/12
I3 = M3*(L3**2)/12
#Rotors Inertias
IR1 = 0.0154
IR2 = 0.0154
IR3 = 0.0076
#Object
MO = 0.062
IO = 0.00169*MO