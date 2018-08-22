#!/usr/bin/env python

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *
import numpy as np

# Create symbols
q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')
d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')

### KUKA KR210 ###
# Create Modified DH parameters
#
s = {alpha0:     0,  a0:     0,  d1:  0.75,
     alpha1: -pi/2,  a1:  0.35,  d2:     0, q2: q2 - pi/2,
     alpha2:     0,  a2:  1.25,  d3:     0,
     alpha3: -pi/2,  a3: -0.054, d4:  1.5,  # -0.0054, 0.96
     alpha4:  pi/2,  a4:     0,  d5:     0,
     alpha5: -pi/2,  a5:     0,  d6:     0,
     alpha6:     0,  a6:     0,  d7:  0.303, q7: 0}

# Define Modified DH Transformation matrix
def TF_Matrix(alpha, a, d, q):
    TF = Matrix([[            cos(q),          -sin(q),           0,             a],
                 [sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                 [sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                 [                  0,               0,           0,             1]])
    return TF

# Create individual transformation matrices
#
T0_1 = TF_Matrix(alpha0, a0, d1, q1).subs(s)
T1_2 = TF_Matrix(alpha1, a1, d2, q2).subs(s)
T2_3 = TF_Matrix(alpha2, a2, d3, q3).subs(s)
T3_4 = TF_Matrix(alpha3, a3, d4, q4).subs(s)
T4_5 = TF_Matrix(alpha4, a4, d5, q5).subs(s)
T5_6 = TF_Matrix(alpha5, a5, d6, q6).subs(s)
T6_G = TF_Matrix(alpha6, a6, d7, q7).subs(s)

T0_2 = (T0_1 * T1_2)
T0_3 = (T0_2 * T2_3)
T0_4 = (T0_3 * T3_4)
T0_5 = (T0_4 * T4_5)
T0_6 = (T0_5 * T5_6)
T0_G = (T0_6 * T6_G)

R_z = Matrix([[    cos(np.pi), -sin(np.pi),  0,  0],
              [    sin(np.pi),  cos(np.pi),  0,  0],
              [         0,           0,  1,  0],
              [         0,           0,  0,  1]])
R_y = Matrix([[ cos(-np.pi/2),  0,  sin(-np.pi/2),  0],
              [             0,  1,              0,  0],
              [-sin(-np.pi/2),  0,  cos(-np.pi/2),  0],
              [             0,  0,              0,  1]])

R_corr = simplify(R_z * R_y)

T_total = (T0_G * R_corr)

# Extract rotation matrices from the transformation matrices
#
yaws = symbols('yaws') # Z
pitchs = symbols('pitchs') # Y
rolls = symbols('rolls') # X

Rotz = Matrix([[    cos(yaws), -sin(yaws),  0,  0],
              [    sin(yaws),  cos(yaws),  0,  0],
              [         0,           0,  1,  0],
              [         0,           0,  0,  1]])
Roty = Matrix([[ cos(pitchs),  0,  sin(pitchs),  0],
              [             0,  1,              0,  0],
              [-sin(pitchs),  0,  cos(pitchs),  0],
              [             0,  0,              0,  1]])
Rotx = Matrix([[1,           0,          0,  0],
              [0,    cos(rolls), -sin(rolls),  0],
              [0,    sin(rolls),  cos(rolls),  0],
              [0,            0,          0,  1]])

Rrpy = (Rotz * Roty)
Rrpy = (Rrpy * Rotx)
Rrpy = (Rrpy * R_corr)


def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:
        pass
    ### Your FK code here
    # FK code was created up above out of the function so that it is calculated
    # only once when the script is first opened rather than everysingle time
    # it is called.
	#
    ###

    # Initialize service response
    joint_trajectory_list = []
    theta4_prev = 0
    for x in xrange(0, len(req.poses)):
        # IK code starts here
        joint_trajectory_point = JointTrajectoryPoint()

	    # Extract end-effector position and orientation from request
	    # px,py,pz = end-effector position
	    # roll, pitch, yaw = end-effector orientation
        px = req.poses[x].position.x
        py = req.poses[x].position.y
        pz = req.poses[x].position.z

        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
            [req.poses[x].orientation.x, req.poses[x].orientation.y,
                req.poses[x].orientation.z, req.poses[x].orientation.w])

        ### Your IK code here
	    # Compensate for rotation discrepancy between DH parameters and Gazebo
        Rrpyn = Rrpy.evalf(subs={rolls: roll, pitchs: pitch, yaws: yaw})
        nx = Rrpyn[0,2]
        ny = Rrpyn[1,2]
        nz = Rrpyn[2,2]

        wx = px - s[d7]*nx
        wy = py - s[d7]*ny
        wz = pz - s[d7]*nz
	    #
	    # Calculate joint angles using Geometric IK method
        theta1 = atan2(wy, wx)
        r = sqrt(wy**2 + wx**2) - s[a1]
        z = wz-s[d1]
        B = sqrt(r**2 + z**2)
        C = s[a2]
        A = sqrt(s[a3]**2 + s[d4]**2)
	    #
        #
        beta = acos( (A**2 + C**2 - B**2)/(2*A*C) )
        theta3 = pi/2 - beta - atan2(-s[a3], s[d4]) #0.0054, a3, d4
	    #
        alpha = acos(  (B**2 + C**2 - A**2)/(2*B*C) )
        phi = atan2(z, r)
        theta2 = pi/2 - phi - alpha
        ###
        # To solve for these, get a numeric value for
        # inv(R0_3) * Rpyn, then use the formula for R3_6 to extract each of
        # the angles.
        T0_3n = T0_3.evalf(subs={q1: theta1, q2: theta2, q3: theta3})
        R3_6 = T0_3n.inv("LU")*Rrpyn
        sintheta5 = sqrt(R3_6[0,2]**2 + R3_6[2,2]**2) # + or - possible
        costheta5 = R3_6[1,2]

        if sintheta5 > 0.01:
            theta4 = atan2(R3_6[2,2], -R3_6[0,2])
            theta4_alternate = atan2(-R3_6[2,2], R3_6[0,2])
            if abs(theta4_alternate - theta4_prev) < abs(theta4 - theta4_prev):
                theta5 = atan2(-sintheta5, costheta5)
                theta6 = atan2(R3_6[1,1], -R3_6[1,0])
            else:
                theta5 = atan2(sintheta5, costheta5)
                theta6 = atan2(-R3_6[1,1], R3_6[1,0])
        else:
            theta4 = theta4_prev
            theta5 = atan2(sintheta5, costheta5)
            theta6 = atan2(-R3_6[0,1], R3_6[0,0]) - theta4_prev

        theta4_prev = theta4
        ##

        # Populate response for the IK request
        # In the next line replace theta1,theta2...,theta6 by your joint angle variables
        joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
        joint_trajectory_list.append(joint_trajectory_point)

    rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
    return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
