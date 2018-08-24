from sympy import *
from time import time
from mpmath import radians
import numpy as np
import tf

'''
Format of test case is [ [[EE position],[EE orientation as quaternions]],[WC location],[joint angles]]
You can generate additional test cases by setting up your kuka project and running `$ roslaunch kuka_arm forward_kinematics.launch`
From here you can adjust the joint angles to find thetas, use the gripper to extract positions and orientation (in quaternion xyzw) and lastly use link 5
to find the position of the wrist center. These newly generated test cases can be added to the test_cases dictionary.
'''

test_cases = {1:[[[2.16135,-1.42635,1.55109],
                  [0.708611,0.186356,-0.157931,0.661967]],
                  [1.89451,-1.44302,1.69366],
                  [-0.65,0.45,-0.36,0.95,0.79,0.49]],
              2:[[[-0.56754,0.93663,3.0038],
                  [0.62073, 0.48318,0.38759,0.480629]],
                  [-0.638,0.64198,2.9988],
                  [-0.79,-0.11,-2.33,1.94,1.14,-3.68]],
              3:[[[-1.3863,0.02074,0.90986],
                  [0.01735,-0.2179,0.9025,0.371016]],
                  [-1.1669,-0.17989,0.85137],
                  [-2.99,-0.12,0.94,4.06,1.29,-4.12]],
              4:[[[-0.434071, 0.712935, 3.16884],
                  [0.393403, 0.620724, 0.515411, 0.440781]],
                  [-0.342598, 0.42728, 3.21177],
                  [2.25, -0.42, -0.69, 2.76, -1.29, -1.32]],
              5:[[[-0.231577, -0.685143, 3.30177],
                  [0.533997, 0.752932, 0.381749, -0.0469985]],
                  [-0.102718, -0.917921, 3.15679],
                  [-1.68, 0.73, -2.52, -2.44, 0.90, 2.35]],
              6:[[[2.08033, -0.714807, 1.27216],
                  [0.892906, -0.0692392, -0.233782, 0.37851]],
                  [1.81336, -0.623717, 1.38278],
                  [-0.33, 0.15, 0.22, -0.63, 0, 2.90]]}

q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')
d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')

### KUKA KR210 ###
# DH parameters
s = {alpha0:     0,  a0:     0,  d1:  0.75,
     alpha1: -pi/2,  a1:  0.35,  d2:     0, q2: q2 - pi/2,
     alpha2:     0,  a2:  1.25,  d3:     0,
     alpha3: -pi/2,  a3: -0.054, d4:  1.5,  # -0.0054, 0.96
     alpha4:  pi/2,  a4:     0,  d5:     0,
     alpha5: -pi/2,  a5:     0,  d6:     0,
     alpha6:     0,  a6:     0,  d7:  0.303, q7: 0}

def TF_Matrix(alpha, a, d, q):
    TF = Matrix([[            cos(q),          -sin(q),           0,             a],
                 [sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                 [sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                 [                  0,               0,           0,             1]])
    return TF

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

#T3_6 = simplify(T3_4*T4_5*T5_6)
#print(T3_6)

yaws = symbols('yaws') # Z
pitchs = symbols('pitchs') # Y
rolls = symbols('rolls') # X -

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

R_z = Rotz.subs(yaws, np.pi)
R_y = Roty.subs(pitchs, -np.pi/2)
R_corr = simplify(R_z * R_y)

T_total = (T0_G * R_corr)

Rrpy = Rotz * Roty * Rotx
Rrpy = Rrpy * R_corr

def test_code(test_case):
    ## Set up code
    ## Do not modify!
    x = 0
    class Position:
        def __init__(self,EE_pos):
            self.x = EE_pos[0]
            self.y = EE_pos[1]
            self.z = EE_pos[2]
    class Orientation:
        def __init__(self,EE_ori):
            self.x = EE_ori[0]
            self.y = EE_ori[1]
            self.z = EE_ori[2]
            self.w = EE_ori[3]

    position = Position(test_case[0][0])
    orientation = Orientation(test_case[0][1])

    class Combine:
        def __init__(self,position,orientation):
            self.position = position
            self.orientation = orientation

    comb = Combine(position,orientation)

    class Pose:
        def __init__(self,comb):
            self.poses = [comb]

    req = Pose(comb)
    start_time = time()

    ########################################################################################
    ##

    ## Insert IK code here!

    for x in xrange(0, len(req.poses)):
        # IK code starts here
        #joint_trajectory_point = JointTrajectoryPoint()

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
        # Most of the IK code is up above, out of the function so that it
        # only executes a single time in the code rather than everytime that
        # the function is called.
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
        r = sqrt(wy**2 + wx**2) - s[a1] # 0.35
        z = wz - s[d1] # 0.75
        B = sqrt(r**2 + z**2)
        C = s[a2]
        A = sqrt(s[a3]**2 + s[d4]**2)
	    # Theta3 is determined from the acos, and from subtracting off
        # the small kink in the robotic arm design.
        beta = acos( (A**2 + C**2 - B**2)/(2*A*C) )
        theta3 = np.pi/2 - beta - atan2(-s[a3], s[d4]) #0.0054, a3, d4
	    #
        alpha = acos( (B**2 + C**2 - A**2)/(2*B*C) )
        phi = atan2(z, r)
        theta2 = np.pi/2 - phi - alpha
        ###
        # To solve for these, get a numeric value for
        # inv(R0_3) * Rpyn, then use the formula for R3_6 to extract each of
        # the angles.
        #T0_3  = T0_1[0:3,0:3] * T1_2[0:3,0:3] * T2_3[0:3,0:3]
        T0_3n = T0_3.evalf(subs={q1: theta1, q2: theta2, q3: theta3})
        #R3_6 = T0_3n[0:3,0:3].inv("LU")*Rrpyn[0:3,0:3]
        # Taking the transpose to get the inverse seems to work whereas the
        # above line did not work.
        # This solution was borrowed from another students project.
        R3_6 = T0_3n[0:3,0:3].transpose() * Rrpyn[0:3,0:3]
        print(R3_6)

        sintheta5 = sqrt(R3_6[0,2]**2 + R3_6[2,2]**2)
        costheta5 = R3_6[1,2]

        if sintheta5 > 0.01:
            theta4 = atan2(R3_6[2,2]/sintheta5, -R3_6[0,2]/sintheta5)
            theta5 = atan2(sintheta5, costheta5)
            theta6 = atan2(-R3_6[1,1]/sintheta5, R3_6[1,0]/sintheta5)
        else:
            theta4 = 0
            theta5 = atan2(sintheta5, costheta5)
            theta6 = atan2(-R3_6[0,1], R3_6[0,0])
    ##
    ########################################################################################

    ########################################################################################
    ## For additional debugging add your forward kinematics here. Use your previously calculated thetas
    ## as the input and output the position of your end effector as your_ee = [x,y,z]

    #T_total = T_total.subs(s)
    #print(T_total.subs(s).evalf(subs={q1: theta1, q2: theta2, q3: theta3, q4: theta4, q5: theta5, q6: theta6}))
    Tn = T0_G.evalf(subs={q1: theta1, q2: theta2, q3: theta3, q4: theta4, q5: theta5, q6: theta6})
    #Tn = T0_G.evalf(subs={q1: test_case[2][0], q2: test_case[2][1], q3: test_case[2][2], q4: test_case[2][3], q5: test_case[2][4], q6: test_case[2][5]})
    effx = Tn[0,3]
    effy = Tn[1,3]
    effz = Tn[2,3]
    print(effx, effy, effz)

    ## End your code input for forward kinematics here!
    ########################################################################################

    ## For error analysis please set the following variables of your WC location and EE location in the format of [x,y,z]
    your_wc = [wx,wy,wz] # <--- Load your calculated WC values in this array
    your_ee = [effx,effy,effz] # <--- Load your calculated end effector value from your forward kinematics
    ########################################################################################

    ## Error analysis
    print ("\nTotal run time to calculate joint angles from pose is %04.4f seconds" % (time()-start_time))

    # Find WC error
    if not(sum(your_wc)==3):
        wc_x_e = abs(your_wc[0]-test_case[1][0])
        wc_y_e = abs(your_wc[1]-test_case[1][1])
        wc_z_e = abs(your_wc[2]-test_case[1][2])
        wc_offset = sqrt(wc_x_e**2 + wc_y_e**2 + wc_z_e**2)
        print ("\nWrist error for x position is: %04.8f" % wc_x_e)
        print ("Wrist error for y position is: %04.8f" % wc_y_e)
        print ("Wrist error for z position is: %04.8f" % wc_z_e)
        print ("Overall wrist offset is: %04.8f units" % wc_offset)

    # Find theta errors
    t_1_e = abs(theta1-test_case[2][0])
    t_2_e = abs(theta2-test_case[2][1])
    t_3_e = abs(theta3-test_case[2][2])
    t_4_e = abs(theta4-test_case[2][3])
    t_5_e = abs(theta5-test_case[2][4])
    t_6_e = abs(theta6-test_case[2][5])
    print ("\nTheta 1 error is: %04.8f" % t_1_e)
    print ("Theta 2 error is: %04.8f" % t_2_e)
    print ("Theta 3 error is: %04.8f" % t_3_e)
    print ("Theta 4 error is: %04.8f" % t_4_e)
    print ("Theta 5 error is: %04.8f" % t_5_e)
    print ("Theta 6 error is: %04.8f" % t_6_e)
    print ("\n**These theta errors may not be a correct representation of your code, due to the fact \
           \nthat the arm can have muliple positions. It is best to add your forward kinmeatics to \
           \nconfirm whether your code is working or not**")
    print (" ")

    # Find FK EE error
    if not(sum(your_ee)==3):
        ee_x_e = abs(your_ee[0]-test_case[0][0][0])
        ee_y_e = abs(your_ee[1]-test_case[0][0][1])
        ee_z_e = abs(your_ee[2]-test_case[0][0][2])
        ee_offset = sqrt(ee_x_e**2 + ee_y_e**2 + ee_z_e**2)
        print ("\nEnd effector error for x position is: %04.8f" % ee_x_e)
        print ("End effector error for y position is: %04.8f" % ee_y_e)
        print ("End effector error for z position is: %04.8f" % ee_z_e)
        print ("Overall end effector offset is: %04.8f units \n" % ee_offset)




if __name__ == "__main__":
    # Change test case number for different scenarios
    test_case_number = 6

    test_code(test_cases[test_case_number])
