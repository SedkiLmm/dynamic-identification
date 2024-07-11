import numpy as np

from controller import  initial_constants 
from sympy import symbols, cos, sin, diff, lambdify

def trajectory_generator(t):
    constants = initial_constants()
    
    Ts = constants['Ts']  # s
    trajectory = constants['trajectory']
    ampl_max = constants['ampl_max']
    start_pos = constants['start_pos']
    pos_max = constants['pos_max']
    vel_max = constants['vel_max']
    acc_max = constants['acc_max']

    t_sym = symbols('t_sym')  # Définir la variable symbolique

    if trajectory == 1:
        freq_sn = np.array([1, 1, 1, 1, 1, 1, 1])

        pos_fun = [
            ampl_max[0] * cos(freq_sn[0] * t_sym) + start_pos[0] - ampl_max[0],
            ampl_max[1] * cos(freq_sn[1] * t_sym) + start_pos[1] - ampl_max[1],
            ampl_max[2] * cos(freq_sn[2] * t_sym) + start_pos[2] - ampl_max[2],
            ampl_max[3] * cos(freq_sn[3] * t_sym) + start_pos[3] - ampl_max[3],
            ampl_max[4] * cos(freq_sn[4] * t_sym) + start_pos[4] - ampl_max[4],
            ampl_max[5] * cos(freq_sn[5] * t_sym) + start_pos[5] - ampl_max[5],
            ampl_max[6] * cos(freq_sn[6] * t_sym) + start_pos[6] - ampl_max[6]
        ]

        pos_fun_lambdified = lambdify(t_sym, pos_fun, 'numpy')
        vel_fun = [diff(p, t_sym) for p in pos_fun]
        vel_fun_lambdified = lambdify(t_sym, vel_fun, 'numpy')
        acc_fun = [diff(v, t_sym) for v in vel_fun]
        acc_fun_lambdified = lambdify(t_sym, acc_fun, 'numpy')

        num_points = len(t)
        pos = np.zeros((7, num_points))
        vel = np.zeros((7, num_points))
        acc = np.zeros((7, num_points))

        for traj_pt in range(num_points):
            pos_joints = pos_fun_lambdified(t[traj_pt])
            vel_joints = vel_fun_lambdified(t[traj_pt])
            acc_joints = acc_fun_lambdified(t[traj_pt])

            if np.any(np.abs(pos_joints) >= pos_max):
                big_pos_idx = np.where(np.abs(pos_joints) >= pos_max)[0]
                for i_pos_big in big_pos_idx:
                    print(f"Joint {i_pos_big + 1} has a position bigger than its limit at t = {t[traj_pt]}")
                raise ValueError("Change the trajectory's parameters! Consider reducing ampl_max or adjusting freq_sn.")

            if np.any(np.abs(vel_joints) >= vel_max):
                big_vel_idx = np.where(np.abs(vel_joints) >= vel_max)[0]
                for i_vel_big in big_vel_idx:
                    print(f"Joint {i_vel_big + 1} has a velocity bigger than its limit at t = {t[traj_pt]}")
                raise ValueError("Change the trajectory's parameters! Consider reducing the frequency or amplitude.")

            if np.any(np.abs(acc_joints) >= acc_max):
                big_acc_idx = np.where(np.abs(acc_joints) >= acc_max)[0]
                for i_acc_big in big_acc_idx:
                    print(f"Joint {i_acc_big + 1} has an acceleration bigger than its limit at t = {t[traj_pt]}")
                raise ValueError("Change the trajectory's parameters! Consider adjusting the trajectory design.")

            acc[:, traj_pt] = acc_joints
            pos[:, traj_pt] = pos_joints
            vel[:, traj_pt] = vel_joints

    else:
        t1 = np.array([0.267302141370746, 0.267302141370746, 0.267302141370746, 0.267302141370746, 0.122017628065276, 0.122017628065276, 0.122017628065276])
        t2 = np.array([1.67420056922819, 1.25565042692115, 1.25565042692115, 1.25565042692115, 1.43041052782148, 1.43041052782148, 1.43041052782148])
        tf = np.array([1.94150271059894, 1.52295256829189, 1.52295256829189, 1.52295256829189, 1.55242815588676, 1.55242815588676, 1.55242815588676])
        b = np.array([-90, 0, 0, 0, 0, 0, 0])
        c = np.array([-99.5795741424448, -9.57957414244479, -9.57957414244479, -9.57957414244479, -8.64300100975044, -8.64300100975044, -8.64300100975044])
        d = np.array([15.708, 0, 0, 0, 0, 0, 0])

        pos_fun = b + c * t_sym + d * t_sym ** 2
        pos_fun_lambdified = lambdify(t_sym, pos_fun, 'numpy')
        vel_fun = diff(pos_fun, t_sym)
        vel_fun_lambdified = lambdify(t_sym, vel_fun, 'numpy')
        acc_fun = diff(vel_fun, t_sym)
        acc_fun_lambdified = lambdify(t_sym, acc_fun, 'numpy')

        num_points = len(t)
        pos = np.zeros((7, num_points))
        vel = np.zeros((7, num_points))
        acc = np.zeros((7, num_points))

        for traj_pt in range(num_points):
            pos[:, traj_pt] = pos_fun_lambdified(t[traj_pt])
            vel[:, traj_pt] = vel_fun_lambdified(t[traj_pt])
            acc[:, traj_pt] = acc_fun_lambdified(t[traj_pt])

    return pos, vel, acc
