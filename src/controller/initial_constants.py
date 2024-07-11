import numpy as np

def initial_constants():
    Ts = 0.001  # s

    # Joints to block
    skip_joint = np.array([0, 0, 0, 0, 0, 0, 0])

    # pos max in deg
    pos_max = np.array([180, 128.9, 180, 147.8, 180, 120.3, 180])

    # end position in deg
    pos_f = np.array([30, 90, 90, 90, 90, 90, 90])

    # start position in deg
    pos_0 = np.array([-60, 90, 90, 90, 90, 90, 90])
    start_pos = np.array([-90, 0, 0, 0, 0, 0, 0])

    # vel max in deg.s^-1
    vel_factor = 0.9
    vel_max = np.array([79.64, 79.64, 79.64, 79.64, 69.91, 69.91, 69.91]) * vel_factor

    # acc max in deg.s^-2
    acc_factor = 0.9
    acc_max = np.array([297.94, 297.94, 297.94, 297.94, 572.95, 572.95, 572.95]) * acc_factor

    ampl_max = np.array([75, 45, 75, 70, 60, 60, 60]) * 0.9

    Q = np.eye(7)  # Par exemple, une matrice identité de taille 7
    S = np.eye(7)  # Par exemple, une matrice identité de taille 7
    R = np.eye(7)  # Par exemple, une matrice identité de taille 7

    outputs = 21
    inputs = 7
    controlled_states = 3
    hz = 10  # horizon period

    innerDyn_length = 4  # Number of inner control loop iterations

    # Choose your trajectory (1,2,3,4,5)
    trajectory = 1

    if trajectory == 1:
        time_length = np.pi * 0.2
    else:
        time_length = 1.94150271059894

    # Limites des actions de contrôle (exemple)
    d_u_max = np.array([1.0, 1.2, 0.9, 1.1, 1.0, 1.3, 0.8])  # Exemple de couples max en Nm pour chaque articulation
    d_u_min = -d_u_max  # Les limites minimales sont les opposées des maximales

    # Limites des états (exemple)
    state_max = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])  # Exemple de limites maximales des états
    state_min = -state_max  # Les limites minimales sont les opposées des maximales

    tau_max = np.array([52, 52, 52, 52, 17, 17, 17])
    tau_min = np.array([0, 0, 0, 0, 0, 0, 0])

    constants = {
        'skip_joint': skip_joint,
        'pos_max': pos_max,
        'pos_f': pos_f,
        'pos_0': pos_0,
        'vel_max': vel_max,
        'acc_max': acc_max,
        'Ts': Ts,
        'Q': Q,
        'S': S,
        'R': R,
        'controlled_states': controlled_states,
        'hz': hz,
        'innerDyn_length': innerDyn_length,
        'trajectory': trajectory,
        'ampl_max': ampl_max,
        'start_pos': start_pos,
        'time_length': time_length,
        'outputs': outputs,
        'inputs': inputs,
        'd_u_max': d_u_max,
        'd_u_min': d_u_min,
        'state_max': state_max,
        'state_min': state_min,
        'tau_max': tau_max,
        'tau_min': tau_min
    }

    return constants
