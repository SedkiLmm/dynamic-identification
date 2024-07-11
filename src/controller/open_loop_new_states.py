import numpy as np
from controller import initial_constants

def open_loop_new_states(t, states, U):
    # Get the necessary constants
    constants = initial_constants()

    g = constants['g']
    m = constants['m']
    Iz = constants['Iz']
    Cf = constants['Cf']
    Cr = constants['Cr']
    lf = constants['lf']
    lr = constants['lr']
    Ts = constants['Ts']
    mju = constants['mju']

    x_dot = states[0]
    y_dot = states[1]
    psi = states[2]
    psi_dot = states[3]
    X = states[4]
    Y = states[5]

    # Inputs
    delta = U[0]
    a = U[1]

    Fyf = Cf * (delta - y_dot / x_dot - lf * psi_dot / x_dot)
    Fyr = Cr * (-y_dot / x_dot + lr * psi_dot / x_dot)

    # The nonlinear equation describing the dynamics of the drone
    dx = np.zeros(6)
    dx[0] = a + (-Fyf * np.sin(delta) - mju * m * g) / m + psi_dot * y_dot
    dx[1] = (Fyf * np.cos(delta) + Fyr) / m - psi_dot * x_dot
    dx[2] = psi_dot
    dx[3] = (Fyf * lf * np.cos(delta) - Fyr * lr) / Iz
    dx[4] = x_dot * np.cos(psi) - y_dot * np.sin(psi)
    dx[5] = x_dot * np.sin(psi) + y_dot * np.cos(psi)

    return dx
