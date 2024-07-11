import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

src_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),'../src'))
sys.path.append(src_folder)

from controller import initial_constants
from controller import trajectory_generator
from controller import mpc_simplification

from dynamics import Robot
from dynamics import StateSpace
from controller import open_loop_new_states

# Désactiver les avertissements
import warnings
warnings.filterwarnings('ignore')

# Créer un objet pour les fonctions de support
constants = initial_constants()

# Charger les valeurs constantes nécessaires dans le fichier principal
Ts = constants['Ts']
outputs = constants['outputs']
hz = constants['hz']
inputs = constants['inputs']
trajectory = constants['trajectory']

# Créer le tableau de temps
t = np.arange(0, constants['time_length'], Ts)

# Importer les valeurs de génération de trajectoire
Theta_ref, Theta_dot_ref, Theta_dot_dot_ref = trajectory_generator(t) # Theta_ref 7 *629
sim_length = len(t)

# Générer le tableau de signal de référence
refSignals = np.zeros((1, len(Theta_ref) * outputs))

Theta = np.zeros_like(Theta_ref)
Theta_dot = np.zeros_like(Theta_dot_ref)
Theta_dot_dot = np.zeros_like(Theta_dot_dot_ref)

# Charger les états initiaux
Theta[:, 0] = Theta_ref[:, 0]
Theta_dot[:, 0] = Theta_dot_ref[:, 0]
Theta_dot_dot[:, 0] = Theta_dot_dot_ref[:, 0]

# Créer des tableaux d'état
states = np.concatenate((Theta, Theta_dot, Theta_dot_dot))
#statesTotal = np.zeros((len(states), len(t)))
#statesTotal[:, 0] = states

# Accélérations
Theta_dot_dot = np.zeros_like(Theta_dot_dot_ref)

accelerations_total = np.zeros((len(Theta_dot_dot), len(t)))

# Initialiser le contrôleur - boucles de simulation
U1 = np.zeros(7)
U2 = np.zeros(7)
UTotal = np.zeros((len(t), 7))  # 2*7 car nous avons U1 et U2 pour 7 DOF
UTotal[0, 0:7] = U1

du = np.zeros(inputs * hz)

# Début de la boucle
k = 1  # pour la lecture des signaux de référence
for i in range(sim_length - 1):
    #q = np.zeros(7)
    q = Theta[:, i]
    dq_dt = np.zeros(7)
    dq_dt[0:7] = Theta_dot[:, i]
    #d2q_d2t = np.zeros(13)
    #d2q_d2t[0:7] = Theta_dot_dot[:, i]

    robot = Robot()
    Etatsys = StateSpace(robot)
    Ad, Bd, Cd, Dd = Etatsys.computeStateMatrices(q, dq_dt)
    
    # Générer le vecteur d'état actuel et le vecteur de référence
    x_aug_t = np.concatenate((states[:, i], U1, U2))
    k = k + outputs
    if k + outputs * hz - 1 <= len(refSignals):
        r = refSignals[k:k + outputs * hz - 1]
    else:
        r = refSignals[k:len(refSignals)]
        hz = hz - 1
    M = robot.computeMassMatrix(q)
    # Générer les matrices de simplification pour la fonction de coût
    Hdb, Fdbt, Cdb, Adc, G, ht = mpc_simplification(Ad, Bd, Cd, Dd, hz, x_aug_t, du, M)
    ft = np.dot(np.concatenate((x_aug_t, r)), Fdbt)
    
    # Appeler l'optimiseur (quadprog)
    options = {'disp': False, 'linalg': 'dense'}
    du, fval = quadprog(Hdb, ft, G, ht, options=options)
    
    if du is None:
        print('Le solveur n\'a pas pu trouver la solution')
        break
    
    U1 = U1 + du[0:7]
    U2 = U2 + du[7:14]
    
    UTotal[i + 1, 0:7] = U1
    UTotal[i + 1, 7:14] = U2
    
    # Simuler les nouveaux états
    time_interval = Ts / 30
    T = np.arange(Ts * (i - 1), Ts * (i - 1) + Ts, time_interval)
    T, x = ode45(lambda t, x: open_loop_new_states(t, x, np.concatenate((U1, U2))), T, states[:, i])
    
    states = x[-1, :]
    statesTotal[:, i + 1] = states
    
    # Accélérations
    accelerations = (x[-1, :] - x[-2, :]) / time_interval
    accelerations_total[:, i + 1] = accelerations
    
    if i % 500 == 0:
        print('Progress (%)')
        print(i / sim_length * 100)