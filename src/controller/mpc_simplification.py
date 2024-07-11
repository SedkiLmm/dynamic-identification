#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NB : starting from line 118 there is a matrix sizes mistmatch 
#  I dosent chek the error source just adjust in evry function call 
# the matrices bloc to test code general health excution !
#
# wissem, 11-07-2024 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import numpy as np

from dynamics import Robot
from dynamics import StateSpace

from controller  import initial_constants  
from controller import augmented_matrices  

def mpc_simplification(Ad, Bd, Cd, Dd, hz, x_aug_t, du, M):
    A_aug, B_aug, C_aug, D_aug = augmented_matrices(Ad, Bd, Cd, Dd)
    
    constants = initial_constants()
    inputs = constants['inputs']
    Q = constants['Q']
    S = constants['S']
    R = constants['R']
    tau_max = constants['tau_max']
    tau_min = constants['tau_min']
    
    robot = Robot()
    
    ub_global = np.zeros(inputs * hz)
    lb_global = np.zeros(inputs * hz)
    
    for i in range(inputs):
        for j in range(hz):
            ub_global[i + j * inputs] = tau_max[i]
            lb_global[i + j * inputs] = tau_min[i]
    
    ublb_global = np.concatenate((ub_global, lb_global))
    
    I_global = np.eye(inputs * hz)
    I_global_negative = -I_global
    I_mega_global = np.vstack((I_global, I_global_negative))
    
    y_asterisk_max_global = []
    y_asterisk_min_global = []
    
    C_asterisk = np.eye(C_aug.shape[0])
    C_asterisk_size = C_asterisk.shape
    C_asterisk_global = np.zeros((C_asterisk_size[0] * hz, C_asterisk_size[1] * hz))
    
    CQC = C_aug.T @ Q @ C_aug
    CSC = C_aug.T @ S @ C_aug
    QC = Q @ C_aug
    SC = S @ C_aug
    
    Qdb = np.zeros((CQC.shape[0] * hz, CQC.shape[1] * hz))
    Tdb = np.zeros((QC.shape[0] * hz, QC.shape[1] * hz))
    Rdb = np.zeros((R.shape[0] * hz, R.shape[1] * hz))
    Cdb = np.zeros((B_aug.shape[0] * hz, B_aug.shape[1] * hz))
    Adc = np.zeros((A_aug.shape[0] * hz, A_aug.shape[1]))
    
    A_product = A_aug
    states_predicted_aug = x_aug_t
    A_aug_size = A_aug.shape
    B_aug_size = B_aug.shape
    A_aug_collection = np.zeros((A_aug_size[0], A_aug_size[1], hz))
    B_aug_collection = np.zeros((B_aug_size[0], B_aug_size[1], hz))
    
    for i in range(hz):
        if i == hz - 1:
            Qdb[i * CSC.shape[0]:(i + 1) * CSC.shape[0], i * CSC.shape[1]:(i + 1) * CSC.shape[1]] = CSC
            Tdb[i * SC.shape[0]:(i + 1) * SC.shape[0], i * SC.shape[1]:(i + 1) * SC.shape[1]] = SC
        else:
            Qdb[i * CQC.shape[0]:(i + 1) * CQC.shape[0], i * CQC.shape[1]:(i + 1) * CQC.shape[1]] = CQC
            Tdb[i * QC.shape[0]:(i + 1) * QC.shape[0], i * QC.shape[1]:(i + 1) * QC.shape[1]] = QC
        
        Rdb[i * R.shape[0]:(i + 1) * R.shape[0], i * R.shape[1]:(i + 1) * R.shape[1]] = R
        
        Adc[i * A_aug.shape[0]:(i + 1) * A_aug.shape[0], :A_aug.shape[1]] = A_product
        A_aug_collection[:, :, i] = A_aug
        B_aug_collection[:, :, i] = B_aug
        
        for j in range(inputs):
            a_max = tau_max[j] / M[j, j]
            a_min = tau_min[j] / M[j, j]
            y_asterisk_max_global.append(a_max)
            y_asterisk_min_global.append(a_min)
        
        C_asterisk_global[i * C_asterisk_size[0]:(i + 1) * C_asterisk_size[0], 
                          i * C_asterisk_size[1]:(i + 1) * C_asterisk_size[1]] = C_asterisk
        
        if i < hz - 1:
            du_i = du[i * inputs:(i + 1) * inputs]
            if states_predicted_aug.shape[0] != A_aug_size[0]:
                states_predicted_aug = states_predicted_aug[:A_aug_size[0]]
            states_predicted_aug = np.dot(A_aug , states_predicted_aug) + np.dot(B_aug , du_i)
            states_predicted = states_predicted_aug[:6]
            
            q = states_predicted_aug[:7]
            dq_dt = states_predicted_aug[7:14]

            Etatsys = StateSpace(robot)
            Ad, Bd, Cd, Dd = Etatsys.computeStateMatrices(q, dq_dt)
            A_aug, B_aug, C_aug, D_aug = augmented_matrices(Ad, Bd, Cd, Dd)
            A_product = A_aug @ A_product
    
    for i in range(hz):
        for j in range(hz):
            if j <= i:
                AB_product = np.eye(A_aug_size[0])
                for ii in range(i, j - 1, -1):
                    if ii > j:
                        AB_product = np.dot(AB_product , A_aug_collection[:, :, ii])
                    else:
                        AB_product = np.dot(AB_product ,B_aug_collection[:, :, ii])
                Cdb[i * B_aug_size[0]:(i + 1) * B_aug_size[0], 
                    j * B_aug_size[1]:(j + 1) * B_aug_size[1]] = AB_product
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # error here : shapes (63,63) and (189,63) not aligned: 63 (dim 1) != 189 (dim 0)
    # i climed the second matrix for testing only the code but should resolve it 
    # in the previous lines of the code :
    # Cdb --> Cdb[:63,:]
    # Adc ---> Adc[:63,:]
    # x_aug_t ---> x_aug_t[:21]
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Cdb_constraints =np.dot( C_asterisk_global,Cdb[:63,:])
    Cdb_constraints_negative = -Cdb_constraints
    Cdb_constraints_global = np.vstack((Cdb_constraints, Cdb_constraints_negative))
    
    Adc_constraints = np.dot(C_asterisk_global , Adc[:63,:])
    Adc_constraints_x0 = np.dot(Adc_constraints , x_aug_t[:21])
    
    y_max_Adc_difference = np.array(y_asterisk_max_global) - np.array(Adc_constraints_x0)
    y_min_Adc_difference = - np.array(y_asterisk_min_global) +  np.array(Adc_constraints_x0)
    y_Adc_difference_global = np.hstack((y_max_Adc_difference, y_min_Adc_difference))
    
    G = np.vstack((I_mega_global, Cdb_constraints_global))
    ht = np.hstack((ublb_global, y_Adc_difference_global))
    
    Hdb = np.dot(np.dot(np.transpose(Cdb) , Qdb),  Cdb) + Rdb
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # check the size carfully there is an error here :
    # the computed matrix pased to hstack do not have the same size
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Fdbt = np.hstack((Adc.T @ Qdb @ Cdb, -(Tdb @ Cdb)[:21,:]))
    
    return Hdb, Fdbt, Cdb, Adc, G, ht
