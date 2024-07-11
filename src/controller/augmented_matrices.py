import numpy as np

def augmented_matrices(Ad, Bd, Cd, Dd):
    A_aug = np.block([
        [Ad, Bd],
        [np.zeros((Bd.shape[1], Ad.shape[1])), np.eye(Bd.shape[1])]
    ])
    B_aug = np.vstack([Bd, np.eye(Bd.shape[1])])
    C_aug = np.hstack([Cd, np.zeros((Cd.shape[0], Bd.shape[1]))])
    D_aug = Dd  # D_aug is not used because it is a zero matrix
    return A_aug, B_aug, C_aug, D_aug
