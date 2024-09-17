import numpy as np
from scipy import linalg, optimize, signal
from scipy.linalg import svd

def normalize_dcm(dcm):
    # Calculate the inverse of the DCM
    dcm_inv = np.linalg.inv(dcm)

    # Calculate the transpose of the inverse
    dcm_inv_T = dcm_inv.T

    # Calculate the normalized DCM
    dcm_normalized = 0.5 * (dcm + dcm_inv_T)

    return dcm_normalized

def update_point_world(state):
    point_world += np.array(state.x, state.y, state.z)

def unit(vector):
    return vector / np.linalg.norm(vector)

def create_extrinsic_matrix(R=np.eye(3), t = np.array([0,0,0])):
    t = t.reshape(3,1)
    E  = np.hstack([R, t.reshape(-1, 1)])
    return E

def plot_coordinate_system(ax,origin=np.array([0,0,0]),R=np.eye(3),length=1):
    x_end = origin + length * R[:, 0]
    y_end = origin + length * R[:, 1]
    z_end = origin + length * R[:, 2]

    # Plot the axes
    ax.quiver(origin[0], origin[1], origin[2], x_end[0] - origin[0], x_end[1] - origin[1], x_end[2] - origin[2], color='r')
    ax.quiver(origin[0], origin[1], origin[2], y_end[0] - origin[0], y_end[1] - origin[1], y_end[2] - origin[2], color='g')
    ax.quiver(origin[0], origin[1], origin[2], z_end[0] - origin[0], z_end[1] - origin[1], z_end[2] - origin[2], color='b')

def triangulate(P_list, x_list):
    A = []
    
    for P, x in zip(P_list, x_list):
        x, y = x
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    
    A = np.array(A)
    
    # Perform SVD
    _, _, Vt = svd(A)
    X = Vt[-1, :]
    
    # Normalize to get homogeneous coordinates
    X = X / X[-1]
    return X[:3]  # Return in non-homogeneous coordinates
    

  