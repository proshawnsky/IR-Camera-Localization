import numpy as np

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
    E = np.concatenate((R,t),axis=1)
    E = np.vstack((E,np.array([0,0,0,1])))
    return E

def plot_coordinate_system(ax,origin=np.array([0,0,0]),R=np.eye(3),length=1):

    x_end = origin + length * R[:, 0]
    y_end = origin + length * R[:, 1]
    z_end = origin + length * R[:, 2]

    # Plot the axes
    ax.quiver(origin[0], origin[1], origin[2], x_end[0] - origin[0], x_end[1] - origin[1], x_end[2] - origin[2], color='r')
    ax.quiver(origin[0], origin[1], origin[2], y_end[0] - origin[0], y_end[1] - origin[1], y_end[2] - origin[2], color='g')
    ax.quiver(origin[0], origin[1], origin[2], z_end[0] - origin[0], z_end[1] - origin[1], z_end[2] - origin[2], color='b')