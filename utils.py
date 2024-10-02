import numpy as np
from scipy import linalg, optimize, signal
from scipy.linalg import svd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    E = np.concatenate((R.T, -R.T @ t.reshape(-1,1)), axis=1)
    return E

def plot_coordinate_system(ax,origin=np.array([0,0,0]),R=np.eye(3),length=1):
    x_end = origin + length * R[:, 0]
    y_end = origin + length * R[:, 1]
    z_end = origin + length * R[:, 2]

    # Plot the axes
    ax.quiver(origin[0], origin[1], origin[2], x_end[0] - origin[0], x_end[1] - origin[1], x_end[2] - origin[2], color='r', zorder=10000, linewidth=4)
    ax.quiver(origin[0], origin[1], origin[2], y_end[0] - origin[0], y_end[1] - origin[1], y_end[2] - origin[2], color='g', zorder=10000, linewidth=4)
    ax.quiver(origin[0], origin[1], origin[2], z_end[0] - origin[0], z_end[1] - origin[1], z_end[2] - origin[2], color='b', zorder=10000, linewidth=4)

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
    

def euler_to_dcm(euler_angles):
    # Convert Euler angles (roll, pitch, yaw) to DCM
    r, p, y = euler_angles
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    dcm = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])
    return dcm    

def plot_chessboard(ax, room_center, square_size=2.25, board_size=8):
    """
    Plots a chessboard on the floor (z = 0) centered at room_center.

    Parameters:
    - ax: The matplotlib 3D axis to plot on.
    - room_center: The (x, y) coordinates for the center of the chessboard.
    - square_size: The size of each square on the chessboard.
    - board_size: The number of squares along one edge of the chessboard (default is 8 for an 8x8 chessboard).
    """
    # Calculate the bottom-left corner of the chessboard
    start_x = room_center[0] - (board_size / 2) * square_size
    start_y = room_center[1] - (board_size / 2) * square_size

    # Generate squares for the chessboard
    for i in range(board_size):
        for j in range(board_size):
            x = start_x + i * square_size
            y = start_y + j * square_size
            
            # Define the corners of the square
            square = np.array([[x, y, 0], 
                               [x + square_size, y, 0], 
                               [x + square_size, y + square_size, 0], 
                               [x, y + square_size, 0]])
            
            # Alternate colors between black and white
            color = 'white' if (i + j) % 2 == 0 else 'black'
            
            # Plot the square
            square_collection = Poly3DCollection([square], color=color)
            ax.add_collection3d(square_collection)
    import numpy as np
import cv2
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_aruco_grid(ax, room_center = np.array([0,0,0]), marker_size=10.75):
    image_size = 500

    # Generate the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)

    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, image_size)  # 200x200 pixels marker
    marker_rgba = cv2.cvtColor(marker_img, cv2.COLOR_BGR2RGBA)/255 
    marker_rgba = np.transpose(marker_rgba, (1, 0, 2))  # Swapping the first two dimensions
    marker_rgba = cv2.rotate(marker_rgba, cv2.ROTATE_90_COUNTERCLOCKWISE)
    marker_rgba[:, :, 3] = 1
    
    x = np.linspace(0, marker_size, image_size) - marker_size/2 + room_center[0]
    y = np.linspace(0, marker_size, image_size) - marker_size/2 + room_center[1]
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape) + room_center[2]
    # Plot the surface with the image as a texture
    print(marker_rgba.shape)
    ax.plot_surface(X, Y, Z, facecolors=marker_rgba, cmap='viridis', edgecolor='none', antialiased=True)

            



