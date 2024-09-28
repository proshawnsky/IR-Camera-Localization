
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils import unit, create_extrinsic_matrix, normalize_dcm, euler_to_dcm
class custom_camera:
    def __init__(self, t = np.array([0,0,0],dtype=np.float32), R = np.eye(3), focal_len = 1, camera_resolution = np.array([64,48]), color = 'r'):
        self.color = color
        self.R = R # from world to camera 
        self.t = t # from world to camera
        self.focal_len = focal_len
        self.camera_resolution = camera_resolution
        self.E = create_extrinsic_matrix(self.R,self.t)
        self.I = np.array([[focal_len, 0, 0], [0, focal_len, 0], [0, 0, 1]], dtype=np.float32)
        self.I = np.array([[1.13412156e+03, 0.00000000e+00, 6.28383380e+02],
            [0.00000000e+00, 1.13406816e+03, 3.87493870e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # self.P = self.I @ self.E
 
    def set_boresight(self, boresight_target):
        self.boresight_target = boresight_target
        b3 = -1*unit(self.t-boresight_target)          # z points away from target
        b2 = np.array([0, 0, -1],dtype=np.float32)  # y is maximally aligned downward
        b1 = unit(np.cross(b2, b3))                 # x completes the triad
        b2 = unit(np.cross(b3, b1))                 # y is orthogonalized
        self.R = np.column_stack((b1, b2, b3))    # world to camera
        self.E = create_extrinsic_matrix(self.R, self.t)
        self.P = self.I @ self.E

    def add_pointing_error(self, std):
        rpy = np.random.normal(0, std, 3) # Generate random roll, pitch, and yaw angles with specified standard deviation
        rpy_dcm = euler_to_dcm(rpy) # Convert Euler angles to DCM
        
        # Multiply the original DCM with the random RPY DCM
        R_error = np.dot(self.R, rpy_dcm)
        E_error = create_extrinsic_matrix(R_error, self.t)
        self.E_est = E_error
        self.P_est = self.I @ self.E_est

    
    def world2camera(self, point_world):
        point_world_aug = np.append(point_world, 1).reshape(-1) # convert to homogeneous coordinates
        point_frame = self.P @ point_world_aug # (4x4)@(4x1) = (4x1) homogeneous coordinates
        self.point_frame = point_frame[:2]/point_frame[2] # keep only the first two coordinates and normalize
        return point_frame[:2]/point_frame[2]
    
    def world2camera_est(self, point_world):
        point_world_aug = np.append(point_world, 1).reshape(-1) # convert to homogeneous coordinates
        point_frame = self.P_est @ point_world_aug # (4x4)@(4x1) = (4x1) homogeneous coordinates
        self.point_frame = point_frame[:2]/point_frame[2] # keep only the first two coordinates and normalize
        return point_frame[:2]/point_frame[2]

    # def camera2world(self, point_camera):
    #     point_world = self.R.T @ point_camera.reshape([-1,1]) + self.t.reshape([-1,1]) # (3x3)@(3x1) + (3x1)
    #     return point_world
    
    def Rt2Pose(self, ax, pyramid_height=1, scale=1, alpha=.5):
        pyramid = scale*np.array([[0, 0, 0], [1, 1, pyramid_height], [1, -1, pyramid_height], [-1, -1, pyramid_height], [-1, 1, pyramid_height]]) # camera frame!!
        vertices = (self.R @ pyramid.T).T + self.t # from camera frame to world frame
        faces = [[0, 1, 2, 0], [0, 2, 3, 0], [0, 3, 4, 0], [0, 4, 1, 0]]
        mesh = Poly3DCollection(vertices[faces], alpha=alpha,facecolors=self.color)
        ax.add_collection3d(mesh)

    