
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils import unit, create_extrinsic_matrix, normalize_dcm
class custom_camera:
    def __init__(self, t = np.array([0,0,0],dtype=np.float32), R = np.eye(3), focal_len = 2, camera_resolution = np.array([64,48]), color = 'r'):
        self.color = color
        self.R = R
        self.t = t
        self.focal_len = focal_len
        self.camera_resolution = camera_resolution
        self.E = create_extrinsic_matrix(self.R,self.t).T
        self.I = np.array([[focal_len, 0, 0], [0, focal_len, 0], [0, 0, 1]], dtype=np.float32)

    def set_boresight(self, boresight_target):
        self.boresight_target = boresight_target
        b3 = unit(self.t-boresight_target) # z points away from target
        b2 = np.array([0, 0, -1],dtype=np.float32) # y is maximally aligned downward
        b1 = unit(np.cross(b2, b3)) # x completes the triad
        b2 = unit(np.cross(b3, b1)) # y is orthogonalized
        self.R = np.column_stack((b1, b2, b3))
        # self.R = normalize_dcm(self.R)
        self.E = create_extrinsic_matrix(self.R,self.t)
        
    def world2camera(self, point_world):
        point_camera = self.R.T @ (point_world - self.t).reshape([-1,1])
        self.point_camera = point_camera
        self.point_frame = self.I @ point_camera

    def Rt2Pose(self, ax):
        pyramid = np.array([[0, 0, 0], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]])
        vertices = (self.R @ pyramid.T).T + self.t
        faces = [[0, 1, 2, 0], [0, 2, 3, 0], [0, 3, 4, 0], [0, 4, 1, 0]]
        mesh = Poly3DCollection(vertices[faces], alpha=0.5,facecolors=self.color)
        ax.add_collection3d(mesh)