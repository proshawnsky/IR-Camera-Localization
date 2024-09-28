
import cv2
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils import unit, create_extrinsic_matrix, normalize_dcm, euler_to_dcm
class custom_real_camera:
    def __init__(self, t = np.array([0,0,0],dtype=np.float32), 
                 R=np.eye(3), 
                 I=np.eye(3), 
                 camera_resolution=np.array([640,480]), 
                 color='r', 
                 show_img=False,
                 pose_depth = 4*12):
        self.color = color
        self.R = R # from world to camera 
        self.t = t # from world to camera
        self.E = create_extrinsic_matrix(self.R,self.t)
        self.I = I
        self.P = self.I @ self.E
        self.show_img = show_img
        self.pose_depth = pose_depth
        self.resolutionX = camera_resolution[0]
        self.resolutionY = camera_resolution[1]
        
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolutionX)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolutionY)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        exposure_value = 10  # Adjust this value (-6 is typically low exposure)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

    def set_boresight(self, boresight_target):
        self.boresight_target = boresight_target
        b3 = -1*unit(self.t-boresight_target)          # z points away from target
        b2 = np.array([0, 0, -1],dtype=np.float32)  # y is maximally aligned downward
        b1 = unit(np.cross(b2, b3))                 # x completes the triad
        b2 = unit(np.cross(b3, b1))                 # y is orthogonalized
        self.R = np.column_stack((b1, b2, b3))    # world to camera
        self.E = create_extrinsic_matrix(self.R, self.t)
        self.P = self.I @ self.E

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

    def camera2world(self):
        d = self.pose_depth
        u = self.bright[0]
        v = self.bright[1]
        f_x = self.I[0, 0]  # Focal length in x direction
        f_y = self.I[1, 1]  # Focal length in y direction
        c_x = self.I[0, 2]  # Principal point x-coordinate
        c_y = self.I[1, 2]  # Principal point y-coordinate
        # Convert pixel coordinates to normalized image coordinates
        x_norm = (u - c_x) / f_x
        y_norm = (v - c_y) / f_y
        point_on_base = np.array([x_norm * d, y_norm * d, d])
        point_on_base = (self.R @ point_on_base.T) + self.t
        return point_on_base
        
    
    def Rt2Pose(self, ax, d=1, alpha=.5):
        f_x = self.I[0, 0]  # Focal length in x direction (mm)
        f_y = self.I[1, 1]  # Focal length in y direction (mm)

        # Sensor size in pixels (can be adjusted)
        sensor_width = self.resolutionX   # Image width in pixels
        sensor_height = self.resolutionY  # Image height in pixels

        # Depth to the base of the pyramid
        d = self.pose_depth * 25.4  # Distance to the base (in arbitrary units, e.g., meters)

        # Calculate the base dimensions of the pyramid
        base_width = (sensor_width / f_x) * d
        base_height = (sensor_height / f_y) * d
        
        # Calculate the vertices of the pyramid
        tip = np.array([0, 0, 0])
        v1 = np.array([-base_width / 2, -base_height / 2, d])
        v2 = np.array([base_width / 2, -base_height / 2, d])
        v3 = np.array([base_width / 2, base_height / 2, d])
        v4 = np.array([-base_width / 2, base_height / 2, d])

        pyramid = np.array([tip, v1, v2, v3, v4]) / 25.4
        vertices = (self.R @ pyramid.T).T + self.t # from camera frame to world frame
        faces = [[0, 1, 2, 0], [0, 2, 3, 0], [0, 3, 4, 0], [0, 4, 1, 0]]
        mesh = Poly3DCollection(vertices[faces], alpha=alpha,facecolors=self.color)
        pose = ax.add_collection3d(mesh)
        pose.set_clip_on(True)
        
    def getFrame(self):
        self.bright = np.array([self.resolutionX/2, self.resolutionY/2])
        _, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold the image to get the bright areas
        _, gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        scaling_factor = 0.5  # 0.5 reduces brightness by 50%, adjust as needed
        gray = cv2.convertScaleAbs(gray, alpha=scaling_factor, beta=0)
        # Find contours of the thresholded image
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width, _ = frame.shape
        # Calculate the center coordinates
        center_x = width // 2
        center_y = height // 2

        if self.show_img:
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        if contours:
            # Find the largest contour, assuming it's the bright object
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate the centroid of the largest contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                self.bright = np.array([cX, cY])
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0),thickness=2)

                
        if self.show_img:
            frame = cv2.resize(frame, (0,0), fx=2.5, fy=2.5)
            cv2.imshow('Frame', frame)
        