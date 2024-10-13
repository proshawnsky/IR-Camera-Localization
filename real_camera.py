
import cv2
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils import unit, create_extrinsic_matrix, normalize_dcm, euler_to_dcm
class custom_real_camera:
    def __init__(self, t = np.array([0,0,0],dtype=np.float32), 
                 R=np.eye(3), 
                 I=np.eye(3), 
                 camera_resolution=np.array([1280,800]), 
                 color='r', 
                 show_img=False,
                 pose_depth = 2*12,
                 image_scale = 1,
                 cameraID = 1,
                 vidCapID = 0,
                 undistort=False,
                 distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                 Inew = np.eye(3), 
                 roi = np.array([0, 0, 0, 0])):
        self.cameraID = cameraID
        self.color = color
        self.R = R # from world to camera 
        self.t = t # from world to camera
        self.E = create_extrinsic_matrix(self.R,self.t)
        self.I = I
        self.undistort = undistort
        self.distortion_coefficients = distortion_coefficients
        self.show_img = show_img
        self.pose_depth = pose_depth
        self.resolutionX = camera_resolution[0]
        self.resolutionY = camera_resolution[1]
        self.Inew = Inew
        self.roi = roi
        # map1, map2 =           cv2.initUndistortRectifyMap(self.I, self.distortion_coefficients, None, self.Inew, (frame.shape[1], frame.shape[0]), cv2.CV_16SC2)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.I, self.distortion_coefficients, None, self.Inew, (self.roi[2], self.roi[3]),  cv2.CV_16SC2)

        self.image_scale = image_scale
        self.E = create_extrinsic_matrix(self.R.T, self.t)
        self.P = self.Inew @ self.E
        self.cap = cv2.VideoCapture(vidCapID, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolutionX)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolutionY)
        self.cap.set(cv2.CAP_PROP_FPS, 120)
        exposure_value = 10  # Adjust this value (-6 is typically low exposure) -8 us good
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Initializing Camera {self.cameraID}:")
        print(f"Actual Resolution: {int(actual_width)} x {int(actual_height)}")
        print(f"Actual Frame Rate: {actual_fps} FPS")

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
        if not hasattr(self, 'bright_points') or len(self.bright_points) == 0:
            return []
        
        d = self.pose_depth
        f_x = self.Inew[0, 0]  # Focal length in x direction
        f_y = self.Inew[1, 1]  # Focal length in y direction
        c_x = self.Inew[0, 2]  # Principal point x-coordinate
        c_y = self.Inew[1, 2]  # Principal point y-coordinate
        # Convert pixel coordinates to normalized image coordinates
        world_points = []
    
    # Loop over each bright point
        for point in self.bright_points:
            u, v = point  # Extract u and v from each bright point
            
            # Convert pixel coordinates to normalized image coordinates
            x_norm = (u - c_x) / f_x
            y_norm = (v - c_y) / f_y
            point_on_base = np.array([x_norm * d, y_norm * d, d])
            
            # Apply rotation and translation to transform to world coordinates
            point_in_world = (self.R @ point_on_base.T) + self.t
            world_points.append(point_in_world)
    
        return np.array(world_points)
        
    
    def Rt2Pose(self, ax, d=1, alpha=.5):
        f_x = self.Inew[0, 0]  # Focal length in x direction (mm)
        f_y = self.Inew[1, 1]  # Focal length in y direction (mm)
        print(self.Inew)
        # Sensor size in pixels (can be adjusted)

        frame_width = self.roi[2]   # Image width in pixels
        frame_height = self.roi[3]  # Image height in pixels

        # Depth to the base of the pyramid
        d = self.pose_depth * 25.4  # Distance to the base (in arbitrary units, e.g., meters)

        # Calculate the base dimensions of the pyramid
        base_width = (frame_width / f_x) * d
        base_height = (frame_height / f_y) * d
        
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

        _, frame = self.cap.read()
        # height, width, _ = frame.shape

        undistorted_frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)   
        undistorted_gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

        # Threshold the image to get the brite areas
        _, undistorted_gray = cv2.threshold(undistorted_gray, 140, 255, cv2.THRESH_BINARY)
        # scaling_factor = 0.5  # 0.5 reduces brightness by 50%, adjust as needed
        # gray = cv2.convertScaleAbs(gray, alpha=scaling_factor, beta=0)
        # Find contours of the thresholded image
        contours, _ = cv2.findContours(undistorted_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate the center coordinates
        
        centroids = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:  # Check to avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append([cX, cY])
                cv2.circle(undistorted_frame, (cX, cY), 5, (0, 255, 0),thickness=4)
        self.bright_points = np.array(centroids)

        # if self.show_img:
        #     center_x = width // 2
        #     center_y = height // 2
        #     cv2.circle(undistorted_frame, (center_x, center_y), 4, (0, 0, 255), -1)
            # cv2.imshow(self.color, undistorted_frame)
        return undistorted_frame