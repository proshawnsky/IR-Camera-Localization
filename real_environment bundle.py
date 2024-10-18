import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from real_camera import custom_real_camera
from utils import *
from scipy.spatial.transform import Rotation as R
import csv
from scipy.optimize import least_squares

# Options 
show_3D_plot = True
show_frames = False      
print_calculations = False  
reprojection_error_threshold = .3

# Define Cameras ___________________________________________________________________________________
cam1_intrinsics = np.load('camera1_calibration.npz')
I1 = cam1_intrinsics['I']
dist_coeffs1 = cam1_intrinsics['dist_coeffs']
cam1_extrinsics = np.load('camera1_extrinsics.npz')
R1 = cam1_extrinsics['R'].T
t1 = cam1_extrinsics['t'].reshape(-1)
Inew1 = cam1_extrinsics['I'] # AFTER UNDISTORTION
roi1 = cam1_extrinsics['roi']
camera1 = custom_real_camera(R = R1,
                        t = t1,
                       color='r',show_img=True, image_scale=1, I = I1, Inew = Inew1, pose_depth=12, vidCapID=0,
                       distortion_coefficients = dist_coeffs1, undistort=False, cameraID=1, roi = roi1)

cam2_intrinsics = np.load('camera2_calibration.npz')
I2 = cam2_intrinsics['I']
dist_coeffs2 = cam2_intrinsics['dist_coeffs']
cam2_extrinsics = np.load('camera2_extrinsics.npz')
R2 = cam2_extrinsics['R'].T # marker position in camera frame
t2 = cam2_extrinsics['t'].reshape(-1)
Inew2 = cam2_extrinsics['I'] # AFTER UNDISTORTION
roi2 = cam2_extrinsics['roi']
camera2 = custom_real_camera(R = R2,
                        t = t2,
                       color='b',show_img=True, image_scale=1, I = I2, Inew = Inew2, pose_depth=12, vidCapID=1,
                       distortion_coefficients = dist_coeffs2, undistort=False, cameraID=2, roi = roi2)

all_cameras = [camera1, camera2]
print("Ready")
# Set up the Wqorld 3D Plot ____________________________________________________________________________
if show_3D_plot:
    fig = plt.figure(1,figsize=(20, 12))
    ax = fig.add_subplot(111, projection='3d')  
    plot_coordinate_system(ax,length=6) # plot wworld coordinate system

    ax.grid(True)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust the margins to remove whitespace
    ax.set_box_aspect([1, 1, 1])

    for i, camera_name in enumerate(all_cameras): # for each camera
        camera_name.Rt2Pose(ax) # plot camera pose
        plot_coordinate_system(ax, origin=camera_name.t, R=camera_name.R, length=6) # plot camera coordinate system

# Capture Frames           
frame1, centroids1 = camera1.getFrame()
frame2, centroids2 = camera2.getFrame()

cam1_rays = camera1.pixels2rays()
cam2_rays = camera2.pixels2rays()
points3D, reprojection_errors = triangulate(camera1.pixels2rays(), camera2.pixels2rays())

candidate_points = []
candidate_points_errors = []

for idx, point in enumerate(points3D):
    if reprojection_errors[idx] < reprojection_error_threshold and point[2] > -.1:
        candidate_points.append(point)
        point_3D_plot, = ax.plot(point[0], point[1], point[2], 'o', color='black')
        candidate_points_errors.append(reprojection_errors[idx])
if show_3D_plot:
        ray_plots = []
        if len(cam1_rays) > 0:
            for ray in cam1_rays:
                ray_plot, = ax.plot(ray[:,0], ray[:,1], ray[:,2], color='red',lw = .5)
                ray_plots.append(ray_plot)
        if len(cam2_rays) > 0:
            for ray in cam2_rays:
                ray_plot, = ax.plot(ray[:,0], ray[:,1], ray[:,2], color='blue',lw = .5)
                ray_plots.append(ray_plot)
# Points Found

def get_reprojection_error(params):
    r2 = params[:3]  # Assuming we use Euler angles or axis-angle for optimization
    # t2 = params[3:6]  # Translation vector for camera 2
    camera2.R = R.from_rotvec(r2).as_matrix()
    # camera2.t = t2
    points3D, reprojection_errors = triangulate(camera1.pixels2rays(), camera2.pixels2rays())
    candidate_points = []
    candidate_points_errors = []
    for idx, point in enumerate(points3D):
        if reprojection_errors[idx] < reprojection_error_threshold and point[2] > -.1:
            candidate_points.append(point)
            candidate_points_errors.append(reprojection_errors[idx])
            ax.plot(point[0], point[1], point[2], 'o', color='black')

    return np.array(candidate_points_errors).ravel()

print(candidate_points_errors)

# initial_params = np.hstack([R.from_matrix(camera2.R).as_rotvec(), camera2.t])
initial_params = np.hstack([R.from_matrix(camera2.R).as_rotvec()])

result = least_squares(get_reprojection_error, initial_params,
                       args=(),
                       method='trf')
    
optimized_r2 = result.x[:3]
# optimized_t2 = result.x[3:6]
R2_optimized = R.from_rotvec(optimized_r2).as_matrix()
camera2.R = R2_optimized
print("Optimized R2:", R2_optimized)
# camera2.t = optimized_t2
camera2.Rt2Pose(ax) # plot camera pose
cam2_rays = camera2.pixels2rays()
for ray in cam2_rays:
    ax.plot(ray[:,0], ray[:,1], ray[:,2], color='green',lw = .5)

points3D, reprojection_errors = triangulate(camera1.pixels2rays(), camera2.pixels2rays())
candidate_points = []
candidate_points_errors = []
for idx, point in enumerate(points3D):
    if reprojection_errors[idx] < reprojection_error_threshold and point[2] > -.1:
        candidate_points.append(point)
        candidate_points_errors.append(reprojection_errors[idx])
        ax.plot(point[0], point[1], point[2], 'o', color='black')
print(candidate_points_errors)  

ax.axis('equal')
ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(0, 70))
plt.show()
camera1.cap.release()
camera2.cap.release()
