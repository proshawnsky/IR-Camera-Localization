import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from real_camera import custom_real_camera
from utils import *
import csv

# Options 
show_3D_plot = True
show_frames = False      
print_calculations = True  
reprojection_error_threshold = 2

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

# Set up the Wqorld 3D Plot ____________________________________________________________________________
if show_3D_plot:
    fig = plt.figure(1,figsize=(20, 12))
    manager = plt.get_current_fig_manager()
    # manager.window.wm_geometry("+0+0") 

    ax = fig.add_subplot(111, projection='3d')
    plot_aruco_grid(ax)
    plot_coordinate_system(ax,length=6) # plot wworld coordinate system

    ax.grid(True)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust the margins to remove whitespace
    ax.set_box_aspect([1, 1, 1])

    for i, camera_name in enumerate(all_cameras): # for each camera
            camera_name.getFrame()
            camera_name.camera2world()
            camera_name.Rt2Pose(ax) # plot camera pose
            plot_coordinate_system(ax, origin=camera_name.t, R=camera_name.R, length=6) # plot camera coordinate system
            # for j, other_camera in enumerate(all_cameras): # for each camera, generate the epipole for each other camera
            #         if j == i:
            #                 continue  # Skip the iteration if j is equal to i
            #         field_name = f"camera{j+1}_epipole" 
            #         setattr(camera_name, field_name, camera_name.world2camera(other_camera.t)) # calculate epipole for each other camera

# set up recording
recording = False
recorded_data = []  # To store the recorded values

# print(camera1.Inew)
# print(camera1.R)
# print(camera1.t)
# cam1_t = -camera1.R.T @ camera1.t.reshape(-1,1)
# cam2_t = -camera2.R.T @ camera2.t.reshape(-1,1)
# P1 = camera1.Inew @ np.hstack((camera1.R.T, cam1_t))  # Projection matrix for camera 1
# P2 = camera2.Inew @ np.hstack((camera2.R.T, cam2_t))  # Projection matrix for camera 2

while 1: #______________________________________________________________________________________________
    if keyboard.is_pressed('esc'): 
        print('Seeyuh!')
        camera1.cap.release()
        cv2.destroyAllWindows()
        break

    frame1, centroids1 = camera1.getFrame()
    points1 = camera1.camera2world()
    frame2, centroids2 = camera2.getFrame()
    points2 = camera2.camera2world()
    
    if len(centroids1) > 0 and len(centroids2) > 0:
        centroid1 = np.array(centroids1[0], dtype=np.float32) 
        centroid2 = np.array(centroids2[0], dtype=np.float32) 

        point_4D_homogeneous = cv2.triangulatePoints(camera1.P, camera2.P, centroid1, centroid2)

        # Convert homogeneous 4D point to 3D
        point_3D = (point_4D_homogeneous[:3] / point_4D_homogeneous[3]).reshape(-1)

        # print(point_3D)
        if show_3D_plot:
            point_3D_plot = ax.scatter(point_3D[0], point_3D[1], point_3D[2], color='black', marker='o')



    rays1 = []
    rays2 = []
    ray_plots = []

    if len(points1) > 0:
        for idx, point in enumerate(points1):
            direction = point - camera1.t
            lam = -camera1.t[2] / direction[2] # calculate intersection with z=0 plane
            ground_intersection = camera1.t + lam*direction
            ray = np.array([camera1.t, ground_intersection])
            rays1.append(ray)
            if show_3D_plot:
                ray_plot, = ax.plot(ray[:,0], ray[:,1], ray[:,2], color='green')
                ray_plots.append(ray_plot)
    
    
    if len(points2) > 0:
        for idx, point in enumerate(points2):
            direction = point - camera2.t
            lam = -camera2.t[2] / direction[2] # calculate intersection with z=0 plane
            ground_intersection = camera2.t + lam*direction
            ray = np.array([camera2.t, ground_intersection])
            rays2.append(ray)
            if show_3D_plot:
                ray_plot, = ax.plot(ray[:,0], ray[:,1], ray[:,2], color='green')
                ray_plots.append(ray_plot)
    
    candidate_points = []
    candidate_point_plots = []
    closest_approaches = []  # List to hold tuples of (midpoint, closest_approach_distance)
    for idx1, ray1 in enumerate(rays1):
        for idx2, ray2 in enumerate(rays2):
            midpoint, closest_approach_distance = closest_approach_between_segments(ray1, ray2) # find where the rays almost intersect
            if closest_approach_distance < reprojection_error_threshold: # if the two rays form a close enough point...
                candidate_points.append((midpoint, closest_approach_distance)) # add to the list of resolved 3D points
                if print_calculations:
                    print(point_3D -midpoint)
                    # print(f"Candidate from Ray {idx1} and Ray {idx2}: Midpoint {midpoint}, Reprojection Error {closest_approach_distance}")
                if show_3D_plot:
                    candidate_point_plot, = ax.plot(midpoint[0], midpoint[1], midpoint[2], 'o', color='red')
                    candidate_point_plots.append(candidate_point_plot)

    # Sort the candidates list by the second element (closest_approach_distance)
    candidate_points.sort(key=lambda x: x[1])  # Sort by distance
  
    if show_3D_plot:
        ax.axis('equal')
        ax.set(xlim=(-50, 50), ylim=(-50, 50), zlim=(0, 96))
        plt.pause(.001)
    
        if len(ray_plots) > 0:
            for ray_plot in ray_plots:
                ray_plot.remove()
                
        if len(candidate_point_plots) > 0:
            point_3D_plot.remove()
            for candidate_point_plot in candidate_point_plots:
                candidate_point_plot.remove()
         
    if show_frames:
        scale = .8
        frame1 = cv2.resize(frame1, (0, 0), fx=scale, fy=scale)
        
        # Resize frame2 to have the same width as frame1 while maintaining its aspect ratio
        height1, width1 = frame1.shape[:2]
        height2, width2 = frame2.shape[:2]  
        scale_factor = width1 / width2
        frame2_resized = cv2.resize(frame2, (width1, int(height2 * scale_factor)))
        
        combined = np.vstack((frame1, frame2_resized))
        cv2.imshow('frames', combined)
        
    key = cv2.waitKey(1)    
    if keyboard.is_pressed('space') or key == 32:  # Space bar
        time.sleep(.3) 
        recording = not recording  # Toggle recording state
        if recording: 
            print("Started recording...")
            start_time = time.time()
        else:
            print("Stopped recording...")
            # Save recorded data to CSV
            with open('MATLAB/recorded_data.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time", "X", "Y", "Z", "Error"])  # Header
                writer.writerows(recorded_data)

    # Record the midpoint and error if recording is active
    if recording:
        elapsed_time = time.time() - start_time
        if len(candidate_points) > 0:  # Threshold to consider a successful approach
            midpoint = candidate_points[0][0]
            distance = candidate_points[0][1]
            recorded_data.append([elapsed_time, midpoint[0], midpoint[1], midpoint[2], distance])
              
 