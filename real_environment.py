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
show_frames = True
print_calculations = True  

# Define Cameras ___________________________________________________________________________________
cam1_intrinsics = np.load('camera1_calibration.npz')
I1 = cam1_intrinsics['I']
dist_coeffs1 = cam1_intrinsics['dist_coeffs']
cam1_extrinsics = np.load('camera1_extrinsics.npz')
R1 = cam1_extrinsics['R'].T
t1 = cam1_extrinsics['t'].reshape(-1)
Inew1 = cam1_extrinsics['I'] # AFTER UNDISTORTION
camera1 = custom_real_camera(R = R1,
                        t = t1,
                       color='r',show_img=False, image_scale=1, I = I1, Inew = Inew1, pose_depth=12, vidCapID=0,
                       distortion_coefficients = dist_coeffs1, undistort=False, cameraID=1)

cam2_intrinsics = np.load('camera2_calibration.npz')
I2 = cam2_intrinsics['I']
dist_coeffs2 = cam2_intrinsics['dist_coeffs']
cam2_extrinsics = np.load('camera2_extrinsics.npz')
R2 = cam2_extrinsics['R'].T # marker position in camera frame
t2 = cam2_extrinsics['t'].reshape(-1)
Inew2 = cam2_extrinsics['I'] # AFTER UNDISTORTION
camera2 = custom_real_camera(R = R2,
                        t = t2,
                       color='b',show_img=False, image_scale=1, I = I2, Inew = Inew2, pose_depth=12, vidCapID=1,
                       distortion_coefficients = dist_coeffs2, undistort=False, cameraID=2)
all_cameras = [camera1, camera2]

# Set up the Wqorld 3D Plot ____________________________________________________________________________
room_center = np.array([0, 0, 0],dtype=np.float32)

if show_3D_plot:
    fig = plt.figure(1,figsize=(20, 12))
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+0+0") 

    ax = fig.add_subplot(111, projection='3d')
    plot_aruco_grid(ax)
    plot_coordinate_system(ax,length=6) # plot wworld coordinate system

    ax.grid(False)
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
            for j, other_camera in enumerate(all_cameras): # for each camera
                    if j == i:
                            continue  # Skip the iteration if j is equal to i
                    field_name = f"camera{j+1}_epipole" 
                    setattr(camera_name, field_name, camera_name.world2camera(other_camera.t)) # calculate epipole for each other camera

# set up recording
recording = False
recorded_data = []  # To store the recorded values

while 1: #______________________________________________________________________________________________
    if keyboard.is_pressed('esc'): 
        print('Seeyuh!')
        camera1.cap.release()
        cv2.destroyAllWindows()
        break

    frame1 = camera1.getFrame()
    points1 = camera1.camera2world()
    frame2 = camera2.getFrame()
    points2 = camera2.camera2world()
    rays1 = []
    rays2 = []
    ray_plots = []

    if len(points1) > 0:
        for idx, point in enumerate(points1):
            direction = point - camera1.t
            lam = -camera1.t[2] / direction[2] # calculate intersection with z=0 plane
            ground_intersection = camera1.t + lam*direction*1.1
            ray = np.array([camera1.t, ground_intersection])
            rays1.append(ray)
            ray_plot, = ax.plot(ray[:,0], ray[:,1], ray[:,2], color='green')
            ray_plots.append(ray_plot)
    ax.axis('equal')
    
    if len(points2) > 0:
        for idx, point in enumerate(points2):
            direction = point - camera2.t
            lam = -camera2.t[2] / direction[2] # calculate intersection with z=0 plane
            ground_intersection = camera2.t + lam*direction*1.1
            ray = np.array([camera2.t, ground_intersection])
            rays2.append(ray)
            ray_plot, = ax.plot(ray[:,0], ray[:,1], ray[:,2], color='green')
            ray_plots.append(ray_plot)
    
    candidate_points = []
    candidate_point_plots = []
    closest_approaches = []  # List to hold tuples of (midpoint, closest_approach_distance)
    for idx1, ray1 in enumerate(rays1):
        for idx2, ray2 in enumerate(rays2):
            midpoint, closest_approach_distance = closest_approach_between_segments(ray1, ray2)
            if closest_approach_distance < 20:
                closest_approaches.append((midpoint, closest_approach_distance))
                
                # candidate_points.append(midpoint)
                # candidate_point_plot, = ax.plot(midpoint[0], midpoint[1], midpoint[2], 'o', color='red')
                # candidate_point_plots.append(candidate_point_plot)
                # print(f"Closest approach distance for ray1 {idx1} and ray2 {idx2}: {closest_approach_distance}")
    # Sort the closest_approaches list by the second element (closest_approach_distance)
    closest_approaches.sort(key=lambda x: x[1])  # Sort by distance
    for midpoint, distance in closest_approaches:
        candidate_points.append(midpoint)
        candidate_point_plot, = ax.plot(midpoint[0], midpoint[1], midpoint[2], 'o', color='red')
        candidate_point_plots.append(candidate_point_plot)
    for idx, (midpoint, distance) in enumerate(closest_approaches):
        print(f"Sorted Candidate {idx}: Midpoint {midpoint}, Distance {distance}")

    ax.axis('equal')
    plt.pause(.1)

    if len(ray_plots) > 0:
        for ray_plot in ray_plots:
            ray_plot.remove()
    if len(candidate_point_plots) > 0:
        for candidate_point_plot in candidate_point_plots:
            candidate_point_plot.remove()
         
    if show_frames:
        scale = .8
        frame1 = cv2.resize(frame1, (0, 0), fx=scale, fy=scale)
        frame2 = cv2.resize(frame2, (0, 0), fx=scale, fy=scale)
        combined = np.vstack((frame1, frame2))
        cv2.imshow('frames', combined)

        
    # midpoint, distance = closest_approach_between_segments(ray[0], ray_line2)

    # key = cv2.waitKey(1)    
    # if keyboard.is_pressed('space') or key == 32:  # Space bar
    #     time.sleep(.3) 
    #     recording = not recording  # Toggle recording state
    #     if recording: 
    #         print("Started recording...")
    #         start_time = time.time()
    #     else:
    #         print("Stopped recording...")
    #         # Save recorded data to CSV
    #         with open('MATLAB/recorded_data.csv', mode='w', newline='') as file:
    #             writer = csv.writer(file)
    #             writer.writerow(["Time", "X", "Y", "Z", "Error"])  # Header
    #             writer.writerows(recorded_data)

    # # Record the midpoint and error if recording is active
    # if recording:
    #     elapsed_time = time.time() - start_time
    #     if distance <= 4:  # Threshold to consider a successful approach
    #         recorded_data.append([elapsed_time, midpoint[0], midpoint[1], midpoint[2], distance])
    #         if print_calculations:
    #             print(f"Time: {elapsed_time:.2f} s, Midpoint: {midpoint}, Error: {distance}")
 