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
camera1 = custom_real_camera(R = R1,
                        t = t1,
                       color='r',show_img=False, image_scale=1, I = I1, pose_depth=12, vidCapID=0,
                       distortion_coefficients = dist_coeffs1, undistort=False)

cam2_intrinsics = np.load('camera2_calibration.npz')
I2 = cam2_intrinsics['I']
dist_coeffs2 = cam2_intrinsics['dist_coeffs']
cam2_extrinsics = np.load('camera2_extrinsics.npz')
R2 = cam2_extrinsics['R'].T # marker position in camera frame
t2 = cam2_extrinsics['t'].reshape(-1)
camera2 = custom_real_camera(R = R2,
                        t = t2,
                       color='b',show_img=False, image_scale=1, I = I2, pose_depth=12, vidCapID=1,
                       distortion_coefficients = dist_coeffs2, undistort=False)
all_cameras = [camera1, camera2]


# Ps = [camera.P for camera in all_cameras]

# Set up the Wqorld 3D Plot ____________________________________________________________________________
room_center = np.array([0, 0, 0],dtype=np.float32)

if show_3D_plot:
    fig = plt.figure(1,figsize=(20, 12))
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+0+0") 

    ax = fig.add_subplot(111, projection='3d')
    plot_aruco_grid(ax)
    plot_coordinate_system(ax,length=6) # plot wworld coordinate system

    point_world_plot1, = ax.plot([room_center[0]], [room_center[1]], [room_center[2]], marker='o', markersize=5,color='k') # point on camera 1's pose pyramid
    point_world_plot2, = ax.plot([room_center[0]], [room_center[1]], [room_center[2]], marker='o', markersize=5,color='k') # point on camera 2's pose pyramid
    point_plot, = ax.plot([room_center[0]], [room_center[1]], [room_center[2]], marker='o', markersize=5,color='k') # resolved 3D point
    ray_plot1, = ax.plot([0], [0], [0], color='green', label='Ray') # ray from camera 1 to 3D point
    ray_plot2, = ax.plot([0], [0], [0], color='green', label='Ray') # ray from camera 2 to 3D point
    # floor_point_plot, = ax.plot([0],[0],[0], marker='o', markersize=5,color='r', label='Floor Point')  REMOVE???
    
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
    P1 = camera1.camera2world()
    frame2 = camera2.getFrame()
    P2 = camera2.camera2world()

    if show_frames:
        scale = .8
        frame1 = cv2.resize(frame1, (0, 0), fx=scale, fy=scale)
        frame2 = cv2.resize(frame2, (0, 0), fx=scale, fy=scale)
        combined = np.vstack((frame1, frame2))
        cv2.imshow('frames', combined)

    direction = P1 - camera1.t
    lam = -camera1.t[2] / direction[2] # calculate intersection with z=0 plane
    ground_intersection = camera1.t + lam*direction*1.1
    ray_line1 = np.array([camera1.t, ground_intersection])

    direction = P2 - camera2.t
    lam = -camera2.t[2] / direction[2] # calculate intersection with z=0 plane
    ground_intersection = camera2.t + lam*direction*1.1
    ray_line2 = np.array([camera2.t, ground_intersection])
        
    midpoint, distance = closest_approach_between_segments(ray_line1, ray_line2)
    

    if show_3D_plot: 
        # camera 1 plots
        point_world_plot1.set_data([P1[0]], [P1[1]])
        point_world_plot1.set_3d_properties([P1[2]])
        ray_plot1.set_data(ray_line1[:, 0], ray_line1[:, 1])
        ray_plot1.set_3d_properties(ray_line1[:, 2])
        ray_plot1.set_zorder(1000)

        # camera 2 plots
        point_world_plot2.set_data([P2[0]], [P2[1]])
        point_world_plot2.set_3d_properties([P2[2]])
        ray_plot2.set_data(ray_line2[:, 0], ray_line2[:, 1])
        ray_plot2.set_3d_properties(ray_line2[:, 2])
        ray_plot2.set_zorder(1000)
        
        point_plot.set_data([midpoint[0]], [midpoint[1]])
        point_plot.set_3d_properties([midpoint[2]])
        ax.axis('equal')
        plt.pause(.001)



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
        if distance <= 4:  # Threshold to consider a successful approach
            recorded_data.append([elapsed_time, midpoint[0], midpoint[1], midpoint[2], distance])
            if print_calculations:
                print(f"Time: {elapsed_time:.2f} s, Midpoint: {midpoint}, Error: {distance}")
 