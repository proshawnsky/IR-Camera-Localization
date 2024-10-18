import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from real_camera import custom_real_camera
from utils import *
import csv
import itertools

# Options 
show_3D_plot = True
show_frames = True      
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
    manager = plt.get_current_fig_manager()
    # manager.window.wm_geometry("+0+0") 

    ax = fig.add_subplot(111, projection='3d')  
    # plot_aruco_grid(ax)
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
known_distances = np.array([7.339773093282274, 9.066638128175956, 11.75268715661555]) # real triangle dimensions (shinches)
triangle_plot = None
reprojection_error_threshold = .5
while 1: #______________________________________________________________________________________________
    if keyboard.is_pressed('esc'): 
        print('Seeyuh!')
        camera1.cap.release()
        cv2.destroyAllWindows()
        break

    frame1, centroids1 = camera1.getFrame()
    frame2, centroids2 = camera2.getFrame()

    cam1_rays = camera1.pixels2rays()
    cam2_rays = camera2.pixels2rays()
    points3D, reprojection_errors = triangulate(camera1.pixels2rays(), camera2.pixels2rays())
    
    filtered_points = []
    filtered_errors = []

    for point, error in zip(points3D, reprojection_errors):
        if error <= reprojection_error_threshold:
            filtered_points.append(point)
            filtered_errors.append(error)

    print(filtered_errors)
    possible_triangles = list(itertools.combinations(filtered_points, 3))
    best_triangle = None
    min_error = float('inf')

    for triangle in possible_triangles:
        error = triangle_similarity(triangle, known_distances)
        if error < min_error and error < 1:
            min_error = error
            best_triangle = triangle
    
    if triangle_plot is not None:
        triangle_plot.remove()

    if best_triangle is not None:
        p1, p2, p3 = best_triangle
        # Create arrays for plotting
        x_vals = [p1[0], p2[0], p3[0], p1[0]]
        y_vals = [p1[1], p2[1], p3[1], p1[1]]
        z_vals = [p1[2], p2[2], p3[2], p1[2]]

        # Plot the triangle lines
        triangle_plot, = ax.plot(x_vals, y_vals, z_vals, 'b-o', label='Best Triangle')
    else:
        triangle_plot = None
    # Plot the rays
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
    
    candidate_points = []
    candidate_point_plots = []
    
    
    for idx, point in enumerate(points3D):
        if reprojection_errors[idx] < reprojection_error_threshold and point[2] > -.1:
            candidate_points.append((point, reprojection_errors[idx]))
            if print_calculations:  
                print(f"Midpoint: {point}, Reprojection Error: {reprojection_errors[idx]}")

            # if show_3D_plot:
            #     point_3D_plot, = ax.plot(point[0], point[1], point[2], 'o', color='black')
            #     candidate_point_plots.append(point_3D_plot)

    # Sort the candidates list by the second element (closest_approach_distance)
    candidate_points.sort(key=lambda x: x[1])  # Sort by distance
  
    if show_3D_plot:
        ax.axis('equal')
        ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(0, 70))
        plt.pause(.001)
    
        if len(ray_plots) > 0:
            for ray_plot in ray_plots:
                ray_plot.remove()
                
        if len(candidate_point_plots) > 0:
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
              
 