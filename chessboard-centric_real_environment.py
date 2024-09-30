import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import spacenavigator
from real_camera import custom_real_camera
from utils import *

# Constants

room_center = np.array([-2.25*3, -2.25*3, 0],dtype=np.float32)

I_default = np.array( [[1.12114535e+03, 0.00000000e+00, 6.50227771e+02],
                        [0.00000000e+00, 1.11768069e+03, 3.56113978e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
R_default = np.array([
    [0.918023, 0.303866, -0.254752],
    [-0.396316, 0.682231, -0.614406],
    [-0.012898, 0.665001, 0.746731]
])
t_default = np.array([-22.521358, -66.545104, 71.727086])
# Define Cameras ___________________________________________________________________________________
camera1 = custom_real_camera(R = R_default,
                        t = t_default,
                        color='r',show_img=True, image_scale=1, I = I_default, pose_depth=12)
print(camera1.t)
all_cameras = [camera1]
Ps = [camera.P for camera in all_cameras]

# Set up the Wqorld 3D Plot ____________________________________________________________________________
fig = plt.figure(1,figsize=(20, 12))
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+0+0") 

ax = fig.add_subplot(111, projection='3d')
plot_coordinate_system(ax,length=12) # plot wworld coordinate system

plot_chessboard(ax, room_center)
point_world_plot, = ax.plot([room_center[0]], [room_center[1]], [room_center[2]], marker='o', markersize=5,color='k')
room_center_plot, = ax.plot([room_center[0]], [room_center[1]], [room_center[2]], marker='o', markersize=5,color='k')
ray_plot, = ax.plot([0], [0], [0], color='green', label='Ray')
floor_point_plot, = ax.plot([0],[0],[0], marker='o', markersize=5,color='r', label='Floor Point')

ax.set_box_aspect([1, 1, 1])
for i, camera_name in enumerate(all_cameras): # for each camera
        camera_name.Rt2Pose(ax) # plot camera pose
        plot_coordinate_system(ax, origin=camera_name.t, R=camera_name.R, length=12) # plot camera coordinate system
        for j, other_camera in enumerate(all_cameras): # for each camera
                if j == i:
                        continue  # Skip the iteration if j is equal to i
                field_name = f"camera{j+1}_epipole" 
                setattr(camera_name, field_name, camera_name.world2camera(other_camera.t)) # calculate epipole for each other camera
                
# Set up the Camera 2D Plot_____________________________________________________________________________

while 1: #______________________________________________________________________________________________
    if keyboard.is_pressed('esc'):
        print('Seeyuh!')
        camera1.cap.release()
        cv2.destroyAllWindows()
        break
    camera1.getFrame()
    P = camera1.camera2world()
    point_world_plot.set_data([P[0]], [P[1]])
    point_world_plot.set_3d_properties([P[2]])

    direction = P - camera1.t
    lam = -camera1.t[2] / direction[2] # calculate intersection with z=0 plane
    ground_intersection = camera1.t + lam*direction
    ray_line = np.array([camera1.t, ground_intersection])
    ray_plot.set_data(ray_line[:, 0], ray_line[:, 1])
    ray_plot.set_3d_properties(ray_line[:, 2])
    ray_plot.set_zorder(1000)

    floor_point_plot.set_data([ground_intersection[0]], [ground_intersection[1]])
    floor_point_plot.set_3d_properties([ground_intersection[2]])
    floor_point_plot.set_zorder(1001)
    ax.axis('equal')
    plt.pause(.001)
        
 