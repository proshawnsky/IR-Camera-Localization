import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import spacenavigator
from real_camera import custom_real_camera
from utils import *

# Constants
INCH2MM = 25.4
MM2INCH = 1/25.4
room_dim = np.array([131, 127, 106.5],dtype=np.float32) # office dimension (inches) x = along window
room_x, room_y, room_z = room_dim
room_center = np.array([room_x/2, room_y/2, 0],dtype=np.float32)

I_default = np.array([[669.5, 0,             320],
                      [0,             669.5, 240  ],
                      [0,             0,             1    ]]) # 640x480 30 fps default intrinsics
I_default = np.array([[500, 0,             320],
                      [0,             500, 240  ],
                      [0,             0,             1    ]]) # 640x480 30 fps default intrinsics
# Define Cameras ___________________________________________________________________________________
camera1 = custom_real_camera(t=np.array([room_x-12,24.5,72.5]),
                             I=I_default,color='r',show_img=True,
                             pose_depth = 2*12)
camera1.set_boresight(room_center)

all_cameras = [camera1]
Ps = [camera.P for camera in all_cameras]

# Set up the Wqorld 3D Plot ____________________________________________________________________________
fig = plt.figure(1,figsize=(15, 12))
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+0+0") 

ax = fig.add_subplot(111, projection='3d')
ax.set(xlim=(0, room_x), ylim=(0, room_y), zlim=(0, room_z), xlabel='x', ylabel='y', zlabel='z')
plot_coordinate_system(ax,length=24) # plot wworld coordinate system

plot_chessboard(ax, room_center)
point_world_plot, = ax.plot([room_center[0]], [room_center[1]], [room_center[2]], marker='o', markersize=5,color='k')
room_center_plot, = ax.plot([room_center[0]], [room_center[1]], [room_center[2]], marker='o', markersize=5,color='k')
ray_plot, = ax.plot([0], [0], [0], color='green', label='Ray')

for i, camera_name in enumerate(all_cameras): # for each camera
        camera_name.Rt2Pose(ax) # plot camera pose
        plot_coordinate_system(ax, origin=camera_name.t, R=camera_name.R, length=24) # plot camera coordinate system
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
    plt.pause(.001)
        
 