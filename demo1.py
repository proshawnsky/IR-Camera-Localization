import numpy as np
from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
import matplotlib.pyplot as plt
import keyboard
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import spacenavigator
from camera import custom_camera
from utils import plot_coordinate_system

# Define Cameras
room_center = np.array([5, 5, 0],dtype=np.float32)
camera1 = custom_camera(t=np.array([0,0,10]),color='r')
camera1.set_boresight(room_center)
camera2 = custom_camera(t=np.array([10,0,10]),color='g')
camera2.set_boresight(room_center)
camera3 = custom_camera(t=np.array([10,10,10]),color='b')
camera3.set_boresight(room_center)
camera4 = custom_camera(t=np.array([0,10,10]),color='m')
camera4.set_boresight(room_center)

all_cameras = [camera1, camera2, camera3, camera4]
colors = ['r','g','b','m']
success = spacenavigator.open()

# Set up the World 3D Plot
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
plot_coordinate_system(ax,length=3)
point_world = np.array([5,5,0],dtype=np.float32)
point_world_plot, = ax.plot([point_world[0]], [point_world[1]], [point_world[2]], marker='o', markersize=10,color='r')
for i, camera_name in enumerate(all_cameras):
        camera_name.Rt2Pose(ax) # plot camera pose
        plot_coordinate_system(ax, origin=camera_name.t, R=camera_name.R, length=2)

# Set up the Camera 2D Plot
fig_camviews, axs_camviews = plt.subplots(2, 2)
camviews_plots = []    

# Initialize the 4 Frame views
for i, axs in enumerate(axs_camviews.flat):
        all_cameras[i].world2camera(point_world)
        plot, = axs.plot(all_cameras[i].point_camera[0], all_cameras[i].point_camera[1], marker='o', markersize=10, color=colors[i])
        camviews_plots.append(plot)
        axs.set(xlim=(-20, 20), ylim=(-20, 20), xlabel='x', ylabel='y')
        axs.grid()

while 1:
    if keyboard.is_pressed('esc'):
        print('esc pressed!')
        break
    # Update the world point position
    state = spacenavigator.read()
    point_world += 2*np.array([state.x, state.y, state.z])
    point_world = point_world.clip(0,10)
    # Update the point world plot
    point_world_plot.set_data([point_world[0]], [point_world[1]])
    point_world_plot.set_3d_properties([point_world[2]])
    plt.pause(.001)
    # plt.ion()

    ax.set(xlim=(-5,15), ylim=(-5,15), zlim=(-5,15), xlabel='x', ylabel='y', zlabel='z')

    for i, axs in enumerate(axs_camviews.flat):
         all_cameras[i].world2camera(point_world)
         camviews_plots[i].set_data([all_cameras[i].point_camera[0]],[all_cameras[i].point_camera[1]])
    # print(point_world.round(1), camera3.point_camera.reshape(3).round(1))
    # print(camera3.R)