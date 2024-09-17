import numpy as np
import matplotlib.pyplot as plt
import keyboard
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import spacenavigator
from camera import custom_camera
from utils import *

# Define Cameras ___________________________________________________________________________________
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
all_cameras = [camera1, camera2, camera3, camera4]

colors = ['r','g','b','m']
success = spacenavigator.open()
# camera2.P = np.concatenate([camera1.R.T @ camera2.R, (camera2.t - camera1.t).reshape(3,1)], axis=-1)
Ps = []
for i, camera in enumerate(all_cameras):
        Ps.append(camera.P)

# Set up the World 3D Plot ____________________________________________________________________________
fig = plt.figure(1,figsize=(10, 10))
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+100+100") 
spacemouse_gain = 2
ax = fig.add_subplot(111, projection='3d')
ax.set(xlim=(0, 10), ylim=(0, 10), zlim=(0, 10))
plot_coordinate_system(ax,length=3) # plot wworld coordinate system
point_world = np.array([5,5,0],dtype=np.float32) # initialize world point
point_world_plot, = ax.plot([point_world[0]], [point_world[1]], [point_world[2]], marker='o', markersize=5,color='k')
for i, camera_name in enumerate(all_cameras): # for each camera
        camera_name.Rt2Pose(ax) # plot camera pose
        plot_coordinate_system(ax, origin=camera_name.t, R=camera_name.R, length=1) # plot camera coordinate system
        for j, other_camera in enumerate(all_cameras): # for each camera
                if j == i:
                        continue  # Skip the iteration if j is equal to i
                field_name = f"camera{j+1}_epipole" 
                setattr(camera_name, field_name, camera_name.world2camera(other_camera.t)) # calculate epipole for each other camera
                
# Set up the Camera 2D Plot_____________________________________________________________________________
fig_camviews, axs_camviews = plt.subplots(2,2,figsize=(8, 6))
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+1200+100") 
camviews_plots = []    

for i, axs in enumerate(axs_camviews.flat):
        all_cameras[i].world2camera(point_world)
        plot, = axs.plot(all_cameras[i].point_frame[0], all_cameras[i].point_frame[1], marker='o', markersize=5, color=colors[i])
        camviews_plots.append(plot)
        axs.set(xlim=(-5, 5), ylim=(-5, 5), xlabel='x', ylabel='y')
        axs.grid()
        axs.title.set_text(f"Camera {i+1}")
        for j, camera_names2 in enumerate(all_cameras):
                if j == i:
                        continue  # Skip the iteration if j is equal to i
                field_name = f"camera{j+1}_epipole" # get epipole for each other camera
                axs.plot(getattr(all_cameras[i],field_name)[0], getattr(all_cameras[i],field_name)[1], marker='o', markersize=5, color=colors[j])
              
while 1: #______________________________________________________________________________________________
    if keyboard.is_pressed('esc'):
        print('Seeyuh!')
        break
    
    # Update the world point position using the space mouse
    state = spacenavigator.read()
    point_world += spacemouse_gain*np.array([state.x, state.y, state.z])
    point_world = point_world.clip(0,10)

    # Update the point world plot
    point_world_plot.set_data([point_world[0]], [point_world[1]])
    point_world_plot.set_3d_properties([point_world[2]])
    
    # Calculate the camera famres and update the plots
    for i, axs in enumerate(axs_camviews.flat):
        point = all_cameras[i].world2camera(point_world)
        camviews_plots[i].set_data([point[0]],[point[1]])

        # for j, camera_names2 in enumerate(all_cameras):
        #       if j == i:
        #         continue
        #       field_name = f"camera{j+1}_epipole" 
        #       epipole = getattr(all_cameras[i], field_name)

    all_points = np.array([camera1.point_frame, camera2.point_frame, camera3.point_frame, camera4.point_frame])
#     print(all_points)
    pos = triangulate(Ps, all_points)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    print(pos)
    plt.pause(.001)
        
 