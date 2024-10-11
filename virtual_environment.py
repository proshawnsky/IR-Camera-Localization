import numpy as np
import matplotlib.pyplot as plt
import keyboard
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import spacenavigator
from virtual_camera import custom_camera
from utils import *
FT2MM = 0.3048*1000
# Define Cameras ___________________________________________________________________________________
room_center = np.array([10/2*FT2MM, 10/2*FT2MM, 2*FT2MM],dtype=np.float32)
camera1 = custom_camera(t=np.array([0,0,6*FT2MM]),color='r')
camera1.set_boresight(room_center)
# camera1.add_pointing_error(.005)

camera2 = custom_camera(t=np.array([10*FT2MM,0,6*FT2MM]),color='g')
camera2.set_boresight(room_center)
# camera2.add_pointing_error(.005)

camera3 = custom_camera(t=np.array([10*FT2MM,10*FT2MM,6*FT2MM]),color='b')
camera3.set_boresight(room_center)
# camera3.add_pointing_error(.005)

camera4 = custom_camera(t=np.array([0,10*FT2MM,6*FT2MM]),color='m')
camera4.set_boresight(room_center)
# camera4.add_pointing_error(.1)

all_cameras = [camera1, camera2, camera3, camera4]
colors = ['r','g','b','m']
Ps = [camera.P for camera in all_cameras]

# Set up the World 3D Plot ____________________________________________________________________________
fig = plt.figure(1,figsize=(10, 10))
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+100+100") 

ax = fig.add_subplot(111, projection='3d')
ax.set(xlim=(0, 10*FT2MM), ylim=(0, 10*FT2MM), zlim=(0, 10*FT2MM), xlabel='x', ylabel='y', zlabel='z')
plot_coordinate_system(ax,length=3) # plot wworld coordinate system
point_world = np.array([10/2*FT2MM, 10/2*FT2MM, 0],dtype=np.float32) # initialize world point
point_world_est = np.array([0,0,0],dtype=np.float32)
point_world_plot, = ax.plot([point_world[0]], [point_world[1]], [point_world[2]], marker='o', markersize=5,color='k')
point_world_est_plot, = ax.plot([point_world_est[0]], [point_world_est[1]], [point_world_est[2]], marker='o', markersize=5,color='r')

for i, camera_name in enumerate(all_cameras): # for each camera
        camera_name.Rt2Pose(ax, pyramid_height=2,scale = 200, alpha = 0.8) # plot camera pose
        plot_coordinate_system(ax, origin=camera_name.t, R=camera_name.R, length=100) # plot camera coordinate system
        for j, other_camera in enumerate(all_cameras): # for each camera
                if j == i:
                        continue  # Skip the iteration if j is equal to i
                field_name = f"camera{j+1}_epipole" 
                setattr(camera_name, field_name, camera_name.world2camera(other_camera.t)) # calculate epipole for each other camera
                
# Set up the Camera 2D Plot_____________________________________________________________________________
fig_camviews, axs_camviews = plt.subplots(2,2,figsize=(8, 6))
success = spacenavigator.open()
spacemouse_gain = 200
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+1200+100") 
camviews_plots = []    

for i, axs in enumerate(axs_camviews.flat):
        all_cameras[i].world2camera(point_world)
        plot, = axs.plot(all_cameras[i].point_frame[0], all_cameras[i].point_frame[1], marker='o', markersize=5, color=colors[i])
        camviews_plots.append(plot)
        axs.set(xlim=(0, 1080), ylim=(0, 720), xlabel='x', ylabel='y')
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
    point_world = point_world.clip(0,9.99*FT2MM)

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
    point_world_est = triangulate(Ps, all_points)
    point_world_est_plot.set_data([point_world_est[0]], [point_world_est[1]])
    point_world_est_plot.set_3d_properties([point_world_est[2]])

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    print(point_world)
    plt.pause(.001)
        
 