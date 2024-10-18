# IR-Camera-Localization
 
FEC = For Each Camera

Step 1: Camera Intrinsics
FEC:
- run generate_calibration_images.py
- take 10+ pictures of a chess board from different angles and distances 
- run calibrate_camera.py to process the chessboard images and calculate the intrinsic matrix and distortion parameters
- within calibrate_camera.py, make sure to change the name of the .npz file

Step 2: Initial Pose Estimation
- place an aruco marker on the floor where it is in all cameras' fields of view 
- FEC, run aruco.py
- when the camera identifies the marker and presses 
