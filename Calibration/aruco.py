"""
Framework   : OpenCV Aruco
Description : Calibration of camera and using that for finding pose of multiple markers
Status      : Working
References  :
    1) https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
    2) https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
    3) https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html
"""

import numpy as np
import cv2
import cv2.aruco as aruco
import time
camSelect = 1

if camSelect == 1:
    camID = 0
    data = np.load('camera1_calibration.npz')
    mtx = data['I']
    dist = data['dist_coeffs']
    print('Loaded Camera 1')
elif camSelect == 2:
    camID = 1
    data = np.load('camera2_calibration.npz')
    mtx = data['I']
    dist = data['dist_coeffs'] 
    print('Loaded Camera 2')
else:
    print("Enter a valid camera number")
    exit()

cap = cv2.VideoCapture(camID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 120)
# 10 = bright, -10 = dark, -13 = very dark
exposure_value = 10  # Adjust this value (-6 is typically low exposure)
cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))





last_print_time = time.time()
###------------------ ARUCO TRACKER ---------------------------
while (True):
    current_time = time.time()
    ret, frame_orig = cap.read()
    
    h, w = frame_orig.shape[:2]
    alpha = .2
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))

    # Undistort the frame
    frame_undistorted = cv2.undistort(frame_orig, mtx, dist, None, new_mtx)

    gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)

    # height, width, _ = frame.shape
    #     # Calculate the center coordinate

    # center_x = width // 2
    # center_y = height // 2
    # cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
    # set dictionary size depending on the aruco marker selected
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshConstant = 15

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        # estimate pose of each marker and return the valueqqs
        # rvet and tvec-different from camera coefficients

        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 10.787, new_mtx, distCoeffs=None)
        rotation_matrix, _ = cv2.Rodrigues(rvec[0])
        rotation_matrix_inv = np.transpose(rotation_matrix)
        camera_position_in_marker_frame = -np.dot(rotation_matrix_inv, tvec[0].reshape(3, 1))

        if current_time - last_print_time >= 2:
            print(f"t = np.array({camera_position_in_marker_frame.reshape(-1).tolist()})")
            print(f"R = np.array({rotation_matrix.tolist()}).T")
            print()
            last_print_time = current_time
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # Overwrite R and t in the npz file
            if camSelect == 1:
                np.savez('camera1_extrinsics.npz', R=rotation_matrix, t=camera_position_in_marker_frame, I = new_mtx)
                print("Wrote camera 1 R and t in file.")
            elif camSelect == 2:
                np.savez('camera2_extrinsics.npz', R=rotation_matrix, t=camera_position_in_marker_frame, I = new_mtx)
                print("Wrote camera 2 R and t in file.")

        for i in range(0, ids.size):
            # draw axis for the aruco markers
            cv2.drawFrameAxes(frame_undistorted, new_mtx, None, rvec[i], tvec[i], 10.787/2, )

        # draw a square around the markers
        aruco.drawDetectedMarkers(frame_undistorted, corners)
    
    # combined = np.hstack((frame_orig, frame_undistorted))
    
    # cv2.imshow('Original vs Undistorted', combined)
    cv2.imshow('frame',frame_undistorted)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()