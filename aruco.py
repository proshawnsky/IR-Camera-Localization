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

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 120)
# cap.set(cv2.CAP_PROP_EXPOSURE, 50)
mtx = np.array([[1.12114535e+03, 0, 6.50227771e+02],
                           [0, 1.12114535e+03, 3.56113978e+02],
                           [0, 0, 1]], dtype=np.float32)

dist = np.array([[-0.46781635, 0.33714806, 0.00609717, -0.00131893, -0.17436972]], dtype=np.float32)

###------------------ ARUCO TRACKER ---------------------------
while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 10.75, mtx, dist)
        rotation_matrix, _ = cv2.Rodrigues(rvec[0])
        rotation_matrix_inv = np.transpose(rotation_matrix)
        camera_position_in_marker_frame = -np.dot(rotation_matrix_inv, tvec[0].reshape(3, 1))

        print(f"t = np.array({camera_position_in_marker_frame.reshape(-1).tolist()})")

# Print rotation matrix `R`
        print(f"R = np.array({rotation_matrix.tolist()})")

        for i in range(0, ids.size):
            # draw axis for the aruco markers
            cv2.drawFrameAxes(frame, mtx, dist, rvec[i], tvec[i], 12)

        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)
    frame = cv2.resize(frame, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_LINEAR)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()