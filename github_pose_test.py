import numpy as np
import cv2
import glob
# Load previously saved data
mtx = np.array([[1.12114535e+03, 0, 6.50227771e+02],
                           [0, 1.12114535e+03, 3.56113978e+02],
                           [0, 0, 1]], dtype=np.float32)

dist = np.array([[-0.46781635, 0.33714806, 0.00609717, -0.00131893, -0.17436972]], dtype=np.float32)

def draw(img, corners, imgpts):
    
    corner = tuple(corners[0].ravel().astype(int))


    img = cv2.line(img, corner, tuple(imgpts[0].astype(int).ravel()),color=(0,255,0), thickness=3)
    img = cv2.line(img, corner, tuple(imgpts[1].astype(int).ravel()),color=(0,0,255), thickness=3)
    img = cv2.line(img, corner, tuple(imgpts[2].astype(int).ravel()),color=(255,0,0), thickness=3)
    return img


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)*2.25
axis = np.float32([[10,0,0], [0,10,0], [0,0,-10]]).reshape(-1,3)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

while True:
    # Capture frame from the webcam
    ret, img = cap.read()    
    
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (img.shape[1], img.shape[0]), 1, (img.shape[1], img.shape[0]))
    # img = cv2.undistort(img, mtx, dist, None, new_camera_matrix)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,7),None)
    if ret == True:
        
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # corners2 = corners2[::-1]
        cv2.drawChessboardCorners(img, (7,7), corners, True)

        # Find the rotation and translation vectors.
        success, rotation_vector, translation_vector = cv2.solvePnP(objp, corners2, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, mtx, dist)
        img = draw(img,corners2,imgpts)
        
        rotation_vector = -rotation_vector
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Invert the rotation matrix
        rotation_matrix_inv = np.linalg.inv(rotation_matrix)

        # Transform the translation vector to the chess board's coordinate system
        translation_vector_cb = rotation_matrix @ translation_vector

        print("R = np.array([")
        print("    [%f, %f, %f]," % (rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2]))
        print("    [%f, %f, %f]," % (rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2]))
        print("    [%f, %f, %f]" % (rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2]))
        print("])")

        print("t = np.array([%f, %f, %f])" % (translation_vector_cb[0], translation_vector_cb[1], translation_vector_cb[2]))

        # project 3D points to image plane
        # print(tvecs)
        

       
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()