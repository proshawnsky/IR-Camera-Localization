import cv2
import numpy as np

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# Camera matrix and distortion coefficients
camera_matrix = np.array([[1.12114535e+03, 0, 6.50227771e+02],
                           [0, 1.12114535e+03, 3.56113978e+02],
                           [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.array([[-0.46781635, 0.33714806, 0.00609717, -0.00131893, -0.17436972]], dtype=np.float32)

chessboard_size = (7, 7)  # 8x8 internal corners
cell_size = 2.25  # Size of each square in meters

# Prepare object points for the chessboard corners (Z=0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * cell_size

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    
    
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    if ret == True:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coeffs)
        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
    

    # Show the frame with the drawn chessboard and axes
    cv2.imshow('Chessboard Pose Estimation', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()