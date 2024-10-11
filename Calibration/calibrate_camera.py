import cv2
import numpy as np
import glob

# Define the dimensions of the chessboard
chessboard_size = (7, 7)  # Number of inner corners in the chessboard (9x6 chessboard has 10x7 squares)
square_size = 2.25  # Size of a square (in some unit, e.g., meters or millimeters)

# Termination criteria for refining corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in the real world space)
# 3D points for the chessboard in real world space, (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Scale object points by square size

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load the images
images = glob.glob('calibration_images/*.png')  # Replace with the path to your chessboard images
for img_file in images:
    # Load each image
    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Chessboard Corners', img)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)  # Add the 3D points of the chessboard
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # Refine corner locations
        imgpoints.append(corners2)  # Add the 2D points of the detected corners

        # Draw and display the corners for visualization
        img_with_corners = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Chessboard Corners', img_with_corners)
        cv2.waitKey(200)

cv2.destroyAllWindows()

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Output the intrinsic camera parameters
print("Camera matrix:\n", camera_matrix)
print("\nDistortion coefficients:\n", dist_coeffs)
# print("\nRotation vectors:\n", rvecs)
# print("\nTranslation vectors:\n", tvecs)

# Optionally save the calibration result for later use
np.savez('camera1_calibration.npz', I=camera_matrix, dist_coeffs=dist_coeffs)


