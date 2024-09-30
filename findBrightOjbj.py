import numpy as np
import cv2
import time

# Initialize webcam capture
# 320 x 200 = 120 fps
# 320 x 240 = 60 fps


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 120)
exposure_value = -10  # Adjust this value (-6 is typically low exposure)
cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Actual Resolution: {int(actual_width)} x {int(actual_height)}")
print(f"Actual Frame Rate: {actual_fps} FPS")
start_time  = time.time()
frame_count = 0
fps = 0
cX, cY = 0, 0
# Set the threshold for brightness (adjust as necessary)
# brightness_threshold = 100
scale = 1
while True:
    # Capture the frame
    _, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Threshold the image to get the bright areas
    _, gray = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # scaling_factor = 0.5  # 0.5 reduces brightness by 50%, adjust as needed
    # gray = cv2.convertScaleAbs(gray, alpha=scaling_factor, beta=0)
    # Find contours of the thresholded image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(frame)

# Draw all contours in white
    height, width, _ = frame.shape
    # Calculate the center coordinates
    center_x = width // 2
    center_y = height // 2
    cv2.circle(frame, (center_x, center_y), 1, (0, 0, 255), -1)
    if contours:
        # Find the largest contour, assuming it's the bright object
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the centroid of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Draw a circle at the centroid
            
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > .2:  # Update FPS every second
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()


#     # Display the result
    
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.circle(frame, (cX, cY), 5, (0, 255, 0),thickness=2)
    cv2.imshow('Frame', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
