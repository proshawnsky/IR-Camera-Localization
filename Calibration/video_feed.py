import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 120)
# 10 = bright, -10 = dark, -13 = very dark
exposure_value = 13  # Adjust this value (-6 is typically low exposure)
cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Actual Resolution: {int(actual_width)} x {int(actual_height)}")
print(f"Set Frame Rate: {actual_fps} FPS")

start_time  = time.time()
frame_count = 0
fps = 0
scale = 1
while True:
    # Capture the frame
    _, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    height, width, _ = frame.shape
    center_x = width // 2
    center_y = height // 2
    cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)
            
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > .2:  # Update FPS every second
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Display the result
    cv2.putText(frame, f'{actual_width}x{actual_height}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Frame', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
