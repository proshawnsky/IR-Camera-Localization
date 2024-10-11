import cv2
import os

# Set up folder to save images
save_folder = "calibration_images"
os.makedirs(save_folder, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 120)
# 10 = bright, -10 = dark, -13 = very dark
exposure_value = 10  # Adjust this value (-6 is typically low exposure)
cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Image counter
img_counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display the frame
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.putText(frame, f'{actual_width}x{actual_height}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Webcam", frame)

    # Wait for key press
    key = cv2.waitKey(1)

    # If spacebar is pressed (key code 32), save the frame
    if key == 32:  # Spacebar
        img_name = f"{save_folder}/image_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"Image {img_counter} saved to {img_name}")
        img_counter += 1

    # If Esc key is pressed (key code 27), exit the loop
    elif key == 27:  # Esc
        print("Exiting...")
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
