import cv2
import numpy as np

def classify_motion(frame, x, y, w, h):
    global prev_gray_roi

    # Extract the region of interest (ROI) from the frame
    roi = frame[y:y+h, x:x+w]

    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if prev_gray_roi is not None and prev_gray_roi.shape == gray_roi.shape:
        # Resize gray_roi to match prev_gray_roi dimensions
        gray_roi_resized = cv2.resize(gray_roi, prev_gray_roi.shape[::-1])

        # Calculate motion type based on intensity difference
        intensity_diff = np.mean(np.abs(gray_roi_resized - prev_gray_roi))

        # Define a threshold for motion detection
        motion_threshold = 20

        if intensity_diff > motion_threshold:
            motion_type = "Moving"
        else:
            motion_type = "Static"
    else:
        motion_type = "Static"

    # Update prev_gray_roi for the next iteration
    prev_gray_roi = gray_roi.copy()

    return motion_type

# Load video
cap = cv2.VideoCapture("video.mp4")

# Define motion classes
motion_classes = ["Moving", "Jumping", "Rolling", "Sliding"]

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize prev_gray_roi for motion analysis
prev_gray_roi = None

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Apply morphological operations to remove noise
    fg_mask = cv2.erode(fg_mask, None, iterations=2)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calculate bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Perform motion analysis and classification here
        motion_type = classify_motion(frame, x, y, w, h)  # Call the motion classification function

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label motion type
        cv2.putText(frame, motion_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display results
    cv2.imshow("Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
