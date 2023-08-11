def classify_motion(frame, x, y, w, h):
    # Extract the region of interest (ROI) from the frame
    roi = frame[y:y+h, x:x+w]

    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction to ROI
    fg_mask_roi = bg_subtractor.apply(gray_roi)

    # Calculate percentage of foreground pixels
    total_pixels = fg_mask_roi.shape[0] * fg_mask_roi.shape[1]
    foreground_pixels = np.count_nonzero(fg_mask_roi)
    foreground_percentage = (foreground_pixels / total_pixels) * 100

    # Define a threshold for motion detection
    motion_percentage_threshold = 5  # Adjust as needed

    if foreground_percentage > motion_percentage_threshold:
        # Calculate bounding box aspect ratio
        aspect_ratio = w / h

        # Calculate bounding box area
        area = w * h

        # Classify motion based on aspect ratio and area
        if aspect_ratio < 0.7:
            motion_type = "Sliding"
        elif area > 10000:  # Adjust area threshold as needed
            motion_type = "Jumping"
        elif area > 2000:   # Adjust area threshold as needed
            motion_type = "Rolling"
        else:
            motion_type = "Moving"
    else:
        motion_type = "Static"

    return motion_type

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    intersection_area = x_intersection * y_intersection
    union_area = w1 * h1 + w2 * h2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

# main code
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load YOLOv3
net = cv2.dnn.readNet("/content/drive/MyDrive/yolov3.weights", "/content/drive/MyDrive/yolov3.cfg")
classes = ["person", "ball"]  # Adjust class names as needed
layer_names = net.getUnconnectedOutLayersNames()

# Load video
cap = cv2.VideoCapture("/content/drive/MyDrive/video.mp4")

# Define motion classes
motion_classes = ["Moving", "Jumping", "Rolling", "Sliding"]

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize prev_gray_roi for motion analysis
prev_gray_roi = None

# Desired output dimensions
output_height = 720
output_width = 1280

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Process detections
    detected_objects = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Adjust confidence threshold as needed
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                detected_objects.append((x, y, w, h, confidence))  # Include confidence

    # Apply non-maximum suppression manually
    indices = []
    for i in range(len(detected_objects)):
        x_i, y_i, w_i, h_i, confidence_i = detected_objects[i]
        keep = True

        for j in range(i + 1, len(detected_objects)):
            x_j, y_j, w_j, h_j, confidence_j = detected_objects[j]
            if iou((x_i, y_i, w_i, h_i), (x_j, y_j, w_j, h_j)) > 0.5:  # Adjust IoU threshold
                if confidence_i < confidence_j:
                    keep = False
                    break

        if keep:
            indices.append(i)


    # Analyze and classify motion for selected detected objects
    for i in indices:
        x, y, w, h, _ = detected_objects[i]
        motion_type = classify_motion(frame, x, y, w, h)

        # Draw bounding box and label
        color = (0, 255, 0)  # Green color
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{motion_type}: {classes[class_id]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Resize the frame to fit the desired output dimensions
    resized_frame = cv2.resize(frame, (output_width, output_height))

    # Display results
    cv2_imshow(resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()