# import libraies
import numpy as np
from ultralytics import YOLO
import cv2
# use for fancy designing
import cvzone
import math
# module for tracking
from sort import *

# video path
cap = cv2.VideoCapture("E:\Python Projects\Object Detection\Videos\cars.mp4")  # For Video
# yolov8n weights path
model = YOLO("E:\Python Projects\Object Detection\Yolo-Weights\yolov8n.pt")
# class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
# read mask image of certain region
mask = cv2.imread("E:\Python Projects\Object Detection\Project 1 - Car Counter\mask.png")
"""
the tracker you're defining will keep track of objects based on detections or updates, but only if they have 
at least 3 hits, and will remove tracks that haven't been updated for 20 frames. Additionally, when associating 
a new detection with an existing track, a minimum overlap of 30% is required."""
# Tracking 
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# a line draw on image to count vehicles
limits = [400, 297, 673, 297]
# list of vehicles count in video
totalCount = []

while True:
    '''success is a boolean variable that indicates whether the frame was successfully read from the video 
    source or not. The cap.read() function returns two values: success and img.'''
    success, img = cap.read()
    # resize video frames with mask region
    imgRegion = cv2.bitwise_and(img, mask)
    # overlay graphics on image, count vehicles 
    imgGraphics = cv2.imread("E:\Python Projects\Object Detection\Project 1 - Car Counter\graphics.png", 
                             cv2.IMREAD_UNCHANGED)
    # graphics image overlay on video with left corner (0,0)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    # returns a list of results, stream is used for generator and more efficient to use this as True
    results = model(imgRegion, stream=True)
    # empty array for detections
    detections = np.empty((0, 5))
    # results set with bounding boxes, confidence and class
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # for opencv
            # Bounding Box with x1, y1, x2, y2, 1st element of array  0
            x1, y1, x2, y2 = box.xyxy[0]
            # Output in tensor values, convert to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Draw rectangle with purple color and thickness 3
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            '''yolo detect on this draw rectangle with corner from cvzone, length of corner 9, roundness of 
            corner 2 and color purple'''
            # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            
            # Round a number upward to its nearest integer, Confidence set in 2 decimal places
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name with float value so, convert to int to get class name from classNames list
            cls = int(box.cls[0])
            # get class name from classNames list
            currentClass = classNames[cls]
            # if class name is car or truck or bus or motorbike and confidence is greater than 0.3 then detect
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                # detect vehicles
                currentArray = np.array([x1, y1, x2, y2, conf])
                # update detections with current
                detections = np.vstack((detections, currentArray))

    # update tracker with detections
    resultsTracker = tracker.update(detections)
    # draw red line on image with limits, thickness 5
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    # loop through results tracker
    for result in resultsTracker:
        # get values from result array
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        # find width and height
        w, h = x2 - x1, y2 - y1
        # draw rectangle with corner from cvzone, length of corner 9, roundness of corner 2 and color purple
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        # add text to image with class name and confidence value with scale 0.6, thickness 1 and offset 3
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        
        # find center of rectangle
        cx, cy = x1 + w // 2, y1 + h // 2
        # draw circle on object that detect on right region
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        # check count if already exist in a frame and assign id then not count
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            # check duplication if not duplicate then touch line its convert green
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

    # show image
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    # show image until press user input
    # show image in 1 millisecond
    # if 0 then frame forward by pressing key
    cv2.waitKey(0)
