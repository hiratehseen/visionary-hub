import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# download yolov8n.pt weights
# model = YOLO('yolov8n.pt')
# add video
cap = cv2.VideoCapture("Videos/people.mp4")  # For Video
# weights of yolo model
model = YOLO("Yolo-Weights/yolov8n.pt")
# classes
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
# read mask
mask = cv2.imread("E:\Python Projects\Object Detection\Project 2 - People Counter\mask.png")

"""
the tracker you're defining will keep track of objects based on detections or updates, but only if they have 
at least 3 hits, and will remove tracks that haven't been updated for 20 frames. Additionally, when associating 
a new detection with an existing track, a minimum overlap of 30% is required."""
# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# up and down lines to count people how manys are going up and down
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]
# count list
totalCountUp = []
totalCountDown = []

# loop
while True:
    # read image
    success, img = cap.read()
    # set video according to region
    imgRegion = cv2.bitwise_and(img, mask)
    # add some graphic image
    imgGraphics = cv2.imread("E:\Python Projects\Object Detection\Project 2 - People Counter\graphics.png", 
                             cv2.IMREAD_UNCHANGED)
    # on video it is set on middle of right side
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))
    # calculate results
    results = model(imgRegion, stream=True)
    # empty NumPy array with a shape of (0, 5)
    detections = np.empty((0, 5))
    # loop for results for detection
    for r in results:
        # set boxes
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            # width and height
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            # if person then detect
            if currentClass == "person" and conf > 0.3:
                # set array of values which detect by model
                currentArray = np.array([x1, y1, x2, y2, conf])
                # update on current detection
                detections = np.vstack((detections, currentArray))

    # update results
    resultsTracker = tracker.update(detections)
    # darw two lines for up and down
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    # again results loop for counter
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        # add circle on a object
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        # circle touch line the count
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
        # down circle touch line then count
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    # # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    # add text on count graphic
    cv2.putText(img,str(len(totalCountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
    cv2.putText(img,str(len(totalCountDown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)
    # video 
    cv2.imshow("Image", img)
    # mask
    # cv2.imshow("ImageRegion", imgRegion)
    # video play without keyboard interaction 
    cv2.waitKey(1)
