# import libraies
import numpy as np
from ultralytics import YOLO
import cv2
# use for fancy designing
import cvzone
import math

# video path
cap = cv2.VideoCapture("E:\Python Projects\Object Detection\Videos\cars.mp4")  # For Video
# yolov8n weights path

model = YOLO("E:\Python Projects\Object Detection\Yolo-Weights\yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

user_input = input("Enter the class names you want (separated by commas): ")
requested_classes = user_input.split(",")

result = []
for class_name in requested_classes:
    class_name = class_name.strip()  # Remove leading/trailing spaces
    if class_name in classNames:
        result.append(class_name)

print("Requested class names:", result)