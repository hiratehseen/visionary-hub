from ultralytics import YOLO
# import cv2

# # model = YOLO('yolov8n.pt') # dowloads the weights
# model = YOLO('../Yolo-Weights/yolov8n.pt')
# results = model("E:/Python Projects/Object Detection/Chapter 5 - Running Yolo/Images/3.png", show=True)
# cv2.waitKey(0)

yolo = YOLO()
yolo.download("yolov3")  # Replace "yolov3" with the desired YOLO version
