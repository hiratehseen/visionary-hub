# import cvlib as cv
# import cv2
# import numpy as np

# webcam = cv2.VideoCapture(0)
    
# padding = 20
# while webcam.isOpened():
#     status, frame = webcam.read()
#     face, confidence = cv.detect_face(frame)
#     for idx, f in enumerate(face):        
#         (startX,startY) = max(0, f[0]-padding), max(0, f[1]-padding)
#         (endX,endY) = min(frame.shape[1]-1, f[2]+padding), min(frame.shape[0]-1, f[3]+padding)
#         cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
#         face_crop = np.copy(frame[startY:endY, startX:endX]) 
#         (label, confidence) = cv.detect_gender(face_crop)
#         idx = np.argmax(confidence)
#         label = label[idx]
#         label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
#         Y = startY - 10 if startY - 10 > 10 else startY + 10
#         cv2.putText(frame, label, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
#                     (0,255,0), 2)
#     cv2.imshow("Real-time gender detection", frame)
#     # press "s" to stop
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         break
# webcam.release()
# cv2.destroyAllWindows()

######################################################################################################
# download model
import urllib.request

# URL of the YOLOv3 weights file
weights_url = "https://pjreddie.com/media/files/yolov3.weights"

# Destination path to save the weights file
destination_path = "yolov3.weights"

# Download the YOLOv3 weights file
urllib.request.urlretrieve(weights_url, destination_path)

print("YOLOv3 weights downloaded successfully!")