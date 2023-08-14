import cv2

# create video capture object
cap = cv2.VideoCapture('video.mp4')

# count the number of frames
fps = cap.get(cv2.CAP_PROP_FPS)
totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
durationInSeconds = totalNoFrames // fps

print("Video Duration In Seconds:", durationInSeconds, "s")