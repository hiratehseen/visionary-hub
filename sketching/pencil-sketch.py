import cv2
import numpy as np

image = cv2.imread("egale.jpg")
# resize image
image = cv2.resize(image, (400, 400))
# convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# invert image
inverted_image = 255 - gray_image
# blur image
blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
# invert blurred image
inverted_blurred = 255 - blurred
# pencil sketch image
pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
# join images horizontally
pencil_sketch_horizontal = np.hstack((gray_image, pencil_sketch))
# display image
cv2.imshow("Original Image", pencil_sketch_horizontal)
cv2.waitKey(0)