import cv2
import numpy as np

img = np.zeros((300,300,3), dtype=np.uint8)

cv2.imwrite("test.png", img)

print("OpenCV working")