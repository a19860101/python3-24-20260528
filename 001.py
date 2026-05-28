import sys

import cv2
import sys

img = cv2.imread('cat.jpg')

if img is None:
    sys.exit()
print(img)
print(img.shape)
print(type(img))