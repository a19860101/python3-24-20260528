import numpy as np
import cv2

img = np.zeros((300, 300, 3), dtype=np.uint8)

print(img[0][0])

img[0:][0:] = [128,255,0]
# BGR



cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
