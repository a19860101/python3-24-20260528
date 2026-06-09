import cv2
import numpy as np

img = cv2.imread("../images/man.png")

kernel = np.array([
    [-1, 0, -1],
    [0, 5, 0],
    [-1, 0, -1]
])

kernel2 = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
kernel3 = np.array([
    [-1, -2, -1],
    [-2, 13, -2],
    [-1, -2, -1]
])

sharp_img = cv2.filter2D(img, -1, kernel)
sharp_img2 = cv2.filter2D(img, -1, kernel2)
sharp_img3 = cv2.filter2D(img, -1, kernel3)

cv2.imshow("s1", sharp_img)
cv2.imshow("s2", sharp_img2)
cv2.imshow("s3", sharp_img3)
cv2.waitKey(0)
cv2.destroyAllWindows()