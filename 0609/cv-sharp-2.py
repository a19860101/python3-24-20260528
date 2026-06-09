import cv2
import numpy as np

img = cv2.imread("../images/man.png")

blur = cv2.GaussianBlur(img, (15, 15), 0)
sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

sharp = cv2.Canny(sharp, 100, 200)

# alpha * 原圖 - beta * 模糊

cv2.imshow("s1", sharp)

cv2.waitKey(0)
cv2.destroyAllWindows()