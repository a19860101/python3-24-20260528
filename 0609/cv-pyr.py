import cv2

img = cv2.imread("../images/man.png")

small = cv2.pyrDown(img)
small_img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

large = cv2.pyrUp(img)
large_img = cv2.resize(img, (0,0), fx=2, fy=2)

cv2.imshow("origin", img)

cv2.imshow("small", small)
cv2.imshow("small_img", small_img)

cv2.imshow("large", large)
cv2.imshow("large_img", large_img)

cv2.waitKey(0)
cv2.destroyAllWindows()