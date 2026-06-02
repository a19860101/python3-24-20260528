import cv2

img = cv2.imread('../images/cat.jpg', cv2.IMREAD_REDUCED_COLOR_8)

cv2.line(img, (0,0), (100,100), (0,0,255), 5)

cv2.rectangle(img, (0,0), (100,100), (255,0,255), 5)
cv2.rectangle(img, (100,100), (200,200), (0,128,255), -1)

cv2.circle(img, (200,200), 20, (123,0,255), -1)
cv2.circle(img, (400,20), 40, (123,0,255), 2)


cv2.imshow('cat', img)
cv2.waitKey(0)
cv2.destroyAllWindows()