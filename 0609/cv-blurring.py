import cv2

img = cv2.imread('../images/man.png')

# img = cv2.blur(img, (30, 30))

# img = cv2.GaussianBlur(img, (111, 111), 0)

# img = cv2.medianBlur(img, 51)

# img = cv2.bilateralFilter(img, 11, 17, 17)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

