import cv2

img = cv2.imread('../images/cat.jpg', cv2.IMREAD_REDUCED_COLOR_8)

img2 = cv2.rotate(img, cv2.ROTATE_180)
img3 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
img4 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

img_flip = cv2.flip(img, 1)

cv2.imshow('cat', img_flip)
# cv2.imshow('cat', img3)
# cv2.imshow('cat', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()