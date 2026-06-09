import cv2

# 均值濾波
# img = cv2.blur(img, (30, 30))

# 高斯模糊
# img = cv2.GaussianBlur(img, (111, 111), 0)

# 中值濾波（消除胡椒鹽雜訊）
# img_noise = cv2.medianBlur(img_noise, 3)

# 雙邊濾波（美肌）
# img1 = cv2.bilateralFilter(img, 1, 20, 20)
# img2 = cv2.bilateralFilter(img, 10, 50, 20)

# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)

# cv2.imshow('img_noise', img_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()

