import cv2

# 載入模型
# face_cascades = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascades = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread("../images/man.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascades.detectMultiScale(img_gray, 1.3, 5)

print(faces)