import cv2

# 載入模型
# face_cascades = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_cascades = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascades = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img = cv2.imread("../images/man.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascades.detectMultiScale(img_gray, 1.1, 10, minSize=(0,0))

print(faces)

for (x,y,w,h) in faces:
    print(x,y,w,h)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()