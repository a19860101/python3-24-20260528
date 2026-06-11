import cv2
path = '../video.mp4'
cap = cv2.VideoCapture(path)
face_cascades = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascades.detectMultiScale(frame_gray, 1.1, 10, minSize=(0, 0))

    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi = frame[y:y+h, x:x+w]
        # region of interest

        # 高斯模糊
        # frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (55, 55), 0)

        # 美顏
        # frame[y:y + h, x:x + w] = cv2.bilateralFilter(roi, 5, 15, 15)

        # 漸層對應
        # frame[y:y + h, x:x + w] = cv2.applyColorMap(roi, cv2.COLORMAP_JET)

        # 馬賽克
        height,width = roi.shape[:2]
        small = cv2.resize(roi, (width//20, height//20))
        mosaic = cv2.resize(small, (width, height),interpolation=cv2.INTER_NEAREST)
        frame[y:y+h, x:x+w] = mosaic



        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()