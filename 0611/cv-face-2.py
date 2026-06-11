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
        # range of interest
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (55, 55), 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()