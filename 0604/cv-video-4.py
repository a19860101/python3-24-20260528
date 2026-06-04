import cv2
import numpy as np

video_source = '../video.mp4'
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()


    frame = cv2.Canny(frame,100,10)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()