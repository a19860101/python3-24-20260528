import cv2
import numpy as np

video_source = '../video.mp4'
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    bg = np.zeros(frame.shape, np.uint8)
    small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

    bg[:240, :320] = small_frame

    cv2.imshow('frame', bg)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()