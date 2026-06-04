import cv2
import numpy as np

video_source = '../video.mp4'
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    bg = np.zeros(frame.shape, np.uint8)
    small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

    # width = int(cap.get(3))
    # height = int(cap.get(4))
    #
    # bg[:height // 2,:width // 2] = small_frame
    # bg[:height // 2,width // 2:] = cv2.flip(small_frame,1)
    # bg[height // 2:,:width // 2] = small_frame
    # bg[height // 2:,width // 2:] = small_frame

    # h,w = small_frame.shape[:2]
    h,w,c = small_frame.shape
    bg[:h,:w] = small_frame
    bg[:h,w:] = small_frame
    bg[h:,:w] = small_frame
    bg[h:,w:] = small_frame

    cv2.imshow('frame', bg)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()