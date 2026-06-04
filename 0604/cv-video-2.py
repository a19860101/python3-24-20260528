import cv2

video_source = '../video.mp4'
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    flip_frame = cv2.flip(small_frame, 1)
    rotate_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


    cv2.imshow('frame', rotate_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()