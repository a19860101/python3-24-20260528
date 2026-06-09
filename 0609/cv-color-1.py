import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # H:hue色相,S:saturation飽和度,V:value明度
    # BGR
    lower = np.array([80,50,50])
    upper = np.array([130,255,255])

    mask = cv2.inRange(hsv, lower, upper)

    cv2.imshow("original", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()