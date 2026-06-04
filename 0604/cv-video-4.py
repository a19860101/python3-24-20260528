import cv2
import numpy as np

video_source = '../video.mp4'
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # 邊緣檢測
    # frame = cv2.Canny(frame,100,10)

    #
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_TWILIGHT_SHIFTED)
    # cv2.COLORMAP_JET
    # cv2.COLORMAP_HOT
    # cv2.COLORMAP_SPRING
    # cv2.COLORMAP_SUMMER
    # cv2.COLORMAP_AUTUMN
    # cv2.COLORMAP_WINTER
    # cv2.COLORMAP_OCEAN
    # cv2.COLORMAP_TWILIGHT
    # cv2.COLORMAP_PINK
    # cv2.COLORMAP_BONE
    # cv2.COLORMAP_CIVIDIS
    # cv2.COLORMAP_COOL
    # cv2.COLORMAP_PARULA
    # cv2.COLORMAP_PLASMA
    # cv2.COLORMAP_DEEPGREEN
    # cv2.COLORMAP_RAINBOW
    # cv2.COLORMAP_HSV
    # cv2.COLORMAP_INFERNO
    # cv2.COLORMAP_MAGMA
    # cv2.COLORMAP_TWILIGHT_SHIFTED
    # cv2.COLORMAP_TURBO
    # cv2.COLORMAP_VIRIDIS

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()