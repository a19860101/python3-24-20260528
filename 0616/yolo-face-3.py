import cv2
from ultralytics import YOLO
import datetime

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('無法開啟攝影機........')
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = model(frame, stream=True,conf=0.4,classes=[0])

    person_detected = False
    for r in result:
        plot_frame = r.plot()
        # print(f'box:{r.boxes}')
        if len(r.boxes) > 0:
            person_detected = True

    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename= time_str+'.jpg'

    if person_detected:
        cv2.imwrite(filename,frame)

    cv2.imshow('YOLO v8', plot_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()