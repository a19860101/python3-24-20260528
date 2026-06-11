import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('無法開啟攝影機........')
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = model(frame, stream=True)

    for r in result:
        plot_frame = r.plot()

    cv2.imshow('YOLO v8', plot_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()