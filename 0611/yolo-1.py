import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# results = model('../images/man.png')
# results = model('../images/cat.jpg')
# print(results)
# results[0].show()


model.predict(source='../video.mp4',show=True, conf=0.8)
# model.predict(source=0, show=True)

