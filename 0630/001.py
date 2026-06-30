# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')
# results = model.train(data='./dataset/data.yaml', epochs=100, imgsz=512, device='cpu')

from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data='./dataset/data.yaml', epochs=500, imgsz=512, device=0, workers=0)