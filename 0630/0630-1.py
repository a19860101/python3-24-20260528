from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data='./0630_dataset/data.yaml', epochs=10, imgsz=512, device=0, workers=0)