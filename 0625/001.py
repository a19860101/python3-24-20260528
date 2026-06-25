from ultralytics import YOLO

# model = YOLO('yolov8n.pt')
model = YOLO('../runs/detect/train-11/weights/best.pt')
results = model.train(data='./dataset/data.yaml', epochs=100, imgsz=512, device='cpu')




