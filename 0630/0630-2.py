from ultralytics import YOLO

custom_model = YOLO('../runs/detect/train-4/weights/best.pt')

results = custom_model.predict(source='./images/test.JPEG', save=True, show=True,conf=0.5)

print("success")