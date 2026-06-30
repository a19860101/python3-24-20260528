from ultralytics import YOLO

custom_model = YOLO('../runs/detect/train-3/weights/best.pt')

results = custom_model.predict(source='./images/test2.JPEG', save=True, show=True,conf=0.5)

print("success")