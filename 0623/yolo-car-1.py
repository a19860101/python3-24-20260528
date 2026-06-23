from ultralytics import YOLO
import cv2
import easyocr

model = YOLO("best.pt")

reader = easyocr.Reader(['en'], gpu=True)

img = cv2.imread("../images/car-3.jpg")

results = model(img)

for result in results:
    # print(result.boxes)
    for box in result.boxes:
        # print(box.xyxy[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        print(x1,y1,x2,y2)
cv2.imshow("Result", img)
cv2.waitKey(0)
