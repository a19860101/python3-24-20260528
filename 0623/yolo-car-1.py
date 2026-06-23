from ultralytics import YOLO
import cv2
import easyocr

model = YOLO("best.pt")

reader = easyocr.Reader(['en'], gpu=True)

img = cv2.imread("../images/car-3.jpg")

results = model(img)

for result in results:
    print(result.boxes)


cv2.imshow("Result", img)
cv2.waitKey(0)