from ultralytics import YOLO
import cv2
import easyocr

model = YOLO("best.pt")

reader = easyocr.Reader(['en'], gpu=True)

img = cv2.imread("../images/car-2.jpg")

H, W, _ = img.shape

results = model(img)

for result in results:
    # print(result.boxes)
    for box in result.boxes:
        # print(box.xyxy[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        print(x1,y1,x2,y2)

        padding = 5
        x1 = max(0,x1 - padding)
        y1 = max(0,y1 - padding)
        x2 = min(W,x2 + padding)
        y2 = min(H,y2 + padding)

        plate = img[y1:y2,x1:x2]

        text = reader.readtext(plate)
        print(text)

        if text:
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)


cv2.imshow("Result", img)
cv2.waitKey(0)
