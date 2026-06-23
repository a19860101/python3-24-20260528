from ultralytics import YOLO
import cv2
import easyocr

model = YOLO("best.pt")

reader = easyocr.Reader(['en'], gpu=True)

img = cv2.imread("../images/car-3.jpg")

H, W, _ = img.shape

results = model(img)
allow_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'

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

        # 放大圖片
        plate_resized = cv2.resize(plate,None,fx=2,fy=2, interpolation=cv2.INTER_CUBIC)
        # 轉灰階
        plate_gray = cv2.cvtColor(plate_resized,cv2.COLOR_BGR2GRAY)
        # 雙邊濾波
        plate_blur = cv2.bilateralFilter(plate_gray,15,15,15)

        #銳利化
        # plate_blur = cv2.GaussianBlur(plate_gray,(9,9),0)
        # plate_sharpened = cv2.addWeighted(plate_blur,1.5,plate_gray,-0.5,0)

        # 臨界值 / 二值化
        binary_plate = cv2.adaptiveThreshold(
            plate_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            5
        )

        cv2.imshow("plate",binary_plate)
        cv2.waitKey(1)

        text = reader.readtext(plate_blur)
        print(text)

        if text:
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)


cv2.imshow("Result", img)
cv2.waitKey(0)
