import cv2

img = cv2.imread('../images/cat.jpg', cv2.IMREAD_REDUCED_COLOR_4)
# 1,cv2.IMREAD_COLOR 載入一般圖片，忽略透明
# 0,cv2.IMREAD_GRAYSCALE 載入灰階圖片
# -1,cv2.IMREAD_UNCHANGED 載入包含alpha的圖片

#其他參數
# cv2.IMREAD_REDUCED_COLOR_2：讀取彩色，長寬變為原本的 1/2。
# cv2.IMREAD_REDUCED_COLOR_4：讀取彩色，長寬變為原本的 1/4。
# cv2.IMREAD_REDUCED_COLOR_8：讀取彩色，長寬變為原本的 1/8。

# cv2.IMREAD_REDUCED_GRAYSCALE_2：讀取並轉灰階，長寬變為原本的 1/2。
# cv2.IMREAD_REDUCED_GRAYSCALE_4：讀取並轉灰階，長寬變為原本的 1/4。
# cv2.IMREAD_REDUCED_GRAYSCALE_8：讀取並轉灰階，長寬變為原本的 1/8。


cv2.imshow('cat', img)

cv2.imwrite('cat_1.jpg', img)
cv2.imwrite('cat_2.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])
cv2.imwrite('cat_3.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 1])

cv2.imwrite('cat_1.png', img)
cv2.imwrite('cat_2.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite('cat_3.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

cv2.imwrite('cat_1.webp', img)
cv2.imwrite('cat_2.webp', img, [cv2.IMWRITE_WEBP_QUALITY, 100])
cv2.imwrite('cat_3.webp', img, [cv2.IMWRITE_WEBP_QUALITY, 1])
cv2.imwrite('cat_4.webp', img, [cv2.IMWRITE_WEBP_QUALITY, 101])


# JPEG格式
# 參數常數：cv2.IMWRITE_JPEG_QUALITY
# 數值範圍：0 到 100（數字越大畫質越好，檔案越大）。
# 預設值：95。

# PNG格式
# 參數常數：cv2.IMWRITE_PNG_COMPRESSION
# 數值範圍：0 到 9（數字越大檔案越小，但存檔花費的時間越久）。
# 預設值：3。

# WEBP
# 參數常數：cv2.IMWRITE_WEBP_QUALITY
# 數值範圍：1 到 100有損壓縮，101以上無損壓縮
# 預設值：無損壓縮。


cv2.waitKey(0)
# 0表示程式會停在這裡，直到按下鍵盤上的任意鍵，才會繼續執行下一行。
# 100則為100毫秒，表示視窗會停留100毫秒（0.1秒）

cv2.destroyAllWindows()