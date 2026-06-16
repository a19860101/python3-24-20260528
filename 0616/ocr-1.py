import easyocr

reader = easyocr.Reader(['ch_tra','en'], gpu=True)
result = reader.readtext('img.png', paragraph=True)

# print(result)
for item in result:
    print(item[1])