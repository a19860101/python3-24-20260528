import easyocr

reader = easyocr.Reader(['ch_tra','en'])
result = reader.readtext('img.png')

# print(result)
for item in result:
    print(item[1])