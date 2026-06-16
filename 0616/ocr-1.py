import easyocr

reader = easyocr.Reader(['ch_tra','en'], gpu=True)
result = reader.readtext('img.png', paragraph=True,detail=0)

# print(result)
with open('output.txt','w',encoding='utf-8') as f:
    for item in result:
        f.write(item)