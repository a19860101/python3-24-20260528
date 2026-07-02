from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

X = [
    [32,0],
    [30,1],
    [30,0],
    [28,0],
    [26,1],
    [25,1],
    [22,1],
    [20,0],
    [18,0],
    [16,0]
]
y = [
    '冰紅茶',
    '珍珠奶茶',
    '可樂',
    '無糖綠茶',
    '氣泡水',
    '牛奶',
    '水',
    '冰美式',
    '熱美式',
    '熱拿鐵'
]

# 建立模型
model = DecisionTreeClassifier()

scores = cross_val_score(model, X, y, cv=5)
print(scores)

# 訓練
model.fit(X,y)

# 預測
result = model.predict([[24,0]])
print(result)