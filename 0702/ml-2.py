from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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

# 設定訓練與測試
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

# 訓練
model.fit(X_train,y_train)

# 預測
result = model.predict([[24,0]])

# 正確率
accuracy = accuracy_score(y_test, result)


print(result)
# print(accuracy)