from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

X = [
    [32,0],[30,1],[30,0],[28,0],[26,1],[25,1],[22,1],[20,0],[18,0],[16,0],
    [32,1],[30,1],[30,0],[28,0],[25,1],[25,1],[22,0],[20,1],[19,0],[16,0],
    [32,0],[30,1],[30,0],[28,1],[26,1],[25,0],[22,1],[20,0],[17,0],[16,1]
]
y = [
    '冰飲','冰飲','冰飲','冰飲','冰飲','冰飲','常溫','熱飲','熱飲','熱飲',
    '冰飲','常溫','冰飲','冰飲','冰飲','冰飲','常溫','常溫','熱飲','熱飲',
    '冰飲','冰飲','冰飲','熱飲','冰飲','冰飲','熱飲','常溫','熱飲','冰飲'
]

# 建立模型
# model = DecisionTreeClassifier()
# model = KNeighborsClassifier(n_neighbors=3)
model = RandomForestClassifier(n_estimators=1000)

# 設定訓練與測試
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

# 訓練
model.fit(X_train,y_train)

# 預測
result = model.predict(X_test)

# 正確率
accuracy = accuracy_score(y_test, result)


print(f'預測答案：{result}')
print(f'答案：{y_test}')
print(f'正確率: {accuracy:.2%}')