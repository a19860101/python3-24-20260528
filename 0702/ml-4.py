from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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

# 選擇模型
print("請選擇模型：")
print("1) 決策樹 Decision Tree")
print("2) KNN")
print("3) 隨機森林 Random Forest")

choice = input("請輸入 1 / 2 / 3：")

if choice == "1":
    model = DecisionTreeClassifier(random_state=42)
    model_name = "決策樹"

elif choice == "2":
    model = KNeighborsClassifier(n_neighbors=3)
    model_name = "KNN"

elif choice == "3":
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model_name = "隨機森林"

else:
    print("輸入錯誤，預設使用 KNN")
    model = KNeighborsClassifier(n_neighbors=3)
    model_name = "KNN"


print(f"\n你選擇的模型是：{model_name}")


# =========================
# 方法一：train / test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 訓練模型
model.fit(X_train, y_train)

# 預測測試資料
result = model.predict(X_test)

# 計算正確率
accuracy = accuracy_score(y_test, result)

print("\n========== train / test 測試結果 ==========")
print(f"預測答案：{result}")
print(f"正確答案：{y_test}")
print(f"正確率：{accuracy:.2%}")


# =========================
# 方法二：交叉驗證
# =========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\n========== 交叉驗證結果 ==========")
print(f"每一次驗證的正確率：{scores}")
print(f"平均正確率：{scores.mean():.2%}")
print(f"正確率標準差：{scores.std():.2%}")


# =========================
# 使用全部資料重新訓練
# =========================

model.fit(X, y)

# 預測新的資料
# 格式：[溫度, 是否下雨]
# 0 = 沒下雨
# 1 = 有下雨

# new_data = [[27, 0]]

d1 = int(input('請輸入氣溫'))
d2 = int(input('請輸入是否下雨1/0'))
new_data = [[d1, d2]]

new_result = model.predict(new_data)

print("\n========== 新資料預測 ==========")
print(f"新的資料：{new_data}")
print(f"預測結果：{new_result[0]}")