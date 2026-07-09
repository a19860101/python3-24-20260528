from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 加入 SVM 分類
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# X：天氣資料
# 第一個數字：氣溫
# 第二個數字：是否下雨，0代表沒下雨，1代表有下雨
X = [
    [32, 0], [31, 0], [30, 1], [29, 0],
    [27, 1], [26, 0], [25, 1], [24, 0],
    [22, 1], [21, 0], [20, 1], [19, 0],
    [18, 1], [17, 0], [16, 1], [15, 0]
]

# y：飲料類別
y = [
    "冰飲", "冰飲", "冰飲", "冰飲",
    "常溫", "常溫", "常溫", "常溫",
    "常溫", "常溫", "熱飲", "熱飲",
    "熱飲", "熱飲", "熱飲", "熱飲"
]


# 切分訓練資料與測試資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)


# 建立不同的分類模型
models = {
    "KNN分類": KNeighborsClassifier(n_neighbors=3),
    "決策樹分類": DecisionTreeClassifier(random_state=42),
    "隨機森林分類": RandomForestClassifier(random_state=42),

    # SVM 分類通常要搭配標準化
    "SVM分類 SVC": make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=10, gamma="scale")
    )
}


# 訓練與比較模型
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"模型：{name}")
    print(f"預測結果：{y_pred}")
    print(f"正確答案：{y_test}")
    print(f"Accuracy：{acc:.2f}")
    print("-" * 30)