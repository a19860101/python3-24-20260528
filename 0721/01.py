import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


X = np.array([
    [22, 20.1, 82, 0],
    [25, 21.5, 85, 0],
    [28, 23.0, 88, 0],
    [31, 24.5, 91, 0],
    [35, 25.2, 94, 1],
    [38, 27.5, 101, 0],
    [42, 28.1, 108, 1],
    [45, 29.3, 112, 0],
    [48, 30.2, 118, 1],
    [52, 31.5, 125, 1],
    [56, 29.8, 121, 0],
    [60, 32.4, 132, 1],
    [24, 22.2, 87, 1],
    [33, 26.1, 96, 0],
    [40, 27.8, 104, 1],
    [44, 28.9, 110, 1],
    [50, 30.7, 122, 0],
    [55, 31.2, 128, 1],
    [29, 23.8, 90, 0],
    [47, 29.5, 116, 1],
    [27, 24.1, 92, 0],
    [36, 26.8, 99, 1],
    [43, 28.4, 109, 0],
    [49, 30.1, 119, 1],
    [58, 32.0, 130, 1],
    [30, 22.9, 89, 1],
    [39, 27.1, 103, 0],
    [46, 29.6, 114, 1],
    [53, 31.0, 126, 0],
    [62, 33.1, 136, 1]
])

y = np.array([
    0, 0, 0, 0, 0,
    0, 1, 1, 1, 1,
    1, 1, 0, 0, 1,
    1, 1, 1, 0, 1,
    0, 0, 1, 1, 1,
    0, 0, 1, 1, 1
])


# 切分訓練資料與測試資料
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# 建立模型
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)


# 訓練模型
model.fit(X_train, y_train)


# 預測測試資料
y_pred = model.predict(X_test)


# 顯示結果
print("真實答案：", y_test)
print("模型預測：", y_pred)

print(
    "正確率：",
    accuracy_score(y_test, y_pred)
)

print("混淆矩陣：")
print(
    confusion_matrix(y_test, y_pred)
)

print("分類報告：")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "風險較低",
            "建議進一步檢查"
        ]
    )
)