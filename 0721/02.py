# ==================================================
# 1. 匯入需要的套件
# ==================================================

# 載入 scikit-learn 內建乳癌資料集
from sklearn.datasets import load_breast_cancer

# 切分訓練資料與測試資料
from sklearn.model_selection import train_test_split

# 標準化
from sklearn.preprocessing import StandardScaler

# 建立 Pipeline
from sklearn.pipeline import make_pipeline

# 邏輯回歸
from sklearn.linear_model import LogisticRegression

# 模型評估
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ==================================================
# 2. 載入乳癌資料集
# ==================================================

cancer = load_breast_cancer()


# X：腫瘤特徵
X = cancer.data


# y：腫瘤分類答案
#
# 0：惡性 malignant
# 1：良性 benign
y = cancer.target


# ==================================================
# 3. 查看資料集基本資訊
# ==================================================

print("資料筆數與特徵數量：")
print(X.shape)

print("\n類別名稱：")
print(cancer.target_names)

print("\n特徵名稱：")
print(cancer.feature_names)

print("\n前 5 筆標籤：")
print(y[:5])


# ==================================================
# 4. 切分訓練資料與測試資料
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,

    # 20% 作為測試資料
    test_size=0.2,

    # 固定切分結果
    random_state=42,

    # 保持惡性與良性的類別比例
    stratify=y
)


# ==================================================
# 5. 建立模型
# ==================================================

# 執行順序：
# 1. StandardScaler 標準化
# 2. LogisticRegression 邏輯回歸分類

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000
    )
)


# ==================================================
# 6. 訓練模型
# ==================================================

model.fit(
    X_train,
    y_train
)


# ==================================================
# 7. 預測測試資料
# ==================================================

y_pred = model.predict(
    X_test
)


# predict_proba() 會取得兩種類別的機率
y_probability = model.predict_proba(
    X_test
)


# ==================================================
# 8. 計算正確率
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n正確率：")
print(accuracy)

print(
    f"正確率百分比：{accuracy:.2%}"
)


# ==================================================
# 9. 顯示混淆矩陣
# ==================================================

matrix = confusion_matrix(
    y_test,
    y_pred
)

print("\n混淆矩陣：")
print(matrix)


# 拆開混淆矩陣
tn, fp, fn, tp = matrix.ravel()

print("\n混淆矩陣詳細結果：")

print("TN：實際惡性，預測惡性：", tn)
print("FP：實際惡性，預測良性：", fp)
print("FN：實際良性，預測惡性：", fn)
print("TP：實際良性，預測良性：", tp)


# ==================================================
# 10. 顯示分類報告
# ==================================================

print("\n分類報告：")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "惡性",
            "良性"
        ],
        zero_division=0
    )
)


# ==================================================
# 11. 顯示前 10 筆測試結果
# ==================================================

print("\n前 10 筆測試結果：")

for index in range(10):

    real = y_test[index]
    pred = y_pred[index]

    # 類別 0 的機率：惡性機率
    malignant_probability = y_probability[index][0]

    # 類別 1 的機率：良性機率
    benign_probability = y_probability[index][1]

    real_text = "良性" if real == 1 else "惡性"
    pred_text = "良性" if pred == 1 else "惡性"

    result = "正確" if real == pred else "錯誤"

    print("-" * 50)

    print("真實答案：", real_text)
    print("模型預測：", pred_text)

    print(
        f"惡性機率："
        f"{malignant_probability:.2%}"
    )

    print(
        f"良性機率："
        f"{benign_probability:.2%}"
    )

    print("結果：", result)