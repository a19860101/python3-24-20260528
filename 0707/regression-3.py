from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np


# =========================
# 1. 準備資料
# =========================

# X 是特徵：散步時間
# sklearn 的 X 要是二維資料
X = [
    [10],
    [20],
    [30],
    [40],
    [50],
    [60],
    [70],
    [80],
    [90],
    [100]
]

# y 是標籤：消耗熱量
y = [42, 82, 121, 160, 198, 240, 278, 322, 358, 399]


# =========================
# 2. 切分訓練資料與測試資料
# =========================

# test_size=0.3 代表 30% 當測試資料
# random_state=0 代表每次切分結果固定，方便教學展示
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)


# =========================
# 3. 建立多個回歸模型
# =========================

models = {
    "線性回歸": LinearRegression(),

    # 決策樹回歸
    # random_state=0 是為了讓結果固定
    "決策樹回歸": DecisionTreeRegressor(random_state=0),

    # KNN 回歸
    # n_neighbors=2 代表參考最近的 2 筆資料取平均
    "KNN 回歸": KNeighborsRegressor(n_neighbors=2),

    # 隨機森林回歸
    # n_estimators=100 代表建立 100 棵決策樹
    "隨機森林回歸": RandomForestRegressor(
        n_estimators=100,
        random_state=0
    )
}


# =========================
# 4. 訓練、預測、計算誤差
# =========================

results = []

for model_name, model in models.items():

    # 訓練模型
    model.fit(X_train, y_train)

    # 使用測試資料進行預測
    y_pred = model.predict(X_test)

    # MAE：平均絕對誤差
    mae = mean_absolute_error(y_test, y_pred)

    # MSE：平均平方誤差
    mse = mean_squared_error(y_test, y_pred)

    # RMSE：均方根誤差
    rmse = np.sqrt(mse)

    # 將結果存起來，等一下統一比較
    results.append({
        "模型": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

    # 顯示每個模型的預測狀況
    print("=" * 50)
    print(model_name)
    print("=" * 50)

    print("測試資料 X：", X_test)
    print("真實值 y_test：", y_test)
    print("預測值 y_pred：", y_pred)

    print(f"MAE  平均絕對誤差：{mae:.3f}")
    print(f"MSE  平均平方誤差：{mse:.3f}")
    print(f"RMSE 均方根誤差：{rmse:.3f}")
    print()


# =========================
# 5. 統一比較所有模型
# =========================

print("=" * 50)
print("模型比較結果")
print("=" * 50)

print("模型\t\tMAE\t\tMSE\t\tRMSE")
print("-" * 50)

for result in results:
    print(
        f"{result['模型']}\t"
        f"{result['MAE']:.3f}\t\t"
        f"{result['MSE']:.3f}\t\t"
        f"{result['RMSE']:.3f}"
    )


# =========================
# 6. 找出 MAE 最小的模型
# =========================

best_model = min(results, key=lambda x: x["MAE"])

print()
print("MAE 最小的模型：")
print(best_model["模型"])