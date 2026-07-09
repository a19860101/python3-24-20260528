from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# 加入 SVM 回歸
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# X：特徵資料
# 這裡代表「散步時間」，單位是分鐘
# 注意：sklearn 的 X 通常要寫成二維資料，所以是 [[5], [10], [15]]
X = [
    [5], [10], [15], [20], [25], [30],
    [35], [40], [45], [50], [55], [60]
]

# y：答案資料
# 這裡代表「消耗熱量」，單位是大卡
y = [
    22, 45, 65, 90, 105, 135,
    150, 175, 190, 220, 240, 260
]


# 將資料切成訓練資料與測試資料
# test_size=0.3 代表 30% 資料拿來測試
# random_state=42 代表固定隨機結果，方便每次執行結果一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)


# 建立多個回歸模型
models = {
    # 線性回歸：找一條直線預測熱量
    "線性回歸": LinearRegression(),

    # KNN回歸：看附近幾筆資料的平均結果
    "KNN回歸": KNeighborsRegressor(n_neighbors=3),

    # 決策樹回歸：用一連串條件切分資料
    "決策樹回歸": DecisionTreeRegressor(random_state=42),

    # 隨機森林回歸：很多棵決策樹一起預測，最後取平均
    "隨機森林回歸": RandomForestRegressor(random_state=42),

    # SVM回歸：也叫 SVR
    # StandardScaler()：先把資料標準化
    # SVR()：使用支援向量機做回歸預測
    "SVM回歸 SVR": make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=100, epsilon=10)
    )
}


# 依序訓練每一個模型，並計算誤差
for name, model in models.items():

    # 用訓練資料讓模型學習
    model.fit(X_train, y_train)

    # 用測試資料進行預測
    y_pred = model.predict(X_test)

    # MAE：平均絕對誤差
    # 數值越小，代表預測越接近正確答案
    mae = mean_absolute_error(y_test, y_pred)

    # MSE：平均平方誤差
    # 會放大較大的錯誤
    mse = mean_squared_error(y_test, y_pred)

    # RMSE：MSE 開根號
    # 不使用 numpy，所以改用 ** 0.5
    rmse = mse ** 0.5

    print(f"模型：{name}")
    print(f"預測結果：{y_pred}")
    print(f"正確答案：{y_test}")
    print(f"MAE：{mae:.2f}")
    print(f"MSE：{mse:.2f}")
    print(f"RMSE：{rmse:.2f}")
    print("-" * 30)