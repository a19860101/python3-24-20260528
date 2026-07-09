import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

plt.rc('font', family='Microsoft Jhenghei')

# =========================
# 1. 準備資料
# =========================

# X：散步時間（分鐘）
# sklearn 的 X 要用「二維資料」
X = [
    [5],
    [10],
    [15],
    [20],
    [25],
    [30],
    [35],
    [40],
    [45],
    [50],
    [55],
    [60]
]

# y：消耗熱量（大卡）
y = [
    18,
    40,
    70,
    105,
    145,
    185,
    220,
    250,
    275,
    295,
    310,
    320
]

# =========================
# 2. 切分訓練資料與測試資料
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# =========================
# 3. 線性回歸
# =========================

linear_model = LinearRegression()

# 訓練線性回歸模型
linear_model.fit(X_train, y_train)

# 預測測試資料
linear_pred = linear_model.predict(X_test)

# =========================
# 4. 多項式回歸
# =========================

# degree=2 代表使用 x 和 x²
poly = PolynomialFeatures(degree=2)

# 將 X_train 轉換成多項式特徵
X_train_poly = poly.fit_transform(X_train)

# 將 X_test 轉換成多項式特徵
X_test_poly = poly.transform(X_test)

# 建立線性回歸模型
poly_model = LinearRegression()

# 用多項式特徵訓練模型
poly_model.fit(X_train_poly, y_train)

# 預測測試資料
poly_pred = poly_model.predict(X_test_poly)

# =========================
# 5. 評估模型
# =========================

# 線性回歸誤差
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_rmse = linear_mse ** 0.5

# 多項式回歸誤差
poly_mae = mean_absolute_error(y_test, poly_pred)
poly_mse = mean_squared_error(y_test, poly_pred)
poly_rmse = poly_mse ** 0.5

print("線性回歸結果")
print("MAE:", linear_mae)
print("MSE:", linear_mse)
print("RMSE:", linear_rmse)

print()

print("多項式回歸結果")
print("MAE:", poly_mae)
print("MSE:", poly_mse)
print("RMSE:", poly_rmse)

# =========================
# 6. 顯示測試資料預測結果
# =========================

print()
print("測試資料預測比較")
print("散步時間 | 真實熱量 | 線性預測 | 多項式預測")

for i in range(len(X_test)):
    print(
        f"{X_test[i][0]:>6} 分鐘 | "
        f"{y_test[i]:>6.0f} 大卡 | "
        f"{linear_pred[i]:>8.2f} | "
        f"{poly_pred[i]:>10.2f}"
    )

# =========================
# 7. 畫圖比較
# =========================

# 產生畫線用的資料
# 不使用 numpy.linspace，改用 range
X_line = []

for i in range(0, 71):
    X_line.append([i])

# 線性回歸預測線
y_linear_line = linear_model.predict(X_line)

# 多項式回歸預測線
X_line_poly = poly.transform(X_line)
y_poly_line = poly_model.predict(X_line_poly)

# 畫出原始資料點
plt.scatter(
    [item[0] for item in X],
    y,
    label="原始資料"
)

# 畫出線性回歸
plt.plot(
    [item[0] for item in X_line],
    y_linear_line,
    label="線性回歸"
)

# 畫出多項式回歸
plt.plot(
    [item[0] for item in X_line],
    y_poly_line,
    label="多項式回歸"
)

plt.xlabel("散步時間（分鐘）")
plt.ylabel("消耗熱量（大卡）")
plt.title("線性回歸 vs 多項式回歸")
plt.legend()
plt.show()