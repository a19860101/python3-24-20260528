import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

plt.rc('font', family='Microsoft Jhenghei')


# 資料：散步時間（分鐘）
# sklearn 的 X 要用二維資料，所以每個數字都要包成 [數字]
X = [
    [5],
    [10],
    [20],
    [30],
    [40],
    [50],
    [60]
]

# 標籤：消耗熱量（大卡）
y = [20, 45, 110, 180, 250, 300, 330]

# 建立多項式特徵
# degree=2 代表建立 x 和 x²
poly = PolynomialFeatures(degree=3)

# 將原本的 X 轉換成多項式特徵
X_poly = poly.fit_transform(X)

# 建立線性回歸模型
model = LinearRegression()

# 用轉換後的資料訓練模型
model.fit(X_poly, y)

# 產生用來畫線的 x 範圍
X_line = []

for i in range(0, 71):
    X_line.append([i])

# 把畫線用的 x 也轉換成多項式特徵
X_line_poly = poly.transform(X_line)

# 預測曲線上的 y
y_line = model.predict(X_line_poly)

# 畫出原始資料點
# 因為 X 是 [[5], [10], [20]...]，畫圖時要取出裡面的數字
plt.scatter(
    [item[0] for item in X],
    y,
    label="資料點"
)

# 畫出多項式回歸曲線
plt.plot(
    [item[0] for item in X_line],
    y_line,
    label="多項式回歸曲線"
)

plt.xlabel("散步時間（分鐘）")
plt.ylabel("消耗熱量（大卡）")
plt.title("多項式回歸：散步時間與消耗熱量")
plt.legend()
plt.show()