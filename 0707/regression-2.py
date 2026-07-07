from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import math
X = [
    [10],
    [20],
    [30],
    [40],
    [50],
    [60]
]

y = [101,212,280,382,472,589]

model = LinearRegression()

model.fit(X, y)

m = model.coef_[0]
b = model.intercept_

print("線性回歸公式：")
print(f"y = {m:.3f}x + {b:.3f}")

y_pred = model.predict(X)
print(y_pred)

mae = []

# 誤差
print('-' * 50)
print(f'走路時間\t真實值\t預測值\t\t誤差絕對\t\t誤差平方誤差\t')
print('-' * 50)
for i in range(len(X)):
    walk = X[i][0]
    true_value = y[i]
    pred_value = y_pred[i]

    error = true_value - pred_value
    abs_error = abs(error)
    squared_error = abs_error**2
    mae.append(abs_error)

    print(
        f'{walk}\t\t'
        f'{true_value}\t\t'
        f'{pred_value}\t\t\t'
        f'{error}\t\t\t'
        f'{abs_error}\t\t\t'
        f'{squared_error}')

print(sum(mae) / len(mae))

MAE = mean_absolute_error(y, y_pred)
print(f'MAE:{MAE}')

MSE = mean_squared_error(y, y_pred)
print(f'MSE:{MSE}')

RMSE = math.sqrt(mean_squared_error(y, y_pred))
print(f'RMSE:{RMSE}')