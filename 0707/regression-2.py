from sklearn.linear_model import LinearRegression

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

# 誤差
for i in range(len(X)):
    walk = X[i][0]
    true_value = y[i]
    pred_value = y_pred[i]

    error = true_value - pred_value
    abs_error = abs(error)

    print(f'誤差:{error}')
    print(f'絕對誤差:{abs_error}')