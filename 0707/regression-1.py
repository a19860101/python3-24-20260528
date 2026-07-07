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
