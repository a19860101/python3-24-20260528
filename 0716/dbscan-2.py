import numpy as np
from sklearn.cluster import DBSCAN

names = ["A", "B", "C", "D", "E", "F", "G", "H"]

X = np.array([
    [1.0],
    [1.5],
    [2.0],
    [2.5],
    [5.0],
    [5.5],
    [6.0],
    [9.0]
])

model = DBSCAN(
    eps=0.6,
    min_samples=3
)

labels = model.fit_predict(X)

for name, data, label in zip(names, X, labels):
    if label == -1:
        result = "雜訊點"
    else:
        result = f"群組 {label}"

    print(f"{name}：位置 {data[0]}，結果：{result}")