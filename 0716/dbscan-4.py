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

# 轉成 set，方便判斷某個索引是不是核心點
core_indices = set(model.core_sample_indices_)

for index, name in enumerate(names):

    if labels[index] == -1:
        point_type = "雜訊點"

    elif index in core_indices:
        point_type = "核心點"

    else:
        point_type = "邊界點"

    print(
        f"{name}：位置={X[index][0]}，"
        f"類型={point_type}，"
        f"群組={labels[index]}"
    )