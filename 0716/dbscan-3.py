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

# 核心點的索引
core_indices = model.core_sample_indices_

print("核心點索引：", core_indices)

for index in core_indices:
    print(
        f"核心點：{names[index]}，"
        f"位置：{X[index][0]}"
    )