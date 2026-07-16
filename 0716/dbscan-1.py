import numpy as np
from sklearn.cluster import DBSCAN

# 一維資料
# 每一筆資料仍然要寫成 [特徵] 的形式
X = np.array([
    [1.0],   # A
    [1.5],   # B
    [2.0],   # C
    [2.5],   # D
    [5.0],   # E
    [5.5],   # F
    [6.0],   # G
    [9.0]    # H
])

# 建立 DBSCAN 模型
model = DBSCAN(
    eps=0.6,
    min_samples=3
)

# 執行分群
labels = model.fit_predict(X)

print(labels)