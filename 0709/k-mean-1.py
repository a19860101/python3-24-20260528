import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 甜度、冰塊量
X = [
    [9, 8],
    [8, 9],
    [7, 8],
    [2, 2],
    [3, 1],
    [2, 3],
    [5, 5],
    [6, 4],
    [9, 9],
    [1, 2],
    [6, 5],
    [5, 4]
]

# 儲存不同 K 值的 inertia
inertias = []

# 測試 K = 1 到 K = 6
for k in range(1, 7):
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(X)
    inertias.append(model.inertia_)

# 畫出手肘圖
plt.plot(range(1, 7), inertias, marker='o')

plt.xlabel("K value")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()