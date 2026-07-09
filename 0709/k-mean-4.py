import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = [
    [0, 0],
    [2, 0],
    [2, 2],
    [4, 2],
    [8, 7],
    [8, 9],
    [10, 7],
    [10, 9],
]

# 儲存不同 K 值的 inertia
inertias = []

# 測試 K = 1 到 K = 6
for k in range(1, 6):
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(X)
    inertias.append(model.inertia_)

print(inertias)

# 畫出手肘圖
# plt.plot(range(1, 7), inertias, marker='o')
#
# plt.xlabel("K value")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# plt.show()