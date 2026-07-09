import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

# 使用手肘法判斷後，選擇 K = 3
model = KMeans(n_clusters=3, random_state=0)
model.fit(X)

labels = model.labels_
centers = model.cluster_centers_

print(labels)
print(centers)

for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], c=f"C{labels[i]}", s=100)
    plt.text(X[i][0] + 0.1, X[i][1] + 0.1, str(i))

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="black",
    marker="X",
    s=200,
    label="Center"
)

plt.xlabel("Sweetness")
plt.ylabel("Ice Level")
plt.title("K-means Clustering Result")
plt.legend()
plt.show()