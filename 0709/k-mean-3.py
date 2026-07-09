import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = [
    [0,0],
    [2,0],
    [2,2],
    [4,2],
    [8,7],
    [8,9],
    [10,7],
    [10,9],
]

# 使用手肘法判斷後，選擇 K = 3
model = KMeans(n_clusters=2)
model.fit(X)

labels = model.labels_
centers = model.cluster_centers_


print(labels)
print(centers)