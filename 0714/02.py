from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

X = [
    [20, 300000],
    [25, 400000],
    [45, 1200000],
    [50, 1500000]
]

# 先標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)

# 再進行階層式分群
result = linkage(X_scaled, method="ward")

dendrogram(result)
plt.ylabel("Distance")
plt.show()