from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# 1. 載入資料
iris = load_iris()
X = iris.data

# 2. 標準化
X_scaled = StandardScaler().fit_transform(X)

# 3. 階層式分群
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

cluster_labels = model.fit_predict(X_scaled)

print(cluster_labels)