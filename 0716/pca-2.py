import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data = {
    "顧客編號": [
        "C001", "C002", "C003", "C004", "C005",
        "C006", "C007", "C008", "C009", "C010",
        "C011", "C012", "C013", "C014", "C015",
        "C016", "C017", "C018", "C019", "C020",
        "C021", "C022", "C023", "C024", "C025",
        "C026", "C027", "C028", "C029", "C030"
    ],

    # 每月到店消費次數
    "每月到店次數": [
        2, 4, 3, 5, 2,
        6, 4, 3, 7, 5,
        9, 11, 8, 12, 10,
        14, 9, 13, 11, 15,
        18, 22, 20, 25, 17,
        24, 19, 27, 21, 23
    ],

    # 每次平均消費金額，單位為元
    "平均消費金額": [
        85, 120, 95, 150, 110,
        180, 135, 160, 210, 175,
        250, 320, 230, 350, 280,
        420, 300, 380, 340, 450,
        520, 680, 590, 750, 480,
        720, 550, 820, 650, 700
    ],

    # 每次平均停留時間，單位為分鐘
    "平均停留時間": [
        12, 18, 15, 25, 10,
        30, 22, 35, 28, 20,
        45, 60, 38, 75, 50,
        90, 55, 80, 65, 100,
        120, 150, 135, 180, 110,
        165, 125, 200, 145, 170
    ],

    # 每月使用優惠券次數
    "優惠券使用次數": [
        0, 1, 0, 2, 1,
        2, 1, 1, 3, 2,
        3, 4, 2, 5, 3,
        5, 3, 4, 4, 6,
        7, 9, 8, 10, 6,
        9, 7, 11, 8, 10
    ]
}

df = pd.DataFrame(data)
# ==================================================
# 1. 選擇特徵
# ==================================================

feature_columns = [
    "每月到店次數",
    "平均消費金額",
    "平均停留時間",
    "優惠券使用次數"
]

X = df[feature_columns]


# ==================================================
# 2. 資料標準化
# ==================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==================================================
# 3. 比較不同群數的輪廓係數
# ==================================================

cluster_numbers = []
silhouette_scores = []


for k in range(2, 8):

    test_model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    test_labels = test_model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        test_labels
    )

    cluster_numbers.append(k)
    silhouette_scores.append(score)

    print(f"{k} 群的輪廓係數：{score:.3f}")


# ==================================================
# 4. 找出輪廓係數最高的群數
# ==================================================

best_index = silhouette_scores.index(
    max(silhouette_scores)
)

best_k = cluster_numbers[best_index]

best_score = silhouette_scores[best_index]

print()
print("建議群數：", best_k)
print("最高輪廓係數：", round(best_score, 3))


# ==================================================
# 5. 使用最佳群數重新分群
# ==================================================

model = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward"
)

df["群組"] = model.fit_predict(X_scaled)


# ==================================================
# 6. PCA：將四維降成二維
# ==================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df["主成分1"] = X_pca[:, 0]
df["主成分2"] = X_pca[:, 1]


# ==================================================
# 7. 顯示 PCA 保留資訊比例
# ==================================================

print()
print("各主成分解釋比例：")
print(pca.explained_variance_ratio_)

print(
    f"兩個主成分共保留："
    f"{pca.explained_variance_ratio_.sum():.2%}"
)


# ==================================================
# 8. 畫出輪廓係數折線圖
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    cluster_numbers,
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Comparison")

plt.xticks(cluster_numbers)
plt.grid()

plt.show()


# ==================================================
# 9. 畫出 PCA 分群圖
# ==================================================

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    df["主成分1"],
    df["主成分2"],
    c=df["群組"],
    s=80
)

for i in range(len(df)):

    plt.text(
        df.loc[i, "主成分1"] + 0.05,
        df.loc[i, "主成分2"] + 0.05,
        df.loc[i, "顧客編號"],
        fontsize=8
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Customer Clustering")

plt.colorbar(scatter, label="Cluster")
plt.grid()

plt.show()


# ==================================================
# 10. 查看每群人數
# ==================================================

print()
print("每群顧客人數：")

print(
    df.groupby("群組")["顧客編號"].count()
)


# ==================================================
# 11. 查看各群原始特徵平均值
# ==================================================

print()
print("各群平均特徵：")

print(
    df.groupby("群組")[feature_columns].mean().round(2)
)