import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


# ==================================================
# 1. 載入紅酒資料集
# ==================================================

wine = load_wine()

# X：13 個紅酒化學特徵
X = wine.data

# y：資料集原本提供的類別答案
# DBSCAN 訓練時不會使用，只留著最後比較
y = wine.target


# ==================================================
# 2. 建立中文欄位名稱
# ==================================================

chinese_columns = [
    "酒精濃度",
    "蘋果酸",
    "灰分",
    "灰分鹼度",
    "鎂含量",
    "總酚含量",
    "類黃酮",
    "非類黃酮酚",
    "原花青素",
    "顏色強度",
    "色調",
    "光學吸收比",
    "脯胺酸"
]


# 將資料轉成 DataFrame
df = pd.DataFrame(
    X,
    columns=chinese_columns
)

# 加入原始類別
# 這只是資料集原本的答案，不會拿來訓練 DBSCAN
df["原始類別"] = y

print("前 5 筆原始資料：")
print(df.head())

print("\n資料形狀：")
print(df.shape)


# ==================================================
# 3. 只取特徵資料
# ==================================================

# 不可以把「原始類別」放進 DBSCAN
X_features = df[chinese_columns]


# ==================================================
# 4. 標準化資料
# ==================================================

# 紅酒的不同特徵數值範圍差很多
# 例如：
# 酒精濃度大約是十幾
# 脯胺酸可能是數百到上千
#
# DBSCAN 依賴距離，所以需要先標準化
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_features)

print("\n標準化後前 5 筆資料：")
print(X_scaled[:5])


# ==================================================
# 5. 建立 DBSCAN 模型
# ==================================================

model = DBSCAN(
    eps=2.2,          # 搜尋鄰居的距離範圍
    min_samples=4     # 至少需要 5 個點才能成為核心點
)


# ==================================================
# 6. 執行分群
# ==================================================

labels = model.fit_predict(X_scaled)

# 將 DBSCAN 分群結果放回 DataFrame
df["DBSCAN群組"] = labels


# ==================================================
# 7. 判斷核心點、邊界點、雜訊點
# ==================================================

# core_sample_indices_ 會記錄核心點的資料索引
core_indices = set(model.core_sample_indices_)

point_types = []

for index in range(len(df)):

    # DBSCAN 標籤為 -1，代表雜訊點
    if labels[index] == -1:
        point_types.append("雜訊點")

    # 索引位於核心點索引中
    elif index in core_indices:
        point_types.append("核心點")

    # 已經屬於某一群，但不是核心點
    else:
        point_types.append("邊界點")


df["點的類型"] = point_types


# ==================================================
# 8. 計算群組數量
# ==================================================

# set(labels) 可能得到：
# {-1, 0, 1, 2}
#
# -1 是雜訊，不算一個群組
cluster_count = len(set(labels))

if -1 in labels:
    cluster_count -= 1


# 計算雜訊點數量
noise_count = np.sum(labels == -1)

# 計算核心點和邊界點數量
core_count = point_types.count("核心點")
border_count = point_types.count("邊界點")


print("\n================ DBSCAN 結果 ================")

print("群組數量：", cluster_count)
print("核心點數量：", core_count)
print("邊界點數量：", border_count)
print("雜訊點數量：", noise_count)


print("\n各群組資料數量：")
print(
    df["DBSCAN群組"]
    .value_counts()
    .sort_index()
)


print("\n各類型資料數量：")
print(
    df["點的類型"]
    .value_counts()
)


# ==================================================
# 9. 顯示部分分群結果
# ==================================================

print("\n前 20 筆分群結果：")

print(
    df[
        [
            "酒精濃度",
            "蘋果酸",
            "顏色強度",
            "脯胺酸",
            "原始類別",
            "DBSCAN群組",
            "點的類型"
        ]
    ].head(20)
)


# ==================================================
# 10. PCA 降成二維
# ==================================================

# 原始紅酒資料有 13 個特徵，無法直接畫在平面上
# 使用 PCA 將 13 維降成 2 維，方便畫圖
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# 將 PCA 結果加入 DataFrame
df["主成分1"] = X_pca[:, 0]
df["主成分2"] = X_pca[:, 1]


print("\nPCA 解釋變異比例：")
print(pca.explained_variance_ratio_)

print(
    "前兩個主成分合計保留：",
    pca.explained_variance_ratio_.sum()
)


# ==================================================
# 11. 建立不同類型資料的遮罩
# ==================================================

core_mask = df["點的類型"] == "核心點"
border_mask = df["點的類型"] == "邊界點"
noise_mask = df["點的類型"] == "雜訊點"


# ==================================================
# 12. 畫出 DBSCAN 分群結果
# ==================================================

plt.figure(figsize=(10, 7))


# 畫核心點
plt.scatter(
    df.loc[core_mask, "主成分1"],
    df.loc[core_mask, "主成分2"],
    c=df.loc[core_mask, "DBSCAN群組"],
    marker="o",
    s=80,
    label="Core Point"
)


# 畫邊界點
plt.scatter(
    df.loc[border_mask, "主成分1"],
    df.loc[border_mask, "主成分2"],
    c=df.loc[border_mask, "DBSCAN群組"],
    marker="^",
    s=90,
    label="Border Point"
)


# 畫雜訊點
plt.scatter(
    df.loc[noise_mask, "主成分1"],
    df.loc[noise_mask, "主成分2"],
    marker="x",
    s=100,
    label="Noise Point"
)


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Wine Clustering")
plt.legend()
plt.grid()
plt.show()


# ==================================================
# 13. 畫出原始類別
# ==================================================

# 這張圖不是 DBSCAN 的結果
# 而是資料集原本提供的三種類別
# 用來和 DBSCAN 分群結果比較
plt.figure(figsize=(10, 7))

plt.scatter(
    df["主成分1"],
    df["主成分2"],
    c=df["原始類別"],
    s=70
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Original Wine Classes")
plt.grid()
plt.show()


# ==================================================
# 14. 建立交叉表比較
# ==================================================

# 比較原始類別和 DBSCAN 群組
comparison_table = pd.crosstab(
    df["原始類別"],
    df["DBSCAN群組"],
    rownames=["原始類別"],
    colnames=["DBSCAN群組"]
)

print("\n原始類別與 DBSCAN 群組比較：")
print(comparison_table)