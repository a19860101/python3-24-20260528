import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram


# ==================================================
# 1. 建立資料
# ==================================================

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

print("原始資料：")
print(df)


# ==================================================
# 2. 取出要分群的特徵
# ==================================================

X = df[
    [
        "每月到店次數",
        "平均消費金額",
        "平均停留時間",
        "優惠券使用次數"
    ]
]


# ==================================================
# 3. 標準化
# ==================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# 將標準化後的資料轉成 DataFrame
scaled_df = pd.DataFrame(
    X_scaled,
    columns=[
        "每月到店次數_標準化",
        "平均消費金額_標準化",
        "平均停留時間_標準化",
        "優惠券使用次數_標準化"
    ]
)

scaled_df.insert(
    0,
    "顧客編號",
    df["顧客編號"]
)

print("\n標準化後資料：")
print(scaled_df.round(2))


# ==================================================
# 4. 四種連結方法
# ==================================================

methods = [
    "single",
    "complete",
    "average",
    "ward"
]


# ==================================================
# 5. 分別畫出四張樹狀圖
# ==================================================

for method in methods:

    # 建立階層式分群結果
    result = linkage(
        X_scaled,
        method=method
    )

    # 印出每一次合併距離
    distances = result[:, 2]

    print("\n" + "=" * 50)
    print(f"{method.upper()} Linkage")
    print("=" * 50)

    print("每一次合併距離：")
    print(np.round(distances, 2))

    # 每一種方法各畫一張樹狀圖
    plt.figure(figsize=(14, 6))

    dendrogram(
        result,
        labels=df["顧客編號"].tolist(),
        leaf_rotation=90,
        leaf_font_size=9
    )

    plt.title(
        f"Hierarchical Clustering - {method.capitalize()} Linkage"
    )

    plt.xlabel("Student")
    plt.ylabel("Distance")

    plt.tight_layout()
    plt.show()