import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# 資料
# A=1、B=3、C=8、D=12
X = [
    [1],
    [3],
    [8],
    [12]
]

labels = ["A", "B", "C", "D"]

# 四種連結方法
methods = ["single", "complete", "average", "ward"]

for method in methods:
    # 建立階層式分群結果
    result = linkage(X, method=method)

    # 印出合併過程
    print(f"\n{method.upper()} Linkage")
    print(result)

    # 每一種方法各畫一張圖
    plt.figure(figsize=(7, 4))

    dendrogram(
        result,
        labels=labels
    )


    plt.title(f"Hierarchical Clustering - {method.capitalize()} Linkage")
    plt.xlabel("Student")
    plt.ylabel("Distance")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()