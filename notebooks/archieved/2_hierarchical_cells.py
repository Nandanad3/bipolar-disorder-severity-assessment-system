# ===== Append to the END of BD_Hierarchical_Cluster notebook =====
# %% CELL 1 - setup + metrics k = 2..10
from ppt_helpers import *
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
setup("hierarchical")

X = feature_matrix(df)
rows = []
for k in range(2, 11):
    l = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward").fit_predict(X)
    rows.append(dict(k=k, Silhouette=silhouette_score(X, l),
                     Davies_Bouldin=davies_bouldin_score(X, l),
                     Calinski_Harabasz=calinski_harabasz_score(X, l)))
res = pd.DataFrame(rows)
metric_panel(res, "01_validation_indices", highlight_k=3)
table_png(res, "02_metric_table", title="Hierarchical (Ward) validation scores",
          highlight_rows=[list(res.k).index(3)])

# %% CELL 2 - readable dendrogram with the 3-cluster cut line
Z = linkage(X, method="ward")
cut = (Z[-3, 2] + Z[-2, 2]) / 2            # height that leaves exactly 3 clusters
fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(Z, truncate_mode="lastp", p=30, color_threshold=cut, show_leaf_counts=True,
           leaf_font_size=11, above_threshold_color="#8C8C8C", ax=ax)
ax.axhline(cut, color=HILITE, ls="--", lw=2.5)
ax.text(ax.get_xlim()[1] * 0.98, cut * 1.02, "cut -> 3 clusters", color=HILITE,
        ha="right", fontsize=14, weight="bold")
ax.set_title("Dendrogram (Ward linkage, last 30 merges shown)")
ax.set_xlabel("Cluster size (number of patients)"); ax.set_ylabel("Merge distance")
ax.grid(False)
save("03_dendrogram")

# %% CELL 3 - merge-distance plot (supports the choice of 3)
dist = Z[-10:, 2][::-1]                  # distances of the last 10 merges
clusters_left = np.arange(2, 12)[:10]    # merge that leaves 2,3,...,11 clusters
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(clusters_left, dist, marker="o", lw=2.6, ms=9, color=COLORS[0])
ax.scatter([3], [dist[1]], s=300, color=HILITE, zorder=5)
ax.set_xlabel("Number of clusters after the merge"); ax.set_ylabel("Merge distance")
ax.set_title("Last 10 merge distances"); ax.set_xticks(clusters_left)
save("04_merge_distances")

# %% CELL 4 - final 3-cluster solution
labels = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage="ward").fit_predict(X)
names = auto_names(X, labels)           # CHECK the printed table; edit by hand if wrong
size_bar(labels, "05_cluster_sizes", names, "Hierarchical (k = 3): patients per cluster")
profile_heatmap(X, labels, "06_profile_heatmap", names, title="Hierarchical: cluster profile")
symptom_bar(X, labels, "07_symptom_summary", names)
pca_scatter(X, labels, "08_pca_scatter", names)

# %% CELL 5 - agreement with K-Means (very useful discussion slide)
km_labels = KMeans(n_clusters=3, random_state=42, n_init="auto").fit_predict(X)
ari, nmi = contingency_heatmap(km_labels, labels, "09_kmeans_vs_hier", "K-Means", "Hierarchical")

# %% CELL 6 - key numbers
key = pd.DataFrame({
    "Item": ["Linkage / distance", "Clusters", "Merge height of the cut", "Silhouette",
             "Davies-Bouldin", "Calinski-Harabasz", "ARI vs K-Means"],
    "Value": ["Ward / Euclidean", 3, f"{Z[-3, 2]:.2f}", f"{silhouette_score(X, labels):.3f}",
              f"{davies_bouldin_score(X, labels):.3f}", f"{calinski_harabasz_score(X, labels):.2f}",
              f"{ari:.3f}"]})
table_png(key, "10_key_numbers", title="Hierarchical final model (k = 3)")
