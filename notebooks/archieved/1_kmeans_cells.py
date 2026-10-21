# ===== Append to the END of BD_Kmeans_Clustering notebook (one cell per "# %%") =====
# %% CELL 1 - setup + metrics for k = 1..10
from ppt_helpers import *
from sklearn.cluster import KMeans
setup("kmeans")

X = feature_matrix(df)            # 342 x 168, label columns removed
rows, wcss = [], []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    wcss.append(km.inertia_)
    if k > 1:
        l = km.labels_
        rows.append(dict(k=k, Silhouette=silhouette_score(X, l),
                         Davies_Bouldin=davies_bouldin_score(X, l),
                         Calinski_Harabasz=calinski_harabasz_score(X, l)))
res = pd.DataFrame(rows)

# %% CELL 2 - Elbow plot  (slide: "Choosing k")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), wcss, marker="o", lw=2.6, ms=9, color=COLORS[0])
ax.scatter([3], [wcss[2]], s=300, color=HILITE, zorder=5)
ax.annotate("k = 3\nWCSS = %.0f" % wcss[2], (3, wcss[2]), xytext=(4.2, wcss[2] + 900),
            fontsize=14, weight="bold", arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("WCSS (within-cluster sum of squares)")
ax.set_title("Elbow method - K-Means"); ax.set_xticks(range(1, 11))
save("01_elbow")

# %% CELL 3 - the three validation indices side by side
metric_panel(res, "02_validation_indices", highlight_k=3)

# %% CELL 4 - metric table (k = 3 highlighted)
table_png(res, "03_metric_table", title="K-Means validation scores",
          highlight_rows=[list(res.k).index(3)])

# %% CELL 5 - final k = 3 model
final = KMeans(n_clusters=3, random_state=42, n_init="auto").fit(X)
labels = final.labels_
names = auto_names(X, labels)       # CHECK the printed table, edit dict by hand if needed
print(names)
size_bar(labels, "04_cluster_sizes", names, "K-Means (k = 3): patients per cluster")

# %% CELL 6 - cluster profile heat-map  (the most important slide)
profile_heatmap(X, labels, "05_profile_heatmap", names,
                title="K-Means: cluster profile")

# %% CELL 7 - manic vs depressive vs unknown summary
symptom_bar(X, labels, "06_symptom_summary", names)

# %% CELL 8 - top differentiating features
diff_features_bar(X, labels, "07_top_features", top=10,
                  title="K-Means: features separating the 3 clusters")

# %% CELL 9 - 2-D picture of the clusters
pca_scatter(X, labels, "08_pca_scatter", names)

# %% CELL 10 - key numbers table
key = pd.DataFrame({
    "Item": ["Patients x features", "Clusters", "Final WCSS", "Iterations to converge",
             "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
    "Value": [f"{X.shape[0]} x {X.shape[1]}", 3, f"{final.inertia_:,.1f}", final.n_iter_,
              f"{silhouette_score(X, labels):.3f}", f"{davies_bouldin_score(X, labels):.3f}",
              f"{calinski_harabasz_score(X, labels):.2f}"]})
table_png(key, "09_key_numbers", title="K-Means final model (k = 3)")
