# ===== Append to the END of BD_DBSCAN notebook =====
# %% CELL 1 - setup + k-distance plot (justifies eps)
from ppt_helpers import *
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
setup("dbscan")

X = feature_matrix(df)
EPS, MIN_S = 4.0, 5
d, _ = NearestNeighbors(n_neighbors=MIN_S).fit(X).kneighbors(X)
kd = np.sort(d[:, -1])
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(kd, lw=2.6, color=COLORS[0])
ax.axhline(EPS, color=HILITE, ls="--", lw=2.5)
ax.text(5, EPS + .1, f"chosen eps = {EPS}", color=HILITE, fontsize=14, weight="bold")
ax.set_xlabel("Patients sorted by distance"); ax.set_ylabel(f"Distance to {MIN_S}th nearest neighbour")
ax.set_title("k-distance plot")
save("01_k_distance")

# %% CELL 2 - eps sensitivity table
rows = []
for eps in [2, 2.5, 3, 3.5, 4, 4.5, 5, 6]:
    l = DBSCAN(eps=eps, min_samples=MIN_S).fit_predict(X)
    rows.append({"eps": eps, "Clusters": len(set(l)) - (1 if -1 in l else 0),
                 "Noise points": int((l == -1).sum()), "Noise (%)": round((l == -1).mean() * 100, 1)})
sens = pd.DataFrame(rows)
table_png(sens, "02_eps_sensitivity", title=f"DBSCAN sensitivity (min_samples = {MIN_S})",
          highlight_rows=[list(sens.eps).index(EPS)], digits=1)

# %% CELL 3 - final DBSCAN result incl. noise
raw = DBSCAN(eps=EPS, min_samples=MIN_S).fit_predict(X)
size_bar(raw, "03_sizes_with_noise", None, "DBSCAN: clusters and noise points")
pca_scatter(X, raw, "04_pca_scatter", None)

# %% CELL 4 - noise re-assigned to nearest cluster (as in the notebook)
core = raw != -1
nn = NearestNeighbors(n_neighbors=1).fit(X.values[core])
_, idx = nn.kneighbors(X.values[~core])
assigned = raw.copy()
assigned[~core] = raw[core][idx[:, 0]]
size_bar(assigned, "05_sizes_noise_assigned", None, "DBSCAN after assigning noise to nearest cluster")

# %% CELL 5 - what drives DBSCAN clusters? (shows the 'Unknown' issue)
diff_features_bar(X, assigned, "06_top_features", top=10,
                  title="DBSCAN: features separating the clusters")
profile_heatmap(X, assigned, "07_profile_heatmap", None, title="DBSCAN: cluster profile")

# %% CELL 6 - metrics table (two views)
def m(Xv, l):
    return [silhouette_score(Xv, l), davies_bouldin_score(Xv, l), calinski_harabasz_score(Xv, l)]
a = m(X.values[core], raw[core])
b = m(X.values, assigned)
mt = pd.DataFrame({"Evaluation set": [f"Clustered points only (n={core.sum()})",
                                      f"All points, noise re-assigned (n={len(X)})"],
                   "Silhouette": [a[0], b[0]], "Davies-Bouldin": [a[1], b[1]],
                   "Calinski-Harabasz": [a[2], b[2]]})
table_png(mt, "08_metrics", title="DBSCAN validation scores")
