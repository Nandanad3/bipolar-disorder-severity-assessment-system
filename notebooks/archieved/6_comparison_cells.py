# ===== Append to the END of the BD-comparison notebook (needs results_df and X from it) =====
# %% CELL 1 - setup + the 3 metrics for the 3 algorithms
from ppt_helpers import *
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
setup("comparison")

X = feature_matrix(X) if "cluster" in "".join(X.columns).lower() else X.astype(float)
metric_panel(results_df, "01_metrics_vs_k", highlight_k=3, hue="Algorithm")

# %% CELL 2 - table at k = 3 and best k per metric
at3 = results_df[results_df.k == 3][["Algorithm", "Silhouette", "Davies_Bouldin", "Calinski_Harabasz"]]
table_png(at3, "02_table_k3", title="All three algorithms at k = 3")
best = []
for alg, s in results_df.groupby("Algorithm", sort=False):
    best.append({"Algorithm": alg,
                 "Best k (Silhouette)": int(s.loc[s.Silhouette.idxmax(), "k"]),
                 "Best k (Davies-Bouldin)": int(s.loc[s.Davies_Bouldin.idxmin(), "k"]),
                 "Best k (Calinski-Harabasz)": int(s.loc[s.Calinski_Harabasz.idxmax(), "k"])})
table_png(pd.DataFrame(best), "03_best_k", title="Which k does each index prefer?")

# %% CELL 3 - agreement between algorithms at k = 3
km = KMeans(n_clusters=3, random_state=42, n_init="auto").fit_predict(X)
hc = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)
gm = GaussianMixture(n_components=3, random_state=42).fit_predict(X)
labs = {"K-Means": km, "Hierarchical": hc, "GMM": gm}
M = pd.DataFrame([[adjusted_rand_score(labs[a], labs[b]) for b in labs] for a in labs],
                 index=labs, columns=labs)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(M, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, ax=ax,
            annot_kws={"size": 16}, cbar_kws={"label": "Adjusted Rand Index"})
ax.set_title("Pairwise agreement (k = 3)"); ax.grid(False)
save("04_pairwise_ari")

# %% CELL 4 - stability over random seeds (K-Means and GMM)
seeds = [0, 1, 2, 3, 4, 5, 10, 20, 42, 100]
ref_km = KMeans(n_clusters=3, random_state=42, n_init="auto").fit_predict(X)
ref_gm = GaussianMixture(n_components=3, random_state=42).fit_predict(X)
st = pd.DataFrame({
    "seed": seeds,
    "K-Means": [adjusted_rand_score(ref_km, KMeans(n_clusters=3, random_state=s, n_init="auto").fit_predict(X)) for s in seeds],
    "GMM": [adjusted_rand_score(ref_gm, GaussianMixture(n_components=3, random_state=s).fit_predict(X)) for s in seeds]})
fig, ax = plt.subplots(figsize=(10, 5))
w = 0.38; x = np.arange(len(seeds))
ax.bar(x - w / 2, st["K-Means"], w, color=COLORS[0], label="K-Means")
ax.bar(x + w / 2, st["GMM"], w, color=COLORS[1], label="GMM")
ax.axhline(st["K-Means"].mean(), color="gray", ls="--")
ax.text(len(seeds) - .5, st["K-Means"].mean() - .06, f"mean ARI = {st['K-Means'].mean():.3f}", ha="right", fontsize=13)
ax.set_xticks(x); ax.set_xticklabels(seeds); ax.set_xlabel("Random seed"); ax.set_ylabel("ARI vs seed 42")
ax.set_ylim(0.5, 1.05); ax.set_title("Cluster stability across initialisations"); ax.legend(ncol=2)
ax.grid(axis="x", visible=False)
save("05_stability")

# %% CELL 5 - final all-algorithm summary (numbers copied from your five notebooks)
summ = pd.DataFrame([
    ["K-Means (k=3)",     3, "0 %",    0.265, 1.433, 124.6, "Selected"],
    ["GMM (3 comp.)",     3, "0 %",    0.265, 1.433, 124.6, "Same partition as K-Means"],
    ["Hierarchical (k=3)",3, "0 %",    0.245, 1.645, 114.9, "Different partition"],
    ["DBSCAN",            5, "8.5 %",  0.298, 1.310, 100.7, "Driven by 'Unknown' flags"],
    ["OPTICS",            4, "89.5 %", 0.651, 0.599, 50.5,  "Unusable: most points noise"],
], columns=["Algorithm", "Clusters", "Noise", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Verdict"])
table_png(summ, "06_final_summary", title="Algorithm comparison", highlight_rows=[0], fontsize=12)
