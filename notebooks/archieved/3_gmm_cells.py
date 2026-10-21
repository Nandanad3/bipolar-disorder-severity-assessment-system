# ===== Append to the END of BD_GMM_Cl notebook =====
# %% CELL 1 - setup + BIC / AIC curves
from ppt_helpers import *
from sklearn.mixture import GaussianMixture
from scipy.stats import f_oneway
setup("gmm")

X = feature_matrix(df)
rows = []
for k in range(1, 11):
    g = GaussianMixture(n_components=k, random_state=42).fit(X)
    rows.append(dict(k=k, AIC=g.aic(X), BIC=g.bic(X)))
ic = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, col in zip(axes, ["BIC", "AIC"]):
    ax.plot(ic.k, ic[col], marker="o", lw=2.6, ms=9, color=COLORS[0])
    best = ic.loc[ic[col].idxmin(), "k"]
    ax.scatter([best], [ic[col].min()], s=300, color=HILITE, zorder=5)
    ax.set_title(f"{col} (lower is better) - minimum at k = {int(best)}")
    ax.set_xlabel("Number of components"); ax.set_xticks(range(1, 11))
plt.tight_layout()
save("01_bic_aic")
table_png(ic, "02_bic_aic_table", title="GMM information criteria", digits=1)

# %% CELL 2 - final 3-component model
gmm = GaussianMixture(n_components=3, random_state=42).fit(X)
labels = gmm.predict(X)
proba = gmm.predict_proba(X)
names = auto_names(X, labels)           # CHECK the printed table; edit by hand if wrong
size_bar(labels, "03_cluster_sizes", names, "GMM (3 components): patients per cluster")
profile_heatmap(X, labels, "04_profile_heatmap", names, title="GMM: component profile")
symptom_bar(X, labels, "05_symptom_summary", names)
pca_scatter(X, labels, "06_pca_scatter", names)

# %% CELL 3 - membership confidence (what makes GMM special: soft clustering)
conf = proba.max(axis=1)
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.hist(conf, bins=20, color=COLORS[0], edgecolor="white")
ax.axvline(0.9, color=HILITE, ls="--", lw=2.5)
ax.set_xlabel("Highest membership probability of a patient"); ax.set_ylabel("Number of patients")
ax.set_title(f"{(conf >= 0.9).mean() * 100:.1f}% of patients are assigned with >= 90% confidence")
save("07_membership_confidence")
bins = pd.cut(conf, [0, .5, .7, .9, 1.0], labels=["<=50%", "50-70%", "70-90%", "90-100%"], include_lowest=True)
bt = bins.value_counts().sort_index().rename_axis("Confidence").reset_index(name="Patients")
bt["Share (%)"] = (bt["Patients"] / len(conf) * 100).round(1)
table_png(bt, "08_confidence_table", title="Assignment confidence", digits=1)

# %% CELL 4 - ANOVA F-score: top features
F = {}
for c in X.columns:
    groups = [X.loc[labels == l, c].values for l in np.unique(labels)]
    try:
        F[c] = f_oneway(*groups)[0]
    except Exception:
        F[c] = 0.0
F = pd.Series(F).replace([np.inf, -np.inf], np.nan).fillna(0).sort_values(ascending=False).head(10)[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([pretty(c) for c in F.index], F.values,
        color=[NOISE_COLOR if c.endswith("_Unknown") else COLORS[0] for c in F.index])
ax.set_xlabel("ANOVA F-score (higher = stronger separation)")
ax.set_title("GMM: top 10 differentiating features"); ax.grid(axis="y", visible=False)
save("09_anova_top_features")

# %% CELL 5 - key numbers
key = pd.DataFrame({
    "Item": ["Components", "Covariance type", "Converged / iterations", "Mixture weights",
             "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
    "Value": [3, gmm.covariance_type, f"{gmm.converged_} / {gmm.n_iter_}",
              ", ".join(f"{w:.3f}" for w in gmm.weights_),
              f"{silhouette_score(X, labels):.3f}", f"{davies_bouldin_score(X, labels):.3f}",
              f"{calinski_harabasz_score(X, labels):.2f}"]})
table_png(key, "10_key_numbers", title="GMM final model (3 components)")
