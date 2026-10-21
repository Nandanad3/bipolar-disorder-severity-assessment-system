# ===== Append to the END of OPTICS_Clustering notebook =====
# %% CELL 1 - setup, reachability plot
from ppt_helpers import *
setup("optics")

Xn = numerical_df.copy()                 # the 23 numeric columns your notebook used
labels = optics_model.labels_
order = optics_model.ordering_
reach = optics_model.reachability_[order].copy()
reach[~np.isfinite(reach)] = np.nan
lab_o = labels[order]

fig, ax = plt.subplots(figsize=(13, 5))
cols = [NOISE_COLOR if l == -1 else COLORS[l % len(COLORS)] for l in lab_o]
ax.bar(range(len(reach)), np.nan_to_num(reach), color=cols, width=1.0)
ax.set_xlabel("Patients in OPTICS ordering"); ax.set_ylabel("Reachability distance")
ax.set_title("OPTICS reachability plot (grey = noise)"); ax.grid(False)
save("01_reachability")

# %% CELL 2 - noise vs clustered
n_clu = len(set(labels)) - (1 if -1 in labels else 0)
size_bar(labels, "02_sizes_with_noise", None, "OPTICS: clusters and noise points")

# %% CELL 3 - metrics table, put next to coverage (IMPORTANT for honest reporting)
ok = labels != -1
sil, dbi, chi = (silhouette_score(Xn[ok], labels[ok]), davies_bouldin_score(Xn[ok], labels[ok]),
                 calinski_harabasz_score(Xn[ok], labels[ok]))
summary = pd.DataFrame({
    "Item": ["Features used", "Clusters found", "Patients clustered", "Patients labelled noise",
             "Silhouette (clustered only)", "Davies-Bouldin (clustered only)",
             "Calinski-Harabasz (clustered only)"],
    "Value": [Xn.shape[1], n_clu, f"{ok.sum()} ({ok.mean() * 100:.1f}%)",
              f"{(~ok).sum()} ({(~ok).mean() * 100:.1f}%)", f"{sil:.3f}", f"{dbi:.3f}", f"{chi:.2f}"]})
table_png(summary, "03_summary", title="OPTICS result")

# %% CELL 4 - 2-D view
pca_scatter(Xn, labels, "04_pca_scatter", None)
