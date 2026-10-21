"""
ppt_helpers.py  -  put this file in the SAME folder as your notebooks.
Every notebook then only needs:  from ppt_helpers import *
All figures are saved as high-resolution PNGs in  ppt_outputs/<algorithm>/
"""
import os, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)

OUT = "ppt_outputs"
COLORS = ["#2E5EAA", "#E07A1F", "#2A9D8F", "#8E5BA8", "#C44E52", "#6C8EAD"]
NOISE_COLOR = "#BDBDBD"
HILITE = "#E07A1F"

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.labelsize": 14, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    "figure.dpi": 100,
})

# ---- symptom vocabulary (taken from your K-Means notebook) -------------------
MANIC = ["racing_thoughts", "inc_speech_rate_and_amount", "grandiosity",
         "distractibility", "persistent_elevated_mood", "persistent_irritability",
         "dec_need_for_sleep", "high_risk_behaviours", "increased_sexual_interest",
         "agitation", "pma_mania"]
DEPRESSIVE = ["low_mood", "anhedonia", "worthless", "guilt", "emptiness",
              "suicidal_ideation", "suicidal_attempt", "poor_concentration",
              "fatigue", "psychomotor_retardation", "loss_of_energy",
              "blunted_affect", "avolition"]
# features shown on the profile heat-maps (missing ones are skipped automatically)
PROFILE_FEATURES = (
    [f + "_Yes" for f in ["persistent_irritability", "dec_need_for_sleep",
                          "inc_speech_rate_and_amount", "pma_mania",
                          "persistent_elevated_mood", "racing_thoughts",
                          "grandiosity", "distractibility"]]
    + [f + "_Yes" for f in ["low_mood", "anhedonia", "emptiness",
                            "poor_concentration", "worthless", "guilt"]]
    + [f + "_Unknown" for f in ["racing_thoughts", "grandiosity", "agitation",
                                "hospitalized"]]
)


def setup(folder):
    """Create ppt_outputs/<folder>/ and make it the save location."""
    global OUT
    OUT = os.path.join("ppt_outputs", folder)
    os.makedirs(OUT, exist_ok=True)
    print("Figures will be saved to:", os.path.abspath(OUT))


def save(name):
    path = os.path.join(OUT, name + ".png")
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.show()
    print("saved ->", path)


def feature_matrix(df):
    """All feature columns as float (drops any column with 'cluster' in its name)."""
    drop = [c for c in df.columns if "cluster" in c.lower()]
    return df.drop(columns=drop).astype(float)


def pretty(col):
    if col.endswith("_Unknown"):
        return col[:-8].replace("_", " ") + " (unknown)"
    return col.replace("_Yes", "").replace("_", " ")


def cname(l, names=None):
    names = names or {}
    return names.get(l, "Noise" if l == -1 else f"Cluster {l}")


def _tick(l, labels, names):
    n = int((np.asarray(labels) == l).sum())
    return "\n".join(textwrap.wrap(cname(l, names), 18)) + f"\n(n={n})"


def _color(i, l):
    return NOISE_COLOR if l == -1 else COLORS[i % len(COLORS)]


# ---- tables -------------------------------------------------------------------
def table_png(df, name, title=None, highlight_rows=(), fontsize=13, digits=3):
    d = df.copy()
    f = lambda v: f"{v:,.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    cells = [[f(v) for v in row] for row in d.values]
    fig, ax = plt.subplots(figsize=(max(6, 2.0 * len(d.columns)),
                                    0.62 * (len(d) + 1) + (0.8 if title else 0.2)))
    ax.axis("off"); ax.grid(False)
    t = ax.table(cellText=cells, colLabels=list(d.columns), cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(fontsize); t.scale(1, 1.7)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor("#2E5EAA"); cell.set_text_props(color="white", weight="bold")
        elif (r - 1) in highlight_rows:
            cell.set_facecolor("#FFF1D6"); cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F7FA")
    if title:
        ax.set_title(title, fontsize=fontsize + 3, weight="bold", pad=10)
    save(name)
    d.to_csv(os.path.join(OUT, name + ".csv"), index=False)


# ---- model-selection curves ---------------------------------------------------
def metric_panel(res, name, highlight_k=None, hue=None, xcol="k"):
    """Silhouette / Davies-Bouldin / Calinski-Harabasz vs k (one panel each)."""
    metrics = [("Silhouette", "higher is better"),
               ("Davies_Bouldin", "lower is better"),
               ("Calinski_Harabasz", "higher is better")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, (m, hint) in zip(axes, metrics):
        if hue:
            for i, (g, s) in enumerate(res.groupby(hue, sort=False)):
                ax.plot(s[xcol], s[m], marker="o", lw=2.4, ms=12 - 3.5 * i,
                        ls=["-", "--", ":"][i % 3], color=COLORS[i], label=g)
        else:
            ax.plot(res[xcol], res[m], marker="o", lw=2.6, ms=9, color=COLORS[0])
        if highlight_k is not None:
            ax.axvspan(highlight_k - 0.4, highlight_k + 0.4, color=HILITE, alpha=0.18)
        ax.set_title(f"{m.replace('_', '-')}\n({hint})", fontsize=15)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(sorted(res[xcol].unique()))
    if hue:
        axes[0].legend()
    plt.tight_layout()
    save(name)


# ---- cluster size / structure -------------------------------------------------
def size_bar(labels, name, names=None, title="Patients per cluster"):
    labels = np.asarray(labels)
    vc = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(1.9 * len(vc) + 3, 5))
    bars = ax.bar([_tick(l, labels, names) for l in vc.index], vc.values,
                  color=[_color(i, l) for i, l in enumerate(vc.index)], width=0.6)
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / len(labels) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=13, weight="bold")
    ax.set_ylim(0, vc.max() * 1.25); ax.set_ylabel("Number of patients"); ax.set_title(title)
    ax.grid(axis="x", visible=False)
    save(name)


def profile_heatmap(X, labels, name, names=None, features=None,
                    title="Cluster profile: share of patients with each feature"):
    feats = [f for f in (features or PROFILE_FEATURES) if f in X.columns]
    labels = np.asarray(labels)
    m = X[feats].groupby(labels).mean().T
    m.columns = [_tick(l, labels, names) for l in m.columns]
    m.index = [pretty(f) for f in m.index]
    fig, ax = plt.subplots(figsize=(2.2 * m.shape[1] + 5, 0.52 * len(m) + 2.5))
    sns.heatmap(m, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, linewidths=.5,
                cbar_kws={"label": "Proportion of patients"}, ax=ax, annot_kws={"size": 12})
    ax.set_title(title); ax.grid(False); ax.tick_params(axis="x", labelsize=12)
    plt.xticks(rotation=0)
    save(name)
    m.to_csv(os.path.join(OUT, name + ".csv"))


def symptom_groups(X):
    manic = [f + "_Yes" for f in MANIC if f + "_Yes" in X.columns]
    dep = [f + "_Yes" for f in DEPRESSIVE if f + "_Yes" in X.columns]
    unk = [c for c in X.columns if c.endswith("_Unknown")]
    return manic, dep, unk


def group_scores(X, labels):
    manic, dep, unk = symptom_groups(X)
    labels = np.asarray(labels)
    return pd.DataFrame({
        "Manic": X[manic].mean(axis=1).groupby(labels).mean(),
        "Depressive": X[dep].mean(axis=1).groupby(labels).mean(),
        "Unknown": X[unk].mean(axis=1).groupby(labels).mean()})


def auto_names(X, labels):
    """Name a 3-cluster solution from its profile (check the printed table!)."""
    sc = group_scores(X, labels).drop(index=-1, errors="ignore")
    print(sc.round(3))
    if len(sc) != 3:
        return {}
    low = sc["Unknown"].idxmax()
    rest = sc.drop(index=low)
    comb = rest["Depressive"].idxmax()
    man = [i for i in rest.index if i != comb][0]
    return {low: "Low recorded symptoms / high unknown",
            man: "Predominantly manic",
            comb: "Combined manic + depressive"}


def symptom_bar(X, labels, name, names=None):
    manic, dep, unk = symptom_groups(X)
    print(f"{len(manic)} manic, {len(dep)} depressive, {len(unk)} unknown-indicator columns used")
    sc = group_scores(X, labels)
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(2.4 * len(sc) + 4, 5.2))
    w = 0.26
    x = np.arange(len(sc))
    for j, (col, c) in enumerate(zip(["Manic", "Depressive", "Unknown"],
                                     [COLORS[1], COLORS[0], NOISE_COLOR])):
        b = ax.bar(x + (j - 1) * w, sc[col], w, label={"Manic": "Manic symptoms",
                   "Depressive": "Depressive symptoms", "Unknown": "Unknown / not recorded"}[col], color=c)
        for r in b:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + .01,
                    f"{r.get_height():.2f}", ha="center", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels([_tick(l, labels, names) for l in sc.index], fontsize=12)
    ax.set_ylabel("Mean proportion (0-1)"); ax.set_ylim(0, 1.12)
    ax.set_title("Manic vs depressive vs unrecorded information"); ax.legend(ncol=3, loc="upper center",
                 bbox_to_anchor=(0.5, -0.22)); ax.grid(axis="x", visible=False)
    save(name)
    sc.round(3).to_csv(os.path.join(OUT, name + ".csv"))


def diff_features_bar(X, labels, name, top=10, title="Top features separating the clusters"):
    labels = np.asarray(labels)
    ok = labels != -1
    m = X[ok].groupby(labels[ok]).mean()
    rng = (m.max() - m.min()).sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.55 * top + 2))
    ax.barh([pretty(c) for c in rng.index], rng.values,
            [NOISE_COLOR if c.endswith("_Unknown") else COLORS[0] for c in rng.index])
    for i, v in enumerate(rng.values):
        ax.text(v + .005, i, f"{v:.2f}", va="center", fontsize=12)
    ax.set_xlabel("Spread of cluster means (max - min)"); ax.set_title(title)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS[0], label="Clinical feature"),
                       Patch(color=NOISE_COLOR, label="'Unknown' indicator (missing data)")],
              loc="lower right")
    ax.grid(axis="y", visible=False)
    save(name)
    rng[::-1].round(4).to_csv(os.path.join(OUT, name + ".csv"))


def pca_scatter(X, labels, name, names=None):
    labels = np.asarray(labels)
    p = PCA(n_components=2, random_state=42).fit(X)
    Z = p.transform(X)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for i, l in enumerate(sorted(set(labels))):
        k = labels == l
        ax.scatter(Z[k, 0], Z[k, 1], s=22 if l == -1 else 42, alpha=.55 if l == -1 else .8,
                   color=_color(i, l), edgecolor="white", linewidth=.4,
                   label=f"{cname(l, names)} (n={k.sum()})")
    ev = p.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}% of variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}% of variance)")
    ax.set_title("Patients projected to 2-D (visualisation only)")
    ax.legend(fontsize=11, loc="best")
    save(name)


def contingency_heatmap(a, b, name, a_name, b_name):
    ct = pd.crosstab(pd.Series(a, name=a_name), pd.Series(b, name=b_name))
    ari, nmi = adjusted_rand_score(a, b), normalized_mutual_info_score(a, b)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1,
                annot_kws={"size": 16}, ax=ax)
    ax.set_title(f"{a_name} vs {b_name}\nARI = {ari:.3f}   NMI = {nmi:.3f}", fontsize=15)
    ax.grid(False)
    save(name)
    return ari, nmi
