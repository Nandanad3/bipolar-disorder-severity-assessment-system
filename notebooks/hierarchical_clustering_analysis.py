import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import f_oneway
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

# Suppress warnings for clean console output
warnings.filterwarnings("ignore")

# Global styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
HILITE = "#d62728"


# ============================================================
# HELPER UTILITIES FOR PRINTING & SAVING FIGURES
# ============================================================
def print_header(title: str, char: str = "=") -> None:
    """Prints a formatted section header to the console."""
    width = 78
    print(f"\n{char * width}")
    print(f" {title.upper()}")
    print(f"{char * width}")


def print_table(df: pd.DataFrame, float_fmt: str = ".4f") -> None:
    """Prints a pandas DataFrame cleanly in pure Python terminal environments."""
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 1000,
        "display.float_format", f"{{:{float_fmt}}}".format,
    ):
        print(df)


def save_or_show(fig: plt.Figure, filename: str, out_dir: Optional[Path] = None) -> None:
    """Saves the figure to disk if out_dir is specified, then displays it."""
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"{filename}.png"
        fig.savefig(filepath, bbox_inches="tight", dpi=150)
        print(f"[Saved Figure] -> {filepath}")
    plt.show()


# ============================================================
# 1. LOAD & PREPARE DATASET
# ============================================================
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    print_header("1. Loading Preprocessed Dataset")
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(path).copy()
    print(f"Loaded '{path.name}' successfully.")
    print(f"Original Shape : {df.shape[0]} rows × {df.shape[1]} columns")

    # Drop any existing cluster label columns
    exclude_cols = [
        "cluster_label", "Cluster", "Hierarchical_Cluster",
        "Spectral_Cluster", "GMM_Cluster", "BIRCH_Cluster",
        "DBSCAN_Cluster", "OPTICS_Cluster", "MeanShift_Cluster",
        "AffinityPropagation_Cluster",
    ]
    X = df.drop(columns=exclude_cols, errors="ignore").copy()
    print(f"Feature Matrix : {X.shape[0]} rows × {X.shape[1]} features")
    return X


# ============================================================
# 2. EVALUATE HIERARCHICAL CLUSTERING (k = 2..10)
# ============================================================
def evaluate_k_range(
    X: pd.DataFrame, k_max: int = 10, highlight_k: int = 3, out_dir: Optional[Path] = None
) -> pd.DataFrame:
    print_header(f"2. Hierarchical Clustering Evaluation (k = 2 to {k_max})")

    rows = []
    for k in range(2, k_max + 1):
        model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        labels = model.fit_predict(X)
        rows.append({
            "k": k,
            "Silhouette": silhouette_score(X, labels),
            "Davies_Bouldin": davies_bouldin_score(X, labels),
            "Calinski_Harabasz": calinski_harabasz_score(X, labels),
        })

    res_df = pd.DataFrame(rows).set_index("k")
    print_table(res_df, float_fmt=".4f")

    best_sil_k = res_df["Silhouette"].idxmax()
    best_db_k = res_df["Davies_Bouldin"].idxmin()
    best_ch_k = res_df["Calinski_Harabasz"].idxmax()

    print("\n--- Best k According to Each Metric ---")
    print(f"Max Silhouette Score        : k = {best_sil_k} ({res_df.loc[best_sil_k, 'Silhouette']:.4f})")
    print(f"Min Davies-Bouldin Index    : k = {best_db_k} ({res_df.loc[best_db_k, 'Davies_Bouldin']:.4f})")
    print(f"Max Calinski-Harabasz Index : k = {best_ch_k} ({res_df.loc[best_ch_k, 'Calinski_Harabasz']:.4f})")

    # Visualization: 1x3 Validation Metric Panel
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metrics_info = [
        ("Silhouette", "Silhouette Score (Higher = Better)", COLORS[0], best_sil_k),
        ("Davies_Bouldin", "Davies-Bouldin Index (Lower = Better)", COLORS[2], best_db_k),
        ("Calinski_Harabasz", "Calinski-Harabasz Index (Higher = Better)", COLORS[1], best_ch_k),
    ]

    for ax, (col, title, color, best_k) in zip(axes, metrics_info):
        ax.plot(res_df.index, res_df[col], marker="o", lw=2.4, ms=7, color=color)
        ax.axvline(best_k, color=color, linestyle="--", alpha=0.5, label=f"Best (k={best_k})")
        ax.scatter(
            [highlight_k], [res_df.loc[highlight_k, col]],
            s=180, color=HILITE, zorder=5, label=f"Selected (k={highlight_k})"
        )
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel(col.replace("_", "-"))
        ax.set_xticks(list(res_df.index))
        ax.legend(fontsize=9)

    plt.suptitle("Hierarchical Clustering (Ward Linkage): Validation Indices", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    save_or_show(fig, "01_validation_indices", out_dir)

    return res_df


# ============================================================
# 3. DENDROGRAM & MERGE DISTANCE ANALYSIS
# ============================================================
def analyze_dendrogram_and_merges(
    X: pd.DataFrame, n_clusters: int = 3, out_dir: Optional[Path] = None
) -> np.ndarray:
    print_header("3. Dendrogram & Linkage Merge Distance Analysis")

    Z = linkage(X, method="ward")
    print(f"Linkage Matrix Shape : {Z.shape}")

    last_15_df = pd.DataFrame(
        Z[-15:],
        columns=["Cluster_1", "Cluster_2", "Merge_Distance", "Cluster_Size"],
        index=range(len(Z) - 15, len(Z)),
    )
    last_15_df.index.name = "Merge_Step"
    print("\n--- Last 15 Merge Steps (Ward Linkage) ---")
    print_table(last_15_df, float_fmt=".4f")

    merge_idx = len(X) - n_clusters - 1
    merge_dist_k = Z[merge_idx, 2]
    cut_height = (Z[-n_clusters, 2] + Z[-(n_clusters - 1), 2]) / 2

    print(f"\nMerge distance creating transition to {n_clusters} clusters : {merge_dist_k:.4f}")
    print(f"Recommended Dendrogram Cut Height (for k={n_clusters})     : {cut_height:.4f}")

    # Plot 1: Truncated Dendrogram with Cut Threshold Line
    fig, ax = plt.subplots(figsize=(13, 6))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30,
        color_threshold=cut_height,
        show_leaf_counts=True,
        leaf_font_size=10,
        above_threshold_color="#8C8C8C",
        ax=ax,
    )
    ax.axhline(cut_height, color=HILITE, ls="--", lw=2.2)
    ax.text(
        ax.get_xlim()[1] * 0.98,
        cut_height * 1.03,
        f"Cut Threshold ({cut_height:.1f}) -> {n_clusters} Clusters",
        color=HILITE, ha="right", fontsize=11.5, weight="bold",
    )
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage, Last 30 Merges)", fontsize=13.5, fontweight="bold")
    ax.set_xlabel("Cluster Size (Number of Patients) or Sample Index", fontsize=11)
    ax.set_ylabel("Ward Merge Distance (Euclidean)", fontsize=11)
    ax.grid(False)
    plt.tight_layout()
    save_or_show(fig, "03_dendrogram", out_dir)

    # Plot 2: Merge Distance Scree Plot (Last 10 Merges)
    dist = Z[-10:, 2][::-1]
    clusters_left = np.arange(2, 12)[:10]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(clusters_left, dist, marker="o", lw=2.4, ms=8, color=COLORS[0])
    ax.scatter([n_clusters], [dist[n_clusters - 2]], s=220, color=HILITE, zorder=5)
    ax.annotate(
        f"k = {n_clusters}\nMerge Dist = {dist[n_clusters - 2]:.2f}",
        xy=(n_clusters, dist[n_clusters - 2]),
        xytext=(n_clusters + 1.2, dist[n_clusters - 2] + 3),
        fontsize=10.5, weight="bold",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
    )
    ax.set_title("Merge Distance Jump Across Last 10 Merges", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Clusters Remaining After Merge", fontsize=11)
    ax.set_ylabel("Ward Merge Distance", fontsize=11)
    ax.set_xticks(clusters_left)
    plt.tight_layout()
    save_or_show(fig, "04_merge_distances", out_dir)

    return Z


# ============================================================
# 4. FIT FINAL MODEL, FEATURE DIFFERENCES & CLINICAL PROFILES
# ============================================================
def analyze_final_clusters(
    X: pd.DataFrame, n_clusters: int = 3, out_dir: Optional[Path] = None
) -> Tuple[np.ndarray, Dict[int, str]]:
    print_header(f"4. Final Agglomerative Clustering Analysis (k = {n_clusters})")

    model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    labels = model.fit_predict(X)

    # Cluster counts
    counts = pd.Series(labels, name="Cluster").value_counts().sort_index()
    dist_df = pd.DataFrame({
        "Patient_Count": counts.values,
        "Percentage_%": (counts.values / len(labels)) * 100,
    }, index=counts.index)

    print("\n--- Cluster Size Distribution ---")
    print_table(dist_df, float_fmt=".2f")

    # Compute cluster means, global means, and differences (fixes notebook NameError)
    X_numeric = X.astype(float)
    df_clustered = X_numeric.assign(cluster_label=labels)
    cluster_means = df_clustered.groupby("cluster_label").mean()
    overall_means = X_numeric.mean()
    mean_diffs = cluster_means.subtract(overall_means, axis=1)

    # Rank features by max-min spread & ANOVA F-score
    mean_range = cluster_means.max(axis=0) - cluster_means.min(axis=0)
    f_scores = {
        col: f_oneway(*[g[col].values for _, g in df_clustered.groupby("cluster_label")])[0]
        for col in X_numeric.columns
    }

    diff_summary = cluster_means.T.copy()
    diff_summary.columns = [f"Cluster_{c}_Mean" for c in diff_summary.columns]
    diff_summary["Overall_Mean"] = overall_means
    diff_summary["Max_Min_Range"] = mean_range
    diff_summary["ANOVA_F"] = pd.Series(f_scores)
    diff_summary = diff_summary.sort_values(by="Max_Min_Range", ascending=False)

    print("\n--- Top 20 Differentiating Features (Cluster Means vs. Overall Mean) ---")
    print_table(diff_summary.head(20), float_fmt=".4f")

    # Clinical Domain Summary (Manic, Depressive, Unknown)
    manic_keywords = [
        "racing_thoughts", "inc_speech_rate", "grandiosity", "distractibility",
        "persistent_elevated_mood", "dec_need_for_sleep", "high_risk_behaviours",
        "increased_sexual_interest", "agitation", "pma_mania", "ever_manic",
    ]
    dep_keywords = [
        "low_mood", "anhedonia", "worthless", "guilt", "emptiness",
        "suicidal", "poor_concentration", "blunted_affect", "avolition", "ever_depressive",
    ]

    manic_yes_cols = [c for c in X.columns if c.endswith("_Yes") and any(k in c for k in manic_keywords)]
    dep_yes_cols = [c for c in X.columns if c.endswith("_Yes") and any(k in c for k in dep_keywords)]
    unknown_cols = [c for c in X.columns if c.endswith("_Unknown")]

    print(
        f"\n[Clinical Domain Grouping]: {len(manic_yes_cols)} Manic (_Yes), "
        f"{len(dep_yes_cols)} Depressive (_Yes), and {len(unknown_cols)} Unknown-indicator columns."
    )

    domain_summary = pd.DataFrame({
        "Manic": cluster_means[manic_yes_cols].mean(axis=1),
        "Depressive": cluster_means[dep_yes_cols].mean(axis=1),
        "Unknown": cluster_means[unknown_cols].mean(axis=1),
    })

    # Assign descriptive clinical names based on domain scores
    cluster_names = {}
    for c in domain_summary.index:
        m_val, d_val, u_val = domain_summary.loc[c, ["Manic", "Depressive", "Unknown"]]
        if u_val > 0.50:
            cluster_names[c] = f"Cluster {c}: High Unknown / Unrecorded (n={counts[c]})"
        elif m_val > 0.50 and d_val > 0.35:
            cluster_names[c] = f"Cluster {c}: Combined Manic + Depressive (n={counts[c]})"
        else:
            cluster_names[c] = f"Cluster {c}: Predominantly Manic (n={counts[c]})"

    domain_summary["Assigned_Profile"] = [cluster_names[c] for c in domain_summary.index]
    print("\n--- Clinical Domain Proportions by Cluster ---")
    print_table(domain_summary, float_fmt=".3f")

    # Visualization 1: Cluster Sizes Bar Chart
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(
        [f"Cluster {c}" for c in counts.index],
        counts.values,
        color=COLORS[:n_clusters],
        edgecolor="black",
        width=0.5,
        alpha=0.9,
    )
    for bar, pct in zip(bars, dist_df["Percentage_%"]):
        h = bar.get_height()
        ax.annotate(
            f"{int(h)} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontweight="bold", fontsize=10,
        )
    ax.set_title("Hierarchical Clustering (k = 3): Patients per Cluster", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Patients")
    ax.set_ylim(0, counts.max() * 1.18)
    plt.tight_layout()
    save_or_show(fig, "05_cluster_sizes", out_dir)

    # Visualization 2: Top 20 Differentiating Features Heatmap
    top_20_cols = diff_summary.head(20).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cluster_means[top_20_cols].T,
        annot=True, fmt=".2f", cmap="YlGnBu",
        xticklabels=[f"Cluster {c}\n(n={counts[c]})" for c in counts.index],
        cbar_kws={"label": "Cluster Mean Value"},
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Hierarchical Clustering: Top 20 Differentiating Features", fontsize=13, fontweight="bold")
    ax.set_ylabel("Features")
    plt.tight_layout()
    save_or_show(fig, "06_profile_heatmap", out_dir)

    # Visualization 3: Grouped Bar Chart of Clinical Domains (Manic, Depressive, Unknown)
    plot_dom = domain_summary[["Manic", "Depressive", "Unknown"]].reset_index().melt(
        id_vars="cluster_label", var_name="Domain", value_name="Mean_Proportion"
    )
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.barplot(
        data=plot_dom, x="cluster_label", y="Mean_Proportion", hue="Domain",
        palette=["#e41a1c", "#377eb8", "#7f7f7f"], edgecolor="black", ax=ax
    )
    ax.set_title("Symptom Domain Summary Across Hierarchical Clusters", fontsize=13