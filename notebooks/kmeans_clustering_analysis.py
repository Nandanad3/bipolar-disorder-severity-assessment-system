import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# Suppress warnings for clean console output
warnings.filterwarnings("ignore")

# Global styling and color palette
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
HILITE = "#d62728"


# ============================================================
# HELPER UTILITIES FOR PRINTING & SAVING FIGURES
# ============================================================
def print_header(title: str, char: str = "=") -> None:
    """Prints a clean section header to the terminal."""
    width = 78
    print(f"\n{char * width}")
    print(f" {title.upper()}")
    print(f"{char * width}")


def print_table(df: pd.DataFrame, float_fmt: str = ".4f") -> None:
    """Prints a pandas DataFrame cleanly in standard terminal environments."""
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 1000,
        "display.float_format", f"{{:{float_fmt}}}".format,
    ):
        print(df)


def save_or_show(fig: plt.Figure, filename: str, out_dir: Optional[Path] = None) -> None:
    """Saves the figure to disk if out_dir is provided, then displays it."""
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

    df = pd.read_csv(path)
    print(f"Loaded '{path.name}' successfully.")
    print(f"Original Shape : {df.shape[0]} rows × {df.shape[1]} columns")

    # Exclude any pre-existing cluster label columns
    exclude_cols = [
        "Cluster", "Hierarchical_Cluster", "Spectral_Cluster",
        "GMM_Cluster", "BIRCH_Cluster", "DBSCAN_Cluster",
        "OPTICS_Cluster", "MeanShift_Cluster", "AffinityPropagation_Cluster",
    ]
    X = df.drop(columns=exclude_cols, errors="ignore").copy()
    print(f"Feature Matrix : {X.shape[0]} rows × {X.shape[1]} features")
    return X


# ============================================================
# 2. EVALUATE K-MEANS FOR k = 1..10
# ============================================================
def evaluate_k_range(
    X: pd.DataFrame, k_max: int = 10, selected_k: int = 3, out_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print_header(f"2. K-Means Model Selection (k = 1 to {k_max})")

    k_values = range(1, k_max + 1)
    wcss = []
    metrics_rows = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)

        if k > 1:
            metrics_rows.append({
                "k": k,
                "WCSS": kmeans.inertia_,
                "Silhouette": silhouette_score(X, labels),
                "Davies_Bouldin": davies_bouldin_score(X, labels),
                "Calinski_Harabasz": calinski_harabasz_score(X, labels),
            })

    wcss_df = pd.DataFrame({"k": list(k_values), "WCSS": wcss}).set_index("k")
    metrics_df = pd.DataFrame(metrics_rows).set_index("k")

    print("\n--- Validation Metrics (k = 2..10) ---")
    print_table(metrics_df, float_fmt=".4f")

    # Best k per metric
    best_sil_k = metrics_df["Silhouette"].idxmax()
    best_db_k = metrics_df["Davies_Bouldin"].idxmin()
    best_ch_k = metrics_df["Calinski_Harabasz"].idxmax()

    print("\n--- Optimal k Summary ---")
    print(f"Max Silhouette Score        : k = {best_sil_k} ({metrics_df.loc[best_sil_k, 'Silhouette']:.4f})")
    print(f"Min Davies-Bouldin Index    : k = {best_db_k} ({metrics_df.loc[best_db_k, 'Davies_Bouldin']:.4f})")
    print(f"Max Calinski-Harabasz Index : k = {best_ch_k} ({metrics_df.loc[best_ch_k, 'Calinski_Harabasz']:.4f})")

    # Visualization: 2x2 Grid of Evaluation Metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # 1. Elbow Method (WCSS)
    ax = axes[0, 0]
    ax.plot(wcss_df.index, wcss_df["WCSS"], marker="o", lw=2.2, color=COLORS[0])
    ax.scatter([selected_k], [wcss_df.loc[selected_k, "WCSS"]], s=180, color=HILITE, zorder=5)
    ax.annotate(
        f"Selected k = {selected_k}\nWCSS = {wcss_df.loc[selected_k, 'WCSS']:,.0f}",
        xy=(selected_k, wcss_df.loc[selected_k, "WCSS"]),
        xytext=(selected_k + 1.2, wcss_df.loc[selected_k, "WCSS"] + 700),
        fontsize=10, weight="bold",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
    )
    ax.set_title("Elbow Method (WCSS)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.set_xticks(list(k_values))

    # 2. Silhouette Score
    ax = axes[0, 1]
    ax.plot(metrics_df.index, metrics_df["Silhouette"], marker="o", lw=2.2, color=COLORS[1])
    ax.axvline(best_sil_k, color=COLORS[1], linestyle="--", alpha=0.6, label=f"Best (k={best_sil_k})")
    ax.scatter([selected_k], [metrics_df.loc[selected_k, "Silhouette"]], s=150, color=HILITE, zorder=5, label=f"Selected (k={selected_k})")
    ax.set_title("Silhouette Score (Higher is Better)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Score")
    ax.set_xticks(list(metrics_df.index))
    ax.legend()

    # 3. Davies-Bouldin Index
    ax = axes[1, 0]
    ax.plot(metrics_df.index, metrics_df["Davies_Bouldin"], marker="s", lw=2.2, color=COLORS[2])
    ax.axvline(best_db_k, color=COLORS[2], linestyle="--", alpha=0.6, label=f"Best (k={best_db_k})")
    ax.scatter([selected_k], [metrics_df.loc[selected_k, "Davies_Bouldin"]], s=150, color=HILITE, zorder=5, label=f"Selected (k={selected_k})")
    ax.set_title("Davies-Bouldin Index (Lower is Better)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Index")
    ax.set_xticks(list(metrics_df.index))
    ax.legend()

    # 4. Calinski-Harabasz Index
    ax = axes[1, 1]
    ax.plot(metrics_df.index, metrics_df["Calinski_Harabasz"], marker="^", lw=2.2, color=COLORS[3])
    ax.axvline(best_ch_k, color=COLORS[3], linestyle="--", alpha=0.6, label=f"Best (k={best_ch_k})")
    ax.scatter([selected_k], [metrics_df.loc[selected_k, "Calinski_Harabasz"]], s=150, color=HILITE, zorder=5, label=f"Selected (k={selected_k})")
    ax.set_title("Calinski-Harabasz Index (Higher is Better)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Index")
    ax.set_xticks(list(metrics_df.index))
    ax.legend()

    plt.suptitle("K-Means Model Selection & Cluster Validation Metrics", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_or_show(fig, "01_kmeans_model_selection", out_dir)

    return wcss_df, metrics_df


# ============================================================
# 3. FIT FINAL K-MEANS MODEL & PLOT DISTRIBUTION
# ============================================================
def fit_final_kmeans(
    X: pd.DataFrame, n_clusters: int = 3, out_dir: Optional[Path] = None
) -> Tuple[KMeans, np.ndarray, pd.DataFrame, pd.DataFrame]:
    print_header(f"3. Final K-Means Model Details (k = {n_clusters})")

    final_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    final_labels = final_kmeans.fit_predict(X)

    sil = silhouette_score(X, final_labels)
    db = davies_bouldin_score(X, final_labels)
    ch = calinski_harabasz_score(X, final_labels)

    print(f"Algorithm           : {final_kmeans.algorithm}")
    print(f"Iterations to Conv. : {final_kmeans.n_iter_}")
    print(f"Final Inertia (WCSS): {final_kmeans.inertia_:,.4f}")
    print(f"Silhouette Score    : {sil:.4f}")
    print(f"Davies-Bouldin      : {db:.4f}")
    print(f"Calinski-Harabasz   : {ch:.4f}")

    # Distribution table
    counts = pd.Series(final_labels, name="Cluster").value_counts().sort_index()
    dist_df = pd.DataFrame({
        "Sample_Count": counts.values,
        "Percentage_%": (counts.values / len(final_labels)) * 100,
    }, index=counts.index)

    print("\n--- Sample Distribution Across Clusters ---")
    print_table(dist_df, float_fmt=".2f")

    # Plot cluster distribution with annotations
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    palette = sns.color_palette("viridis", n_clusters)
    bars = ax.bar(
        counts.index.astype(str),
        counts.values,
        color=palette,
        edgecolor="black",
        width=0.55,
        alpha=0.9,
    )

    for bar, pct in zip(bars, dist_df["Percentage_%"]):
        h = bar.get_height()
        ax.annotate(
            f"{int(h)} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    ax.set_title(f"Distribution of Patients Across {n_clusters} K-Means Clusters", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Cluster ID", fontsize=11)
    ax.set_ylabel("Number of Patients", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_or_show(fig, "02_cluster_distribution", out_dir)

    # Compute cluster means & standard deviations without fragmenting DataFrame
    df_clustered = X.assign(Cluster=final_labels)
    cluster_means = df_clustered.groupby("Cluster").mean()
    cluster_std = df_clustered.groupby("Cluster").std()

    return final_kmeans, final_labels, cluster_means, cluster_std


# ============================================================
# 4. IDENTIFY TOP DIFFERENTIATING FEATURES
# ============================================================
def identify_top_features(
    cluster_means: pd.DataFrame, cluster_std: pd.DataFrame, top_n: int = 30
) -> List[str]:
    print_header(f"4. Top {top_n} Differentiating Features (By Max-Min Mean Range)")

    mean_range = cluster_means.max(axis=0) - cluster_means.min(axis=0)
    sorted_ranges = mean_range.sort_values(ascending=False)
    top_features = sorted_ranges.head(top_n).index.tolist()

    # Combine Means and Range into a clean transposed summary
    summary_table = cluster_means[top_features].T.copy()
    summary_table.columns = [f"Cluster_{c}_Mean" for c in summary_table.columns]
    summary_table["Mean_Range"] = sorted_ranges.head(top_n)
    summary_table["Avg_Within_Std"] = cluster_std[top_features].mean(axis=0)

    print_table(summary_table, float_fmt=".4f")

    # Return full sorted list of features for downstream filtering
    return sorted_ranges.index.tolist()


# ============================================================
# 5. CLINICAL INTERPRETATION: MANIC & DEPRESSIVE SYMPTOMS
# ============================================================
def analyze_symptom_profiles(
    cluster_means: pd.DataFrame,
    sorted_features: List[str],
    out_dir: Optional[Path] = None,
) -> None:
    print_header("5. Clinical Symptom Profiling (Manic vs. Depressive)")

    manic_keywords = [
        "racing_thoughts", "inc_speech_rate_and_amount", "grandiosity",
        "distractibility", "persistent_elevated_mood", "dec_need_for_sleep",
        "high_risk_behaviours", "increased_sexual_interest", "agitation", "pma_mania",
    ]

    depressive_keywords = [
        "low_mood", "anhedonia", "worthless", "guilt", "emptiness",
        "suicidal_ideation", "suicidal_attempt", "poor_concentration",
        "fatigue", "sleep_disturbance", "appetite_disturbance",
        "psychomotor_agitation", "psychomotor_retardation",
        "loss_of_energy", "blunted_affect", "avolition",
    ]

    manic_features = [f for f in sorted_features if any(k in f for k in manic_keywords)]
    dep_features = [f for f in sorted_features if any(k in f for k in depressive_keywords)]

    print(f"\n--- Manic Symptom Features ({len(manic_features)} identified) ---")
    print_table(cluster_means[manic_features].T, float_fmt=".4f")

    print(f"\n--- Depressive Symptom Features ({len(dep_features)} identified) ---")
    print_table(cluster_means[dep_features].T, float_fmt=".4f")

    # Visualization: Side-by-Side Heatmaps for Manic & Depressive Symptoms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.5))

    sns.heatmap(
        cluster_means[manic_features].T,
        annot=True, fmt=".2f", cmap="OrRd",
        cbar_kws={"label": "Cluster Mean (Proportion)"},
        linewidths=0.5, ax=ax1,
    )
    ax1.set_title("Manic Symptoms Across Clusters", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Cluster ID", fontsize=11)
    ax1.set_ylabel("Manic Features", fontsize=11)

    sns.heatmap(
        cluster_means[dep_features].T,
        annot=True, fmt=".2f", cmap="PuBu",
        cbar_kws={"label": "Cluster Mean (Proportion)"},
        linewidths=0.5, ax=ax2,
    )
    ax2.set_title("Depressive Symptoms Across Clusters", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Cluster ID", fontsize=11)
    ax2.set_ylabel("Depressive Features", fontsize=11)

    plt.suptitle("K-Means Cluster Characterization: Manic & Depressive Profiles", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_or_show(fig, "03_clinical_symptom_heatmaps", out_dir)


# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================
def main():
    FILE_PATH = "/home/abhajkhan/Code/mca-tkmce/bipolar-disorder-severity-assessment-system/data/Bipolar_preprocessed.csv"
    OUT_DIR = Path("/home/abhajkhan/Code/mca-tkmce/bipolar-disorder-severity-assessment-system/notebooks/ppt_outputs/kmeans")

    # 1. Load Data
    X_km = load_and_clean_data(FILE_PATH)

    # 2. Evaluate k = 1..10 & Plot Elbow/Validation Curves
    evaluate_k_range(X_km, k_max=10, selected_k=3, out_dir=OUT_DIR)

    # 3. Fit Final Model (k=3) & Plot Cluster Sizes
    _, _, cluster_means, cluster_std = fit_final_kmeans(X_km, n_clusters=3, out_dir=OUT_DIR)

    # 4. Top 30 Differentiating Features
    all_sorted_features = identify_top_features(cluster_means, cluster_std, top_n=30)

    # 5. Clinical Symptom Heatmaps (Manic vs. Depressive)
    analyze_symptom_profiles(cluster_means, all_sorted_features, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()