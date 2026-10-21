import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.stats import f_oneway

# Suppress warnings for clean console output
warnings.filterwarnings("ignore")

# Global styling and color palette
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


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
def load_dataset(file_path: str) -> pd.DataFrame:
    print_header("1. Loading Preprocessed Dataset")
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(path).copy()
    print(f"Loaded '{path.name}' successfully.")
    print(f"Dataset Shape : {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ============================================================
# 2. FIT DBSCAN MODEL
# ============================================================
def fit_dbscan(
    df: pd.DataFrame, eps: float = 4.0, min_samples: int = 5
) -> Tuple[pd.DataFrame, np.ndarray]:
    print_header("2. Fitting DBSCAN Clustering Model")
    
    # Exclude any pre-existing cluster columns
    exclude_cols = [
        "DBSCAN_Cluster", "DBSCAN_Cluster_with_noise_assigned",
        "Cluster", "GMM_Cluster", "Hierarchical_Cluster"
    ]
    X = df.drop(columns=exclude_cols, errors="ignore").copy()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X)
    
    df["DBSCAN_Cluster"] = clusters

    n_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
    n_noise = np.sum(clusters == -1)

    print(f"DBSCAN Parameters  : eps = {eps}, min_samples = {min_samples}")
    print(f"Clusters Discovered: {n_clusters} (excluding noise)")
    print(f"Noise Points (-1)  : {n_noise} ({n_noise / len(clusters) * 100:.2f}%)")

    # Cluster distribution table
    counts = pd.Series(clusters, name="Cluster_Label").value_counts().sort_index()
    dist_df = pd.DataFrame({
        "Sample_Count": counts.values,
        "Percentage_%": (counts.values / len(clusters)) * 100,
    }, index=counts.index)
    print_table(dist_df, float_fmt=".2f")

    return df, clusters


# ============================================================
# 3. EVALUATE CLUSTERS (EXCLUDING NOISE)
# ============================================================
def evaluate_clusters(df: pd.DataFrame, clusters: np.ndarray) -> None:
    print_header("3. Internal Cluster Validation Metrics (Excluding Noise)")

    valid_indices = clusters != -1
    exclude_cols = ["DBSCAN_Cluster", "DBSCAN_Cluster_with_noise_assigned", "Cluster"]
    filtered_X = df.drop(columns=exclude_cols, errors="ignore")[valid_indices]
    filtered_clusters = clusters[valid_indices]

    if len(np.unique(filtered_clusters)) < 2:
        print("Warning: Fewer than 2 clusters remain after excluding noise. Metrics cannot be computed.")
        return

    sil = silhouette_score(filtered_X, filtered_clusters)
    db = davies_bouldin_score(filtered_X, filtered_clusters)
    ch = calinski_harabasz_score(filtered_X, filtered_clusters)

    print(f"Silhouette Score        : {sil:.4f}  (Higher = better cohesion/separation)")
    print(f"Davies-Bouldin Index    : {db:.4f}  (Lower = better partitioning)")
    print(f"Calinski-Harabasz Index : {ch:.4f}  (Higher = denser, well-separated clusters)")


# ============================================================
# 4. ASSIGN NOISE POINTS USING K-NEAREST NEIGHBORS
# ============================================================
def assign_noise_points(df: pd.DataFrame, clusters: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    print_header("4. Assigning Noise Points via Nearest Neighbors (k=1)")

    new_clusters = clusters.copy()
    noise_indices = np.where(clusters == -1)[0]
    clustered_indices = np.where(clusters != -1)[0]

    exclude_cols = ["DBSCAN_Cluster", "DBSCAN_Cluster_with_noise_assigned", "Cluster"]
    X = df.drop(columns=exclude_cols, errors="ignore").copy()

    if len(noise_indices) > 0 and len(clustered_indices) > 0:
        X_clustered = X.iloc[clustered_indices]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_clustered)
        _, indices = nbrs.kneighbors(X.iloc[noise_indices])

        for i, noise_idx in enumerate(noise_indices):
            nearest_original_idx = clustered_indices[indices[i][0]]
            new_clusters[noise_idx] = clusters[nearest_original_idx]

        print(f"Successfully reassigned {len(noise_indices)} noise points to their nearest clusters.")
    else:
        print("No noise points found or insufficient clustered points.")

    df["DBSCAN_Cluster_with_noise_assigned"] = new_clusters

    updated_counts = pd.Series(new_clusters, name="Assigned_Cluster").value_counts().sort_index()
    print("\n--- Updated Cluster Sizes (Noise Assigned) ---")
    print(updated_counts)

    return df, new_clusters


# ============================================================
# 5. FEATURE DIFFERENTIATION & VISUALIZATIONS
# ============================================================
def analyze_and_visualize(
    df: pd.DataFrame, new_clusters: np.ndarray, out_dir: Optional[Path] = None
) -> None:
    print_header("5. Feature Differentiation & Visualizations")

    exclude_cols = ["DBSCAN_Cluster", "DBSCAN_Cluster_with_noise_assigned", "Cluster"]
    X = df.drop(columns=exclude_cols, errors="ignore").select_dtypes(include=[np.number])

    cluster_means = X.assign(Cluster=new_clusters).groupby("Cluster").mean()
    overall_means = X.mean()

    # Absolute difference from overall mean averaged across clusters
    feature_diff = cluster_means.subtract(overall_means, axis=1).abs().mean(axis=0)
    top_10 = feature_diff.nlargest(10)

    print("\n--- Top 10 Differentiating Features Across Assigned Clusters ---")
    print_table(top_10, float_fmt=".4f")

    # Visualization 1: Cluster Distribution Bar Chart
    counts = pd.Series(new_clusters).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        [f"Cluster {c}" for c in counts.index],
        counts.values,
        color=COLORS[:len(counts)],
        edgecolor="black",
        width=0.55,
        alpha=0.9,
    )
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{int(h)}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontweight="bold", fontsize=10,
        )
    ax.set_title("DBSCAN Cluster Sizes (Noise Assigned)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Patients")
    ax.set_ylim(0, counts.max() * 1.18)
    plt.tight_layout()
    save_or_show(fig, "01_cluster_sizes_assigned", out_dir)

    # Visualization 2: PCA 2D Projection Scatter Plot
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for c in counts.index:
        mask = new_clusters == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=55, alpha=0.8, color=COLORS[c % len(COLORS)], edgecolors="k", linewidth=0.4,
            label=f"Cluster {c} (n={counts[c]})",
        )
    ax.set_title("PCA 2D Projection of DBSCAN Clusters", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}% Variance)")
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}% Variance)")
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    save_or_show(fig, "02_pca_scatter", out_dir)

    # Visualization 3: Heatmap of Top Differentiating Features
    top_10_cols = top_10.index.tolist()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.heatmap(
        cluster_means[top_10_cols].T,
        annot=True, fmt=".2f", cmap="YlGnBu",
        xticklabels=[f"Cluster {c}\n(n={counts[c]})" for c in counts.index],
        cbar_kws={"label": "Cluster Mean Value"},
        linewidths=0.5, ax=ax,
    )
    ax.set_title("DBSCAN: Top 10 Differentiating Features", fontsize=13, fontweight="bold")
    ax.set_ylabel("Features")
    plt.tight_layout()
    save_or_show(fig, "03_profile_heatmap", out_dir)


# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================
def main():
    FILE_PATH = "/home/abhajkhan/Code/mca-tkmce/bipolar-disorder-severity-assessment-system/data/Bipolar_preprocessed.csv"
    OUT_DIR = Path("/home/abhajkhan/Code/mca-tkmce/bipolar-disorder-severity-assessment-system/notebooks/ppt_outputs/dbscan")

    # 1. Load Data
    df = load_dataset(FILE_PATH)

    # 2. Fit DBSCAN
    df, clusters = fit_dbscan(df, eps=4.0, min_samples=5)

    # 3. Evaluate Metrics
    evaluate_clusters(df, clusters)

    # 4. Assign Noise Points via k-NN
    df, new_clusters = assign_noise_points(df, clusters)

    # 5. Analyze Differentiating Features & Generate Visualizations
    analyze_and_visualize(df, new_clusters, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()