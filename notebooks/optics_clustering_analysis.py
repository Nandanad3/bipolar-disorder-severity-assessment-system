import warnings
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
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

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
NOISE_COLOR = "#7f7f7f"


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
# 2. FIT OPTICS MODEL
# ============================================================
def fit_optics(
    df: pd.DataFrame, min_samples: int = 5, xi: float = 0.05, min_cluster_size: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, OPTICS]:
    print_header("2. Fitting OPTICS Clustering Model")

    # Select numerical columns and handle potential NaNs
    numerical_df = df.select_dtypes(include=np.number).copy()
    numerical_df = numerical_df.fillna(numerical_df.mean())

    optics_model = OPTICS(
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
        n_jobs=-1
    )
    optics_model.fit(numerical_df)

    labels = optics_model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    print(f"OPTICS Parameters   : min_samples={min_samples}, xi={xi}, min_cluster_size={min_cluster_size}")
    print(f"Estimated Clusters  : {n_clusters} (excluding noise)")
    print(f"Noise Points (-1)   : {n_noise} ({n_noise / len(labels) * 100:.2f}%)")

    # Distribution table
    counts = pd.Series(labels, name="Cluster_Label").value_counts().sort_index()
    dist_df = pd.DataFrame({
        "Sample_Count": counts.values,
        "Percentage_%": (counts.values / len(labels)) * 100,
    }, index=counts.index)
    print_table(dist_df, float_fmt=".2f")

    return df, numerical_df, labels, optics_model


# ============================================================
# 3. EVALUATE CLUSTERS (EXCLUDING NOISE)
# ============================================================
def evaluate_clusters(numerical_df: pd.DataFrame, labels: np.ndarray) -> None:
    print_header("3. Internal Cluster Validation Metrics (Excluding Noise)")

    ok_mask = labels != -1
    clustered_data = numerical_df[ok_mask]
    clustered_labels = labels[ok_mask]
    n_clusters = len(set(clustered_labels))

    if n_clusters > 1:
        sil = silhouette_score(clustered_data, clustered_labels)
        db = davies_bouldin_score(clustered_data, clustered_labels)
        ch = calinski_harabasz_score(clustered_data, clustered_labels)

        print(f"Silhouette Score        : {sil:.4f}  (Higher = better cohesion/separation)")
        print(f"Davies-Bouldin Index    : {db:.4f}  (Lower = better partitioning)")
        print(f"Calinski-Harabasz Index : {ch:.4f}  (Higher = denser, well-separated clusters)")
    elif n_clusters == 1:
        print("Only one cluster found (excluding noise), metrics cannot be computed.")
    else:
        print("No clusters found (only noise points).")


# ============================================================
# 4. VISUALIZATION & SUMMARY
# ============================================================
def generate_visualizations(
    numerical_df: pd.DataFrame, labels: np.ndarray, model: OPTICS, out_dir: Optional[Path] = None
) -> None:
    print_header("4. Generating Professional Visualizations")

    # Visualization 1: OPTICS Reachability Plot
    order = model.ordering_
    reach = model.reachability_[order].copy()
    reach[~np.isfinite(reach)] = np.nan
    lab_o = labels[order]

    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors = [NOISE_COLOR if l == -1 else COLORS[l % len(COLORS)] for l in lab_o]
    ax.bar(range(len(reach)), np.nan_to_num(reach), color=bar_colors, width=1.0)
    ax.set_title("OPTICS Reachability Plot (Grey = Noise)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Patients in OPTICS Ordering")
    ax.set_ylabel("Reachability Distance")
    ax.grid(False)
    plt.tight_layout()
    save_or_show(fig, "01_reachability", out_dir)

    # Visualization 2: Cluster Sizes Bar Chart
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_colors_counts = [NOISE_COLOR if idx == -1 else COLORS[idx % len(COLORS)] for idx in counts.index]
    bars = ax.bar(
        [("Noise (-1)" if idx == -1 else f"Cluster {idx}") for idx in counts.index],
        counts.values,
        color=bar_colors_counts,
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
    ax.set_title("OPTICS Cluster Sizes & Noise Points", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Patients")
    ax.set_ylim(0, counts.max() * 1.18)
    plt.tight_layout()
    save_or_show(fig, "02_sizes_with_noise", out_dir)

    # Visualization 3: PCA 2D Projection Scatter Plot
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(numerical_df)
    var_exp = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for idx in counts.index:
        mask = labels == idx
        color = NOISE_COLOR if idx == -1 else COLORS[idx % len(COLORS)]
        label_name = "Noise Points" if idx == -1 else f"Cluster {idx} (n={counts[idx]})"
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=55, alpha=0.8, color=color, edgecolors="k", linewidth=0.4,
            label=label_name,
        )
    ax.set_title("PCA 2D Projection of OPTICS Clusters", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}% Variance)")
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}% Variance)")
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    save_or_show(fig, "04_pca_scatter", out_dir)


# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================
def main():
    FILE_PATH = "/home/abhajkhan/Code/mca-tkmce/bipolar-disorder-severity-assessment-system/data/Bipolar_preprocessed.csv"
    OUT_DIR = Path("/home/abhajkhan/Code/mca-tkmce/bipolar-disorder-severity-assessment-system/notebooks/ppt_outputs/optics")

    # 1. Load Data
    df = load_dataset(FILE_PATH)

    # 2. Fit OPTICS Model
    df, numerical_df, labels, model = fit_optics(df, min_samples=5, xi=0.05, min_cluster_size=5)

    # 3. Evaluate Validation Metrics
    evaluate_clusters(numerical_df, labels)

    # 4. Generate Visualizations
    generate_visualizations(numerical_df, labels, model, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()