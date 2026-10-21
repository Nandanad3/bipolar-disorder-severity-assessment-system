import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.stats import f_oneway

# Suppress convergence and syntax warnings for cleaner console output
warnings.filterwarnings("ignore")

# Set global visualization style
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


# ============================================================
# HELPER UTILITIES FOR CONSOLE PRINTING
# ============================================================
def print_header(title: str, char: str = "=") -> None:
    """Prints a formatted section header to the console."""
    width = 75
    print(f"\n{char * width}")
    print(f" {title.upper()}")
    print(f"{char * width}")


def print_table(df: pd.DataFrame, float_fmt: str = ".4f") -> None:
    """Prints a pandas DataFrame cleanly in pure Python environments."""
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 1000,
        "display.float_format", f"{{:{float_fmt}}}".format,
    ):
        print(df)


# ============================================================
# 1. LOAD PREPROCESSED DATASET
# ============================================================
def load_dataset(file_path: str) -> pd.DataFrame:
    print_header("1. Loading Preprocessed Dataset")
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(path)
    print(f"Successfully loaded dataset from: {path.name}")
    print(f"Dataset Shape : {df.shape[0]} rows × {df.shape[1]} columns")
    
    X_gmm = df.drop(columns=["GMM_Cluster"], errors="ignore")
    return X_gmm


# ============================================================
# 2. GMM MODEL SELECTION (AIC / BIC)
# ============================================================
def evaluate_gmm_components(
    X: pd.DataFrame, max_k: int = 10, random_state: int = 42
) -> pd.DataFrame:
    print_header("2. GMM Model Selection (AIC / BIC)")
    
    results = []
    for k in range(1, max_k + 1):
        model = GaussianMixture(n_components=k, random_state=random_state)
        model.fit(X)
        results.append({
            "n_components": k,
            "AIC": model.aic(X),
            "BIC": model.bic(X),
            "Log_Likelihood": model.lower_bound_,
            "Converged": model.converged_,
            "Iterations": model.n_iter_,
        })

    results_df = pd.DataFrame(results)
    print_table(results_df, float_fmt=".2f")

    min_aic_row = results_df.loc[results_df["AIC"].idxmin()]
    min_bic_row = results_df.loc[results_df["BIC"].idxmin()]

    print("\n--- Optimal Component Summary ---")
    print(f"Minimum AIC : k = {int(min_aic_row['n_components'])} (AIC = {min_aic_row['AIC']:,.2f})")
    print(f"Minimum BIC : k = {int(min_bic_row['n_components'])} (BIC = {min_bic_row['BIC']:,.2f})")

    # Visualization: AIC & BIC Elbow Curve
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    
    ax1.plot(results_df["n_components"], results_df["AIC"], marker="o", linewidth=2, label="AIC", color="#1f77b4")
    ax1.plot(results_df["n_components"], results_df["BIC"], marker="s", linewidth=2, label="BIC", color="#ff7f0e")
    
    ax1.axvline(min_bic_row["n_components"], color="#ff7f0e", linestyle="--", alpha=0.6, label=f"Min BIC (k={int(min_bic_row['n_components'])})")
    ax1.axvline(3, color="#2ca02c", linestyle=":", linewidth=2, label="Selected Model (k=3)")

    ax1.set_title("GMM Model Selection: AIC & BIC across Components", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Number of Components (k)", fontsize=11)
    ax1.set_ylabel("Information Criterion Score (Lower is Better)", fontsize=11)
    ax1.set_xticks(range(1, max_k + 1))
    ax1.legend()
    plt.tight_layout()
    plt.show()

    return results_df


# ============================================================
# 3. FIT FINAL GMM & EVALUATE CLUSTERS
# ============================================================
def fit_and_evaluate_gmm(
    X: pd.DataFrame, n_components: int = 3, random_state: int = 42
) -> Tuple[GaussianMixture, np.ndarray, np.ndarray]:
    print_header(f"3. Final GMM Diagnostics & Validation (k = {n_components})")

    final_gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    final_gmm.fit(X)

    labels = final_gmm.predict(X)
    probs = final_gmm.predict_proba(X)

    # Model Parameters & Convergence
    print(f"Covariance Type : {final_gmm.covariance_type}")
    print(f"Converged       : {final_gmm.converged_} (in {final_gmm.n_iter_} iterations)")
    print(f"Log-Likelihood  : {final_gmm.lower_bound_:.4f}")

    # Cluster Distribution Summary
    counts = pd.Series(labels).value_counts().sort_index()
    cluster_summary = pd.DataFrame({
        "Cluster": counts.index,
        "Sample_Count": counts.values,
        "Percentage_%": (counts.values / len(labels)) * 100,
        "Mixture_Weight": final_gmm.weights_,
    }).set_index("Cluster")
    
    print("\n--- Cluster Size & Weight Distribution ---")
    print_table(cluster_summary, float_fmt=".4f")

    # Cluster Validation Metrics
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    print("\n--- Internal Cluster Validation Metrics ---")
    print(f"Silhouette Score        : {sil:>10.4f}  (Higher = better cohesion/separation)")
    print(f"Davies-Bouldin Index    : {db:>10.4f}  (Lower  = better partitioning)")
    print(f"Calinski-Harabasz Index : {ch:>10.4f}  (Higher = denser, well-separated clusters)")

    # Membership Uncertainty Analysis
    max_probs = probs.max(axis=1)
    bins = [0, 0.50, 0.70, 0.90, 1.00]
    bin_labels = ["<=50% (Low)", "50-70% (Moderate)", "70-90% (High)", "90-100% (Very High)"]
    
    conf_cuts = pd.cut(max_probs, bins=bins, labels=bin_labels, include_lowest=True)
    conf_counts = conf_cuts.value_counts().sort_index()
    
    conf_df = pd.DataFrame({
        "Patient_Count": conf_counts.values,
        "Percentage_%": (conf_counts.values / len(max_probs)) * 100
    }, index=conf_counts.index)

    print("\n--- Assignment Confidence Breakdown ---")
    print_table(conf_df, float_fmt=".2f")

    return final_gmm, labels, probs


# ============================================================
# 4. IDENTIFY TOP DIFFERENTIATING FEATURES
# ============================================================
def analyze_differentiating_features(
    X: pd.DataFrame, labels: np.ndarray, top_n: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print_header(f"4. Top {top_n} Differentiating Features Across Clusters")

    df_gmm = X.copy()
    df_gmm["GMM_Cluster"] = labels
    feature_cols = [c for c in X.columns]

    cluster_means = df_gmm.groupby("GMM_Cluster")[feature_cols].mean()
    feature_spread = cluster_means.max() - cluster_means.min()

    stats_df = pd.DataFrame({
        "Mean_Range": feature_spread,
        "Mean_Std": cluster_means.std(axis=0),
        "Overall_Mean": df_gmm[feature_cols].mean(),
    })

    stats_df["Rel_Variation_%"] = (
        stats_df["Mean_Range"] / (stats_df["Overall_Mean"].abs() + 1e-8)
    ) * 100

    # Calculate One-Way ANOVA F-Stat for each feature
    f_scores = {}
    for col in feature_cols:
        groups = [group[col].values for _, group in df_gmm.groupby("GMM_Cluster")]
        f_stat, _ = f_oneway(*groups)
        f_scores[col] = f_stat

    stats_df["ANOVA_F_Score"] = pd.Series(f_scores)
    stats_df = stats_df.sort_values(by="ANOVA_F_Score", ascending=False)
    stats_df["Global_Rank"] = range(1, len(stats_df) + 1)

    print_table(stats_df.head(top_n), float_fmt=".4f")

    # Visualization: Grid of Barplots with Direct Value Annotations
    top_features = stats_df.head(top_n).index.tolist()
    n_cols = 2
    n_rows = (top_n + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.2 * n_rows))
    axes = axes.flatten()
    palette = sns.color_palette("viridis", len(cluster_means))

    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        means = cluster_means[feature]
        bars = ax.bar(
            means.index.astype(str),
            means.values,
            color=palette,
            edgecolor="black",
            alpha=0.85,
            width=0.55,
        )

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        f_val = stats_df.loc[feature, "ANOVA_F_Score"]
        ax.set_title(f"{feature}\n(ANOVA F: {f_val:,.1f})", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Mean Value", fontsize=9.5)
        ax.set_xlabel("GMM Cluster", fontsize=9.5)
        ax.set_ylim(0, max(means.values) * 1.2 + 0.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    for idx in range(len(top_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(
        f"Top {top_n} Differentiating Features Across GMM Clusters",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.show()

    return stats_df, cluster_means


# ============================================================
# 5. CLINICAL CHARACTERIZATION: DEPRESSIVE & MOOD FEATURES
# ============================================================
def analyze_clinical_features(
    X: pd.DataFrame, stats_df: pd.DataFrame, cluster_means: pd.DataFrame
) -> None:
    print_header("5. Detailed Cluster Characterization: Clinical & Depressive Features")

    # Target keywords used to filter relevant clinical/depressive indicators
    keywords = [
        "depress", "elevated_mood", "sleep", "sexual", "agitation",
        "low_mood", "anhedonia", "guilt", "worthless", "concentration", "suicidal"
    ]
    
    dep_features: List[str] = [
        col for col in X.columns 
        if any(k in col.lower() for k in keywords) and "duration" not in col.lower()
    ]

    print(f"Identified {len(dep_features)} clinical/depressive features out of {X.shape[1]} total features.")

    # Breakdown Table
    dep_stats = stats_df.loc[dep_features, ["ANOVA_F_Score", "Global_Rank", "Rel_Variation_%"]].sort_values(
        by="ANOVA_F_Score", ascending=False
    )
    print("\n--- Statistical Breakdown & Global Ranking ---")
    print_table(dep_stats, float_fmt=".2f")

    # Calculate Composite Z-Score across the selected clinical features
    dep_means = cluster_means[dep_features].copy()
    z_scores = (dep_means - dep_means.mean()) / (dep_means.std() + 1e-8)
    dep_means["Composite_Depression_Score (Z)"] = z_scores.mean(axis=1)

    print("\n--- Cluster Mean Values & Composite Score ---")
    # Transposed for much cleaner readability in terminal output
    print_table(dep_means.T, float_fmt=".4f")

    # Visualization: Clinical Profile Heatmap across Clusters
    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [3.5, 1.2]}
    )

    # Sort features by F-score for the heatmap
    sorted_dep_cols = dep_stats.index.tolist()
    heatmap_data = cluster_means[sorted_dep_cols].T

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Cluster Mean Value"},
        ax=ax_heat,
        linewidths=0.5,
    )
    ax_heat.set_title("Clinical Feature Means by Cluster", fontsize=13, fontweight="bold")
    ax_heat.set_xlabel("GMM Cluster", fontsize=11)
    ax_heat.set_ylabel("Clinical / Depressive Features", fontsize=11)

    # Bar plot for Composite Z-Score
    comp_scores = dep_means["Composite_Depression_Score (Z)"]
    colors = ["#d95f02" if v > 0 else "#7570b3" for v in comp_scores.values]
    ax_bar.barh(comp_scores.index.astype(str), comp_scores.values, color=colors, edgecolor="black", height=0.4)
    ax_bar.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_bar.set_title("Composite Z-Score", fontsize=13, fontweight="bold")
    ax_bar.set_xlabel("Mean Z-Score", fontsize=11)
    ax_bar.set_ylabel("GMM Cluster", fontsize=11)

    for idx, val in enumerate(comp_scores.values):
        ax_bar.text(
            val + (0.002 if val >= 0 else -0.002),
            idx,
            f"{val:+.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontweight="bold",
            fontsize=9,
        )

    ax_bar.set_xlim(comp_scores.min() * 1.4, comp_scores.max() * 1.4)
    plt.suptitle("Clinical & Depressive Feature Profiling Across Clusters", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================
def main():
    FILE_PATH = "/home/abhajkhan/Code/mca-tkmce/bipolar-disorder-severity-assessment-system/data/Bipolar_preprocessed.csv"
    
    # 1. Load Data
    X_gmm = load_dataset(FILE_PATH)

    # 2. Model Selection (k=1 to 10)
    evaluate_gmm_components(X_gmm, max_k=10, random_state=42)

    # 3. Fit Final Model (k=3) & Run Diagnostics
    _, gmm_labels, _ = fit_and_evaluate_gmm(X_gmm, n_components=3, random_state=42)

    # 4. Top Differentiating Features (ANOVA F-Test)
    stats_df, cluster_means = analyze_differentiating_features(X_gmm, gmm_labels, top_n=10)

    # 5. Clinical / Depressive Feature Characterization
    analyze_clinical_features(X_gmm, stats_df, cluster_means)


if __name__ == "__main__":
    main()