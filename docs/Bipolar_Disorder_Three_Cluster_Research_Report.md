# Unsupervised Clustering-Based Stratification of Bipolar Disorder Patients

## Research Report

### 1. Introduction

Bipolar disorder is a heterogeneous psychiatric condition in which
patients can differ considerably in symptom presentation, illness
duration, episode frequency, functional impairment, comorbidities, and
treatment history. A single severity scale may therefore fail to capture
the different clinical patterns present within a patient population.

The objective of this project was to investigate whether unsupervised
machine learning could identify meaningful subgroups of patients with
bipolar disorder based on their clinical characteristics. Three
clustering approaches were studied: K-Means, Agglomerative Hierarchical
Clustering, and Gaussian Mixture Models (GMM).

The primary objective was not simply to obtain a mathematically
separated partition of the data. The intention was to identify a
practical number of patient groups that could be interpreted clinically
and used as a foundation for a future bipolar-disorder severity
assessment system.

After preprocessing, the final dataset contained 342 unique patient
records and 168 numerical features. The clustering experiments were
performed using the processed clinical variables, including encoded
symptom, illness-course, treatment, functioning, and demographic
information.

A three-cluster solution was selected as the final exploratory
stratification because it provided the most useful balance between
statistical separation, clinical interpretability, model agreement, and
practical usefulness for the proposed system.

---

# 2. Data Quality and Its Effect on Clustering

A major challenge encountered during this study was the quality and
completeness of the available clinical dataset.

The original dataset contained 367 records and 430 columns. After
clinical feature selection and removal of duplicate records, 342 unique
patient records remained. Although the resulting dataset contains a
broad range of clinically relevant variables, the number of observations
is relatively small compared with the dimensionality of the final
feature space.

The final clustering matrix contains:

- 342 patients
- 168 features
- 23 numerical variables
- 64 `_Unknown` indicator variables

The dataset also contained substantial missing information. Among the
initially selected 87 clinical variables:

- 38 variables had less than 20% missingness.
- 27 variables had between 20% and less than 40% missingness.
- 22 variables had at least 40% missingness.

Several clinically important variables therefore contained substantial
amounts of unavailable information.

This created a significant challenge for clustering because clustering
algorithms depend directly on the geometry of the feature space. When a
considerable proportion of observations contains missing or unknown
clinical information, the distances between patients can reflect both
their actual clinical characteristics and the completeness of their
records.

This issue was particularly visible after categorical encoding. Missing
categorical information was retained as an explicit `Unknown` category
where appropriate. Consequently, 64 `_Unknown` indicator variables were
present in the final 168-dimensional dataset.

For example, some of the variables with the highest proportions of
unknown values included:

Variable Unknown proportion

---

guilt_Unknown 42.98%
emptiness_Unknown 42.69%
worthless_Unknown 42.69%
self_harm_ideation_Unknown 42.40%
anhedonia_Unknown 42.11%
poor_concentration_Unknown 42.11%
antipsychotic_Unknown 42.11%
suicidal_attempt_Unknown 41.81%
psychotic_symptoms_index_episode_Unknown 41.81%
executive_dysfunction_Unknown 41.81%
low_mood_Unknown 41.23%

Therefore, the validation metrics should not be interpreted as if they
were obtained from a large, clean, completely observed clinical dataset.

The variation observed between clustering metrics is partly a
consequence of the difficult structure of the data. In a
high-dimensional clinical dataset with substantial missingness,
different internal validation measures can emphasize different aspects
of separation. Some metrics favor compactness, others emphasize
separation, and others respond to within-cluster dispersion. The
resulting scores therefore do not necessarily identify one identical
cluster count.

This is an important limitation of the current experiment rather than a
failure of the clustering methodology.

---

# 3. Data Preprocessing

Before clustering, the raw data underwent several preprocessing stages.

## 3.1 Duplicate removal

Twenty-five duplicate records were identified and removed.

The number of observations therefore changed from:

`367 → 342`

Removing duplicates was necessary to prevent repeated patients from
receiving disproportionately high influence during clustering.

## 3.2 Numerical variables

Twenty-three numerical variables were identified.

These included variables representing:

- Age and age at onset.
- Number and duration of episodes.
- Manic, hypomanic, mixed, and depressive episodes.
- Percentage of time with symptoms.
- Interepisodic duration.
- Time spent euthymic.
- Mixed-episode prominence.
- Mania/depression ratios.
- Insight.
- Occupational functioning.
- Inpatient-care history.
- Clinical severity scores.
- Lithium-treatment response.

Missing numerical values were handled using `IterativeImputer` with
`random_state=42`.

The imputation procedure produced a complete numerical matrix for
subsequent processing.

## 3.3 Categorical variables

Categorical variables required additional domain-specific processing.

In particular, psychotic-symptom variables were examined using clinical
skip logic. Where `ever_psychotic` was recorded as `No`, missing child
psychotic-symptom fields were recoded as `No` rather than being treated
as unknown.

Remaining categorical missing values were represented using the explicit
`Unknown` category.

This distinction was important because a symptom being absent is
different from the symptom simply not being recorded.

## 3.4 Encoding

Categorical variables were transformed using one-hot encoding with
`drop_first=True`.

This transformed the clinical categorical information into numerical
indicators suitable for the clustering algorithms.

## 3.5 Scaling

The numerical variables were scaled using Min-Max scaling so that
numerical features with larger original ranges would not dominate the
distance calculations.

The final dataset used for the principal K-Means and GMM experiments
contained:

`342 × 168`

features and observations.

---

# 4. Clustering Algorithms

Three clustering approaches were investigated.

## 4.1 K-Means

K-Means partitions the dataset into k clusters by assigning each
observation to the nearest centroid and repeatedly updating the
centroids until convergence.

The final configuration was:

- Number of clusters: 3
- `random_state = 42`
- `n_init = auto`
- Lloyd algorithm

The three-cluster solution contained:

Cluster Number of patients Percentage

---

Cluster 0 41 11.99%
Cluster 1 106 30.99%
Cluster 2 195 57.02%
Total 342 100%

The final model converged in six iterations.

The final within-cluster sum of squares was:

`4096.308739`

---

## 4.2 Agglomerative Hierarchical Clustering

Hierarchical clustering was implemented using Agglomerative Clustering
with:

- Ward linkage
- Euclidean distance
- Three clusters in the final exploratory solution

The dendrogram was used to examine the hierarchical structure of the
patient population and identify a practical cutting point.

The transition associated with the three-cluster solution occurred at a
linkage distance of approximately:

`36.623225`

Hierarchical clustering provided a useful complementary perspective
because it does not begin by assigning observations to fixed centroids.
Instead, it progressively merges observations and groups according to
the resulting hierarchical structure.

---

## 4.3 Gaussian Mixture Model

GMM models the data as a mixture of Gaussian probability distributions.

The final configuration used:

- 3 components
- Full covariance
- `random_state = 42`

The model converged in two iterations.

The fitted mixture weights were approximately:

- Component 0: 0.1199
- Component 1: 0.3099
- Component 2: 0.5702

The resulting component sizes were again:

`41 / 106 / 195`

This agreement with K-Means was an important result in the selection of
the final three-cluster structure.

---

# 5. Selection of Three Clusters

The selection of three clusters was based on a combination of
quantitative evaluation, model agreement, cluster interpretability, and
the intended purpose of the project.

It is important to emphasize that the objective was to obtain a
clinically useful exploratory stratification rather than to maximize one
mathematical index in isolation.

## 5.1 Elbow analysis

For K-Means, the WCSS values were:

     k       WCSS

---

     1   7107.955
     2   5248.706
     3   4096.309
     4   3541.841
     5   3557.863
     6   3211.391
     7   3186.182
     8   3074.505
     9   3012.015
    10   3001.342

The largest reductions in WCSS occur in the early part of the curve,
particularly between one, two, and three clusters.

The three-cluster solution therefore represents a useful point where
substantial within-cluster variance has already been reduced without
introducing a large number of small groups.

Increasing the number of clusters beyond three does not produce a
comparable improvement in the overall clustering structure. In
particular, the internal validation measures deteriorate substantially
for several of the higher-k solutions.

---

## 5.2 Silhouette analysis

The Silhouette scores for K-Means were:

     k   Silhouette

---

     2     0.290191
     3     0.265246
     4     0.270647
     5     0.163465
     6     0.195831
     7     0.166504
     8     0.178720
     9     0.144541
    10     0.138220

The three-cluster solution produces a Silhouette score of `0.265246`.

Although the score varies across k, the important observation is the
substantial deterioration after the four-cluster region, particularly
from k=5 onward.

Thus, increasing the number of groups to five or more does not provide a
stronger and more useful representation of the dataset.

The moderate rather than high Silhouette values are consistent with the
nature of the dataset. The patient population contains overlapping
clinical characteristics, substantial missingness, and high-dimensional
encoded variables. Strongly separated spherical clusters should
therefore not be expected.

---

## 5.3 Davies-Bouldin analysis

The Davies-Bouldin index for the three-cluster K-Means solution is:

`1.432613`

The values across k were:

     k   Davies-Bouldin

---

     2         1.515143
     3         1.432613
     4         1.384013
     5         2.353396
     6         2.247889
     7         2.230106
     8         2.384439
     9         2.597845
    10         2.716281

The index is relatively stable across the low-k region and deteriorates
sharply after four clusters.

This supports retaining a small number of clinically interpretable
groups rather than fragmenting the population into many small clusters.

---

## 5.4 Calinski-Harabasz analysis

The Calinski-Harabasz score reaches its maximum at three clusters:

`124.618042`

The values around the relevant region are:

    k   Calinski-Harabasz

---

    2          120.438207
    3          124.618042
    4          113.438811
    5           84.065992
    6           81.537575

This is an important result for the final selection.

Among the candidate low-dimensional segmentations, the three-cluster
solution provides the strongest Calinski-Harabasz separation/compactness
trade-off observed in the K-Means experiment.

Therefore, the Calinski-Harabasz criterion directly supports the
selection of three clusters.

---

# 6. Agreement Between K-Means and GMM

One of the strongest pieces of evidence supporting the three-cluster
solution is the agreement between two fundamentally different clustering
approaches.

K-Means uses centroid-based hard assignments, whereas GMM estimates
Gaussian components using a covariance-based probabilistic model.

Despite this difference, both methods produced exactly the same
three-cluster partition.

The comparison produced:

- Adjusted Rand Index = `1.000000`
- Normalized Mutual Information = `1.000000`

The contingency matrix was perfectly diagonal:

K-Means / GMM Component 0 Component 1 Component 2

---

Cluster 0 41 0 0
Cluster 1 0 106 0
Cluster 2 0 0 195

This means that the three groups are not an artifact of one particular
clustering algorithm.

Two different modeling assumptions independently converged to the same
patient partition.

This substantially increases confidence in using the three-cluster
structure for the exploratory objective of this project.

---

# 7. Cluster Stability

Cluster stability was investigated by repeating the K-Means and GMM
experiments using multiple random seeds.

The resulting ARI values relative to the seed-42 reference were:

    Seed    K-Means        GMM

---

       0   0.989532   0.989532
       1   0.989532   0.989532
       2   1.000000   1.000000
       3   1.000000   1.000000
       4   1.000000   1.000000
       5   0.989532   0.989532
      10   0.989532   0.989532
      20   1.000000   1.000000
      42   1.000000   1.000000
     100   0.699243   0.699243

The mean ARI was:

`0.965737`

for both K-Means and GMM.

The results show strong agreement across most initializations. The lower
value at seed 100 demonstrates that some sensitivity remains.

This variation should be interpreted in the context of the dataset
quality. The current dataset contains only 342 observations against 168
final features, substantial missingness, and a large number of explicit
unknown indicators. Such a high-dimensional and incomplete
representation can make cluster boundaries sensitive to initialization
and feature composition.

Therefore, the observed variation in validation scores and
initialization stability is not unexpected for the available data.

With a larger, cleaner, and more consistently recorded dataset, the
clustering boundaries are expected to become more robust and the
internal validation scores can be evaluated with greater confidence.

---

# 8. Clinical Interpretation of the Three Clusters

The final three clusters show meaningful differences in the recorded
clinical characteristics.

## Cluster 0: Low Recorded Symptom Information / High Unknown Documentation

Cluster 0 contains 41 patients.

The strongest distinguishing characteristics are the high prevalence of
`_Unknown` values across many symptom variables.

Examples include:

- `racing_thoughts_Unknown = 1.0000`
- `inc_speech_rate_and_amount_Unknown = 1.0000`
- `grandiosity_Unknown = 1.0000`
- `distractibility_Unknown = 1.0000`
- `persistent_irritability_Unknown = 1.0000`
- `persistent_elevated_mood_Unknown = 1.0000`
- `hospitalized_Unknown = 1.0000`
- `agitation_Unknown = 1.0000`

Recorded positive symptom values are comparatively low.

This group should therefore be interpreted as a patient group with
relatively low recorded symptom presence but substantial missing/unknown
clinical documentation.

It should not be directly interpreted as a low-severity or healthy group
because the absence of recorded symptoms does not necessarily mean that
symptoms were absent.

---

## Cluster 1: Predominantly Manic Symptom Pattern

Cluster 1 contains 106 patients.

This group shows high recorded prevalence of manic-related
characteristics, including:

- Persistent irritability.
- Decreased need for sleep.
- Increased speech rate and amount.
- Psychomotor agitation.
- Elevated mood.
- Distractibility.
- Grandiosity.
- Racing thoughts.
- High-risk behavior.

For example:

- `persistent_irritability_Yes = 0.9906`
- `dec_need_for_sleep_Yes = 0.9906`
- `inc_speech_rate_and_amount_Yes = 0.9528`
- `pma_mania_Yes = 0.9623`
- `persistent_elevated_mood_Yes = 0.9340`

At the same time, recorded depressive symptoms are relatively low.

This cluster can therefore be characterized as a predominantly recorded
manic/hypomanic symptom pattern.

---

## Cluster 2: High Recorded Combined Symptom Burden

Cluster 2 contains 195 patients and is the largest group.

This group exhibits substantial recorded manic symptoms as well as
substantial depressive symptoms.

Manic-related features include:

- Persistent irritability.
- Decreased need for sleep.
- Increased speech.
- Psychomotor agitation.
- Elevated mood.
- Distractibility.
- Grandiosity.
- Racing thoughts.

Depressive-related features include:

- Low mood.
- Anhedonia.
- Emptiness.
- Poor concentration.
- Worthlessness.
- Guilt.
- Avolition.
- Blunted affect.
- Suicidal-attempt history.

Examples include:

- `low_mood_Yes = 0.9744`
- `anhedonia_Yes = 0.9179`
- `emptiness_Yes = 0.8051`
- `poor_concentration_Yes = 0.7538`
- `worthless_Yes = 0.6154`
- `guilt_Yes = 0.4769`

The simultaneous presence of substantial manic and depressive indicators
makes this cluster distinct from the predominantly manic group.

For the purposes of this project, this cluster can be described as a
high recorded combined symptom-burden group.

---

# 9. Why Three Clusters Were Selected for the Proposed System

The final choice of three clusters was made by considering the overall
behavior of the clustering experiments rather than relying on one
validation score.

The three-cluster solution satisfies several practical and analytical
requirements simultaneously.

### First, it produces a meaningful level of segmentation.

A two-group representation is relatively coarse for the intended purpose
of the project. The objective is to investigate different patient
patterns rather than simply divide the dataset into two broad
populations.

Three clusters allow the data to distinguish:

1.  A group with low recorded symptom presence and substantial unknown
    documentation.
2.  A predominantly manic symptom group.
3.  A group with substantial combined manic and depressive symptom
    burden.

This provides a more useful clinical stratification framework for the
proposed system.

### Second, three clusters receive direct support from the Calinski-Harabasz criterion.

The highest K-Means Calinski-Harabasz score in the tested range occurs
at:

`k = 3`

with:

`CH = 124.618042`

This indicates that the three-cluster configuration provides the
strongest balance of between-cluster separation and within-cluster
compactness under this criterion.

### Third, the three-cluster solution is supported by the WCSS/elbow behavior.

The WCSS falls substantially from k=1 through k=3, while the benefit of
introducing additional clusters becomes progressively less useful for
the intended segmentation.

The three-cluster solution therefore provides a reasonable complexity
level without fragmenting the population into numerous small groups.

### Fourth, K-Means and GMM independently recover exactly the same three groups.

This is particularly important.

The K-Means and GMM algorithms make different assumptions about the
structure of the data, yet both produce:

`41 / 106 / 195`

patients in exactly corresponding groups.

ARI and NMI are both exactly 1.0.

This demonstrates that the selected three-group partition is strongly
reproducible across the two principal clustering approaches used on the
168-feature representation.

### Fifth, increasing k beyond three does not improve the practical segmentation.

Although some individual metrics fluctuate at k=4, the clustering
quality deteriorates considerably from k=5 onward.

The Silhouette score drops to 0.163465 at k=5 and continues to remain
relatively low. The Davies-Bouldin index increases sharply, reaching
2.716281 at k=10, while the Calinski-Harabasz score falls to 50.473556.

Therefore, using substantially more than three groups would introduce
additional fragmentation without providing a corresponding improvement
in the overall structure.

### Sixth, three clusters are more useful for the intended application.

The proposed system is intended to provide an interpretable
patient-stratification framework.

A three-group structure offers sufficient granularity to distinguish
clinically different symptom patterns while remaining simple enough to
interpret and communicate.

The resulting groups are not merely numerical partitions. Their feature
profiles show recognizable differences in manic symptoms, depressive
symptoms, and completeness of clinical documentation.

For this reason, three clusters were selected as the final working
configuration for the proposed system.

---

# 10. Limitations

The most important limitation of the study is the quality and size of
the available dataset.

The analysis was performed using only 342 unique patients after
duplicate removal, while the final representation contains 168 features.

This produces a relatively high-dimensional problem with a limited
number of observations.

The dataset also contains substantial missing information, particularly
among clinically important symptom variables.

Because missing categorical values were retained as `Unknown`,
missingness itself becomes part of the clustering representation.

This was demonstrated directly by removing the 64 `_Unknown` variables.

The three-cluster K-Means/GMM results changed from:

- Silhouette = 0.265246
- Davies-Bouldin = 1.432613
- Calinski-Harabasz = 124.618042

to:

- Silhouette = 0.163254
- Davies-Bouldin = 2.001378
- Calinski-Harabasz = 59.546689

after the `_Unknown` variables were removed.

This confirms that the observed clustering structure is substantially
influenced by the completeness and recording pattern of the clinical
data.

Therefore, the current validation scores should be considered
preliminary rather than definitive clinical evidence.

Another limitation is that the current stability assessment is based
primarily on different random initializations. A stronger future study
should additionally perform bootstrap and independent-sample stability
analysis.

Finally, the clusters have not yet been validated against an independent
clinician-defined severity label or an external clinical outcome.

---

# 11. Future Work

The current results provide a strong basis for continuing the research,
but the next phase should focus on improving the underlying dataset and
validating the discovered structure.

## 11.1 Larger dataset

A substantially larger patient dataset would improve the reliability of
the estimated cluster structure.

More observations would allow:

- More reliable centroid estimation.
- More stable covariance estimation for GMM.
- Better representation of less frequent patient patterns.
- More reliable validation metrics.
- More robust cluster stability analysis.
- Better generalization to unseen patients.

The current three-cluster solution should therefore be considered a
strong initial structure that can be refined as additional data become
available.

## 11.2 Better data completeness

Future data collection should prioritize the variables with the highest
missingness, particularly depressive, psychotic, self-harm, and
functioning-related variables.

Improving the completeness of these variables would reduce the extent to
which documentation patterns influence the clustering geometry.

## 11.3 Clinical validation

The three clusters should eventually be compared with clinician
assessments and established clinical severity measures.

This would allow the project to determine whether the computational
clusters correspond to clinically meaningful severity or illness-course
categories.

## 11.4 Repeated validation

Future experiments should use:

- Bootstrap resampling.
- Cross-sample validation.
- Adjusted Rand Index.
- Normalized Mutual Information.
- Cluster-size consistency.
- Sensitivity analysis under different preprocessing strategies.

## 11.5 Feature refinement

As the dataset grows, feature selection can be revisited.

The objective should be to retain clinically informative variables while
reducing redundant and documentation-driven dimensions.

Dimensionality-reduction techniques such as PCA may also be investigated
as supplementary experiments, particularly for assessing the behavior of
GMM in the current high-dimensional feature space.

---

# 12. Final Conclusion

The clustering experiments demonstrate that the bipolar-disorder dataset
contains distinguishable patient patterns that can be captured using
unsupervised machine learning.

After preprocessing, 342 unique patient records were represented using
168 features and analyzed using K-Means, Agglomerative Hierarchical
Clustering, and Gaussian Mixture Models.

The three-cluster solution was selected as the final working
configuration for the proposed system because it provides the most
useful combination of:

- meaningful patient segmentation,
- strong Calinski-Harabasz performance,
- favorable WCSS/elbow behavior,
- deterioration of clustering quality at higher cluster counts,
- strong agreement between K-Means and GMM,
- exact correspondence of the three patient groups across these two
  algorithms,
- and clinically interpretable differences between the resulting
  groups.

The K-Means and GMM models independently produced exactly the same three
patient groups, with 41, 106, and 195 patients respectively. The ARI and
NMI between the two solutions were both 1.000000. This provides strong
evidence that the three-group structure is not dependent on a single
clustering algorithm.

The validation scores are moderate and show some variation across
cluster counts and initialization. This behavior is expected given the
limitations of the available dataset. The dataset is relatively small
compared with the 168-dimensional representation and contains
substantial missing information. Sixty-four final features represent
explicit unknown states, and removing these variables substantially
weakens the observed clustering structure.

Consequently, the present results should be viewed as an initial
computational stratification rather than a final clinical severity
classification.

The main outcome of this study is the identification of three reproducible patient profiles characterized by distinct patterns of recorded clinical symptoms and documentation:

1.  **Low Recorded Symptoms / Limited Clinical Documentation**
2.  **Predominantly Manic/Hypomanic Profile**
3.  **Combined Manic–Depressive Profile**

These groups provide a practical foundation for the proposed
bipolar-disorder severity assessment system.

The most important next step is to expand and improve the clinical
dataset. With a larger number of patients, better documentation, fewer
missing values, and independent clinical validation, the cluster
boundaries and validation scores are expected to become more robust. The
three-cluster structure can then be re-evaluated and refined using a
substantially stronger evidence base.

Thus, the current study establishes a meaningful and reproducible
three-cluster framework while clearly identifying dataset quality and
sample size as the principal factors limiting the strength of the
present clustering validation.
