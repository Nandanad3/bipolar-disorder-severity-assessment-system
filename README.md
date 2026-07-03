# An Intelligent Bipolar Disorder Severity Assessment System Using Unsupervised Machine Learning

## 📖 Overview
This project explores unsupervised machine learning techniques to assess the severity of **bipolar disorder**, aiming to support intelligent, data‑driven clinical decision‑making. Building upon the base paper *[Classification of Patients with Bipolar Disorder Using K-Means Clustering](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0210314)*, our system extends the methodology by implementing and comparing multiple clustering algorithms on patient data.

## 🎯 Objectives
- Develop a robust framework for **severity stratification** of bipolar disorder patients.  
- Apply diverse **unsupervised clustering algorithms** to clinical, cognitive, physical health, and functional datasets.  
- Evaluate clustering performance using standard metrics to identify the most suitable algorithm.  
- Provide meaningful insights to assist healthcare professionals in understanding disease severity.

## 🧠 Methodology
We implement and compare the following clustering algorithms:
- **Hierarchical Clustering**  
- **Spectral Clustering**  
- **Gaussian Mixture Models (GMM)**  
- **BIRCH**  
- **DBSCAN**  
- **OPTICS**  
- **Mean Shift**  
- **Affinity Propagation**  

### Evaluation Metrics
- **Silhouette Score**  
- **Davies-Bouldin Index**  
- **Calinski-Harabasz Index**  

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn)  
- **Visualization**: Matplotlib, Seaborn  
- **Environment**: Python (Google Colab / Jupyter Notebook)  

## 📂 Repository Structure
```
├── notebooks/           # Jupyter notebooks for experiments
├── src/                 # Core implementation scripts
├── results/             # Visualizations and evaluation outputs
├── README.md            # Project documentation
```

## 📌 References
- Base Paper: *Classification of Patients with Bipolar Disorder Using K-Means Clustering*  
  [Link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0210314)

