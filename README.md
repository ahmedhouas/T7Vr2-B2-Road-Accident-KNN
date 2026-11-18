# Road Accident Severity Classification Using K-Nearest Neighbors (KNN)

**Project ID:** T7Vr2-B2  
**Course:** ISII – Machine Learning Mini-Project (2026–2027)  
**Theme:** Th7 – Road Accidents (Addis Ababa)  
**Group:** G2 – Classification  
**Assigned Model:** K-Nearest Neighbors (KNN)  
**Project Status:** In Progress

---

## 1. Project Overview

This project investigates the application of the **K-Nearest Neighbors (KNN)** algorithm to classify road traffic accident severity using police-recorded accident data from Addis Ababa, Ethiopia.

Road traffic accidents represent a significant public-safety challenge. Predicting their severity from pre-crash circumstances is a challenging classification problem, particularly because the available explanatory variables are predominantly categorical and the target classes are strongly imbalanced.

The study focuses on understanding how different preprocessing strategies, distance calculations, neighborhood configurations, and class imbalance handling techniques influence KNN classification performance.

Rather than focusing exclusively on overall accuracy, the project emphasizes minority-class recognition, model stability, computational performance, and the practical implications of classification errors, particularly missed fatal accidents.

The work follows a controlled experimental methodology to ensure reproducibility, prevent data leakage, and support meaningful comparisons with the other five models in Theme 7, Group G2.

## 2. Research Problem and Objectives

### 2.1 Main Research Question

How accurately can a K-Nearest Neighbors classifier predict the severity of a road accident (slight, serious, or fatal) in Addis Ababa from pre-crash circumstances, and how sensitive is its neighborhood voting to encoding, scaling, class imbalance, and data-splitting regimes?

### 2.2 Research Objectives

The main objectives are to:

1. Understand the mathematical principles, distance assumptions, and inductive bias of KNN classification.
2. Develop a reproducible machine learning pipeline from raw accident records to severity predictions.
3. Investigate the influence of categorical encoding and feature standardization on Euclidean distance calculations.
4. Evaluate the effect of neighborhood size and distance-weighted voting on classification performance.
5. Investigate class imbalance handling techniques and their influence on fatal-class recognition.
6. Evaluate model performance using imbalance-aware metrics and computational execution times.
7. Analyze classification errors, especially fatal accidents incorrectly classified as less severe.
8. Examine model stability under different data-splitting regimes.
9. Compare KNN performance with the other five models in Theme 7, Group G2.

## 3. Research Questions and Hypotheses

### 3.1 Research Questions

The study investigates the following questions:

- How do Pure One-Hot Encoding and Mixed Ordinal + One-Hot Encoding affect KNN performance?
- How does StandardScaler influence Euclidean distance calculations and classification performance?
- What is the bias–variance trade-off across different neighborhood sizes?
- Does distance-weighted voting improve classification compared with uniform voting?
- Can undersampling or SMOTENC improve fatal-class recall without excessively reducing Macro-F1?
- How stable is KNN across repeated random splits and different data regimes?
- What is the predictive contribution of driver-related variables?

### 3.2 Initial Hypotheses

At least two explicit, falsifiable hypotheses will be formulated and documented before model training, as required by the project specification.

Their experimental evaluation and final acceptance or rejection will be discussed in the final report.

*See `00_project_docs/hypotheses.md` for the formal hypotheses and their evaluation criteria.*

## 4. Dataset Description

### 4.1 Dataset Source

The project uses the **Addis Ababa City Road Traffic Accident Severity Dataset**, published by Enyew (2025).

- Dataset: Addis Ababa City Road Traffic Accident Severity Dataset
- Collection period: 2016–2022
- Expected observations: 13,064 accident records
- Number of variables: 31
- Source: Figshare
- DOI: [10.6084/m9.figshare.28122899](https://doi.org/10.6084/m9.figshare.28122899)

The uploaded dataset will be audited to verify its dimensions, integrity, missing values, duplicate records, and consistency with the assignment specification.

### 4.2 Target Variable

The classification target is accident severity, consisting of three classes:

| Class | Description |
|---|---|
| Slight | Slight injury |
| Serious | Serious injury |
| Fatal | Fatal injury |

The target distribution is strongly imbalanced, with fatal accidents representing a small minority.

### 4.3 Permitted Input Features

The project uses pre-crash circumstances specified in the assignment:

- **Temporal:** Time of day, day of week.
- **Environmental:** Weather and light conditions.
- **Road characteristics:** Road surface, road alignment, junction type.
- **Accident characteristics:** Collision type.
- **Driver characteristics:** Age band, sex, education, driving experience.
- **Vehicle characteristics:** Vehicle type, service years, defect, movement.
- **Vehicle involvement:** Number of vehicles involved.

Only permitted circumstance variables will be considered as model inputs.

### 4.4 Data Leakage Prevention

Variables that directly encode accident severity or casualty outcomes must be excluded from all model inputs.

These include:

- Casualty class.
- Casualty severity.
- Casualty age.
- Casualty sex.
- Casualty fitness.
- Direct casualty counts.

The permitted number of vehicles involved must not be confused with the number of casualties.

All learned preprocessing operations must be fitted exclusively on training data, and resampling must occur inside cross-validation folds.

## 5. Methodology

The project follows a structured experimental workflow.

### Stage 0 – Research Hypotheses and Reference Split

- Formulate and document at least two falsifiable hypotheses.
- Audit the original dataset.
- Identify and exclude leakage variables.
- Establish a stratified training/test partition using `random_state=42`.
- Record the partition in `dataset_clean.csv` through the `split_ref` column.
- Freeze the reference test set for final evaluation.

### Stage 1 – Exploratory Data Analysis and Baselines

- Investigate dataset dimensions and integrity.
- Analyze missing and unknown values.
- Identify duplicate records and rare categorical levels.
- Examine target class distribution.
- Explore relationships between accident circumstances and severity.
- Establish majority-class and default KNN baselines.

### Stage 2 – Preprocessing and Hyperparameter Tuning

Implement the required preprocessing alternatives:

**Encoding Configuration A – Pure One-Hot Encoding**

Apply OneHotEncoder to all categorical predictors.

**Encoding Configuration B – Mixed Ordinal + One-Hot Encoding**

Apply OrdinalEncoder to naturally ordered bands:

- Driver age band.
- Driving experience.
- Vehicle service years.

Apply OneHotEncoder to nominal categorical variables.

**Feature Standardization**

Apply StandardScaler to the encoded features, fitted exclusively on training folds.

**KNN Hyperparameter Tuning**

Use GridSearchCV with:

- Cross-validation: RepeatedStratifiedKFold.
- Number of folds: 5.
- Number of repetitions: 3.
- Random state: 42.
- Distance metric: Euclidean.

Tune the following parameters:

| Parameter | Values |
|---|---|
| n_neighbors | 3, 5, 7, 11, 15 |
| weights | uniform, distance |
| metric | euclidean (fixed) |

The required tuning visualization is a 5 × 2 heatmap or equivalent validation curve.

### Stage 3 – Class Imbalance Experiments

Compare:

1. KNN without resampling.
2. KNN with random undersampling.
3. KNN with SMOTENC.

Resampling must be performed inside the cross-validation pipeline to prevent data leakage.

### Stage 4 – Encoding and Scaling Comparison

Evaluate the influence of:

- Pure One-Hot vs. Mixed Ordinal + One-Hot Encoding.
- StandardScaler vs. no StandardScaler.

The comparisons will investigate how preprocessing choices influence the Euclidean neighborhood geometry and classification performance.

### Stage 5 – Final Held-Out Evaluation

Evaluate the selected pipeline once on the untouched reference test set.

The evaluation includes:

- Classification metrics.
- 3 × 3 confusion matrix.
- Class-specific performance.
- Misclassification analysis.
- Fatal-class false-negative investigation.
- Execution latency.

### Stage 6 – Validation Splits and Feature Ablation

Evaluate generalization under alternative data-splitting protocols:

**Experiment 1 – Repeated Stratified Hold-Out**

- 80% training / 20% testing.
- 10 random seeds.
- Report mean and standard deviation.

**Experiment 2 – Out-of-Period / Regime Validation**

- Train on earlier years and test on later years if a year variable is available.
- Otherwise, train on daytime accidents and test on nighttime accidents.

**Experiment 3 – Small-Data Regime (Optional)**

- Train on a stratified 10% subset.
- Test on the remaining 90%.

**Driver Feature Ablation (Optional)**

Remove the driver feature family:

- Age band.
- Sex.
- Education.
- Driving experience.

Measure the resulting changes in Macro-F1, fatal-class recall, and Balanced Accuracy.

### Stage 7 – Group G2 Comparison

Compare the results of KNN with the five peer models:

1. Logistic Regression.
2. K-Nearest Neighbors.
3. Support Vector Machine.
4. Random Forest.
5. Decision Tree.
6. Multi-Layer Perceptron.

The comparison will use the shared reference dataset, experimental protocol, evaluation metrics, and execution-time measurements.

## 6. Evaluation Metrics

Because the target classes are strongly imbalanced, overall accuracy alone is insufficient.

The main evaluation metrics are:

| Metric | Purpose |
|---|---|
| Macro-F1 | Primary comparison metric, treating all classes equally |
| Fatal-class Recall | Measures recognition of fatal accidents |
| Balanced Accuracy | Measures average recall across the three classes |
| Confusion Matrix | Identifies class-specific classification errors |

Computational measurements:

- `fit_time_s`
- `predict_time_s`
- `tuning_time_s`

All execution times will be measured using Python's `time.perf_counter()`.

## 7. Repository Structure

The repository is organized to separate research documentation, data, analysis, implementation, experimentation, testing, and final deliverables.

```text
T7Vr2-B2-Road-Accident-KNN/
│
├── 00_project_docs/
│   ├── project_specification.pdf
│   ├── project_plan.md
│   ├── research_questions.md
│   ├── hypotheses.md
│   ├── dataset_description.md
│   └── decisions_log.md
│
├── 01_data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── 02_notebooks/
│
├── 03_src/
│
├── 04_tests/
│
├── 05_experiments/
│   ├── configs/
│   ├── logs/
│   └── results/
│
├── 06_reports/
│   ├── figures/
│   ├── tables/
│   └── drafts/
│
├── 07_presentation/
│
├── 08_documentation/
│
├── .gitignore
├── .gitattributes
├── requirements.txt
└── README.md
```

The subdirectories will be progressively populated as the project develops.

## 8. Development Environment

The project is developed locally using:

| Component | Technology |
|---|---|
| Operating System | Windows 11 |
| IDE | Visual Studio Code |
| Programming Language | Python 3.12 |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Imbalance Handling | Imbalanced-learn |
| Visualization | Matplotlib, Seaborn |
| Experimentation | Jupyter Notebook |
| Version Control | Git and GitHub |

Dependencies will be documented and pinned in `requirements.txt` to support reproducibility.

## 9. Reproducibility

Reproducibility is a central requirement of this project.

The following practices will be maintained:

- Fixed random seed: `random_state=42`.
- Isolated Python virtual environment.
- Pinned library dependencies.
- Consistent experimental configurations.
- Explicit recording of hyperparameters and random seeds.
- Preprocessing and resampling inside training pipelines.
- Strict separation between training and held-out test data.
- Programmatically generated results, figures, and evaluation tables.

The final notebook must execute from beginning to end without errors.

## 10. Project Deliverables

The required submission artifacts are:

| Deliverable | Description |
|---|---|
| `T7Vr2-B2_report.pdf` | Written scientific report |
| `T7Vr2-B2_notebook.ipynb` | Executable Jupyter notebook |
| `dataset_clean.csv` | Cleaned dataset with reference split |
| `results.csv` | Consolidated experiment results |
| `requirements.txt` | Pinned dependencies |
| `README.md` | Project description and reproduction instructions |
| `T7Vr2-B2_presentation.pdf` | Ten-slide oral presentation |

The written report will follow the length and structure confirmed by the instructor.

## 11. References

1. Enyew, G. G. (2025). *Addis Ababa City Road Traffic Accident Severity Dataset*. Figshare. https://doi.org/10.6084/m9.figshare.28122899

2. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory, 13*(1), 21–27.

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

---

**Project:** T7Vr2-B2  
**Theme:** Road Accident Severity Classification  
**Model:** K-Nearest Neighbors (KNN)  
**Academic Year:** 2026–2027