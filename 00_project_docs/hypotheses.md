# Research Hypotheses

**Project:** Comparative Study of K-Nearest Neighbors for Road Accident Severity Classification  
**Pair ID:** T7Vr2-B2  
**Theme:** Th7 — Road Accidents (Addis Ababa)  
**Group:** G2 — Classification  
**Assigned Model:** K-Nearest Neighbors (KNN)

---

## 1. Purpose

This document records the hypotheses established before model training, as required by the project specification.

The study investigates how feature scaling and class-imbalance handling affect the performance of a K-Nearest Neighbors (KNN) classifier in predicting road accident severity.

The hypotheses will be evaluated through controlled experiments using the prescribed training data and cross-validation protocol. They will be explicitly confirmed or rejected in the final project discussion.

## 2. Hypothesis H1 — Effect of Feature Scaling

**Hypothesis statement:**

Applying `StandardScaler` to the encoded input features will improve the Macro-F1 score of the KNN classifier compared with using the same configuration without feature scaling.

**Rationale:**

KNN is a distance-based algorithm. It determines predictions by identifying the nearest observations in the feature space. Consequently, differences in feature scales and distributions can influence the Euclidean distances used to identify neighbors.

Standardization may help prevent features with larger numerical ranges or variances from disproportionately influencing the distance calculations.

**Experimental method:**

- Use the same training data, cross-validation folds, feature encoding strategy, and KNN configuration.
- Evaluate one configuration with `StandardScaler` and another without it.
- Fit the scaler only within the training folds to prevent data leakage.
- Compare the resulting Macro-F1 scores.

**Evaluation criterion:**

H1 will be supported if the configuration using `StandardScaler` achieves a higher mean Macro-F1 score than the configuration without scaling under the controlled comparison. Otherwise, H1 will not be supported.

## 3. Hypothesis H2 — Effect of Class-Imbalance Handling

**Hypothesis statement:**

Applying a class-imbalance handling strategy during training will improve the recall of the fatal accident class compared with training without resampling.

**Rationale:**

The accident severity dataset has an imbalanced target distribution, with fatal accidents representing a much smaller proportion of observations than slight and serious accidents.

A classifier trained on this distribution may favor the majority class and fail to identify a sufficient proportion of fatal accidents. Resampling may help the classifier learn from the underrepresented classes.

**Experimental method:**

Compare the following three configurations:

1. **No resampling:** Train using the original class distribution.
2. **Random undersampling:** Reduce the number of majority-class observations in the training data.
3. **SMOTENC:** Apply synthetic minority oversampling for the categorical-feature setting.

All resampling operations must be performed inside the training cross-validation pipeline. Validation and test data must not be resampled.

**Evaluation criterion:**

H2 will be supported if at least one resampling strategy achieves a higher mean fatal-class recall than the no-resampling configuration.

Macro-F1 and Balanced Accuracy will also be examined to determine whether any improvement in fatal-class recall comes at the expense of overall classification performance.

## 4. Experimental Controls

To make the comparisons meaningful and reproducible:

- Use the same reference training partition and cross-validation protocol for the corresponding comparisons.
- Use `RepeatedStratifiedKFold` with 5 folds, 3 repeats, and `random_state=42`.
- Keep other experimental settings identical when comparing configurations.
- Fit all learned preprocessing and resampling operations exclusively on the relevant training folds.
- Use Macro-F1 as the primary evaluation metric, alongside fatal-class recall and Balanced Accuracy.
- Do not use the held-out reference test set for tuning or experimental decisions.

## 5. Hypothesis Evaluation Record

The following table will be completed after the corresponding experiments.

| Hypothesis | Main comparison | Expected outcome | Final status |
|---|---|---|---|
| H1 | With vs. without StandardScaler | Higher Macro-F1 with scaling | Pending |
| H2 | Resampling vs. no resampling | Higher fatal-class recall with at least one resampling strategy | Pending |

**Important:** The hypotheses are predictions, not established findings. Their outcomes must be reported honestly, including cases where the experimental evidence does not support the expected result.

## 6. Final Discussion Requirement

In the final report, each hypothesis must be explicitly addressed using the experimental results.

For each one, the discussion should:

- State whether the hypothesis was supported or not supported.
- Present the relevant quantitative results.
- Explain the observed behavior in the context of KNN and the dataset characteristics.
- Discuss relevant limitations or trade-offs.