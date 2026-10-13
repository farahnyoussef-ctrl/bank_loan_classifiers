# Bank Personal Loan Prediction: Classifier Benchmark (with and without SMOTE)

Predicting which bank customers accept a personal loan, and measuring whether SMOTE oversampling helps on an imbalanced dataset (about 9% positives).

Six classifiers are compared under two setups on the same train/test split:
**KNN, SVM (RBF), Gaussian Naive Bayes, AdaBoost (decision stumps), AdaBoost + SVM, AdaBoost + Naive Bayes.**

## Dataset

Bank Personal Loan Modelling (public bank-customer dataset) (5,000 customers, 14 columns). Target: `Personal Loan` (1 = accepted).
Place `Bank_Personal_Loan_Modelling.xlsx` next to the notebook before running.

## Files

| File | Description |
|---|---|
| `Bank_Loan_Classifiers.ipynb` | Full pipeline: cleaning, split, benchmark, SMOTE comparison, clustering. Saved with outputs. |
| `results.csv` | Metrics for every model and setup. |

## Method

1. **Cleaning:** removed 1 invalid ZIP code, fixed 52 negative `Experience` values, removed 105 `Mortgage` outliers (z-score > 3), converted `CCAvg` from monthly to annual to match `Income`. Result: 4,894 rows (8.8% positives). No duplicates or missing values.
2. **Features:** `ID` and `ZIP Code` dropped (row number and unordered code).
3. **Split:** stratified 70/30 (3,425 train / 1,469 test, 130 positives in the test set), `random_state=0`.
4. **Pipelines:** scaling happens inside each pipeline. SMOTE is applied only to training data, including inside each cross-validation fold, so the test set is never resampled.
5. **KNN:** `k` chosen by 5-fold CV on F1 (k = 3 in both setups).
6. **Evaluation:** precision, recall and F1 on the loan class (not accuracy alone), ROC-AUC, plus 5-fold CV F1 on the training set. Models are selected by CV F1, not by test score.

## Results

Majority-class baseline ("nobody takes a loan"): **accuracy 0.912, recall 0.000, F1 0.000**. Accuracy is therefore not informative here; precision, recall and F1 of the loan class are reported instead.

**Without SMOTE** (test set, loan class)

| Model | Accuracy | Precision | Recall | F1 | CV F1 (mean ± std) |
|---|---|---|---|---|---|
| KNN | 0.963 | 0.952 | 0.615 | 0.748 | 0.742 ± 0.046 |
| SVM (RBF) | 0.974 | 0.969 | 0.723 | 0.828 | 0.820 ± 0.030 |
| Naive Bayes | 0.884 | 0.390 | 0.561 | 0.461 | 0.496 ± 0.040 |
| AdaBoost | 0.969 | 0.947 | 0.692 | 0.800 | 0.795 ± 0.046 |
| AdaBoost + SVM | 0.911 | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| AdaBoost + NB | 0.928 | 0.613 | 0.500 | 0.551 | 0.265 ± 0.217 |

**With SMOTE** (training data only)

| Model | Accuracy | Precision | Recall | F1 | CV F1 (mean ± std) |
|---|---|---|---|---|---|
| KNN | 0.963 | 0.768 | 0.839 | 0.801 | 0.746 ± 0.038 |
| SVM (RBF) | 0.971 | 0.801 | 0.900 | 0.848 | 0.795 ± 0.027 |
| Naive Bayes | 0.873 | 0.392 | 0.785 | 0.523 | 0.559 ± 0.007 |
| AdaBoost | 0.879 | 0.417 | 0.931 | 0.576 | 0.615 ± 0.021 |
| AdaBoost + SVM | 0.830 | 0.332 | 0.915 | 0.488 | 0.505 ± 0.011 |
| AdaBoost + NB | 0.796 | 0.267 | 0.746 | 0.393 | 0.428 ± 0.032 |

### Findings

- **Best model:** SVM (RBF). Without SMOTE: precision 0.969, recall 0.723, F1 0.828 (CV F1 0.820 ± 0.030).
- **SMOTE raises recall for every model** (SVM 0.723 → 0.900, KNN 0.615 → 0.838) but lowers precision (SVM 0.969 → 0.801), so it catches more real loan takers at the cost of more false alarms.
- **SMOTE does not clearly improve F1.** SVM F1 goes from 0.828 to 0.848 on the test split, but cross-validated F1 goes from 0.820 to 0.795; with 130 test positives, a difference this small is within noise. For AdaBoost and AdaBoost + NB, F1 drops with SMOTE.
- **AdaBoost + SVM without SMOTE collapses to the majority class** (recall 0, accuracy 0.912, equal to the baseline). A strong learner such as an RBF SVM is a poor boosting base estimator.
- Choice between setups depends on the cost of errors: SMOTE if missing a potential customer is costlier than contacting an uninterested one, no SMOTE if precision matters more.

## Limitations

- One dataset and one train/test split; the test set has only 130 positives, so single-split differences of a few points are not reliable. Cross-validated figures are included for that reason.
- Hyperparameters were tuned only for KNN (`k`); the other models use fixed, reasonable defaults. Tuning them (e.g. `C` and `gamma` for the SVM) could change the ranking.
- The clustering section is exploratory and does not feed into the classifiers.

## Requirements

Python 3.10+ with:

```
pandas numpy scipy scikit-learn imbalanced-learn matplotlib seaborn openpyxl
```

Install with `pip install pandas numpy scipy scikit-learn imbalanced-learn matplotlib seaborn openpyxl`, then run the notebook top to bottom (a few minutes).
