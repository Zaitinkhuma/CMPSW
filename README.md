# Class-Mass-Preserving Support Weighting

Class-Mass-Preserving Support Weighting (CMPSW) for imbalanced cardiovascular classification with XGBoost.

## Repository

The repository contains the Google Colab implementation used for the experiments, together with the required Python packages.

## Datasets

All datasets are publicly available through the UCI Machine Learning Repository.

| Dataset | Samples | Predictors used | Target | Majority | Minority | Imbalance ratio | Source |
|---|---:|---:|---|---:|---:|---:|---|
| Z Alizadeh Sani | 303 | 55 | Cath | 216 | 87 | 2.483 | [UCI](https://archive.ics.uci.edu/dataset/412/z) |
| Heart Failure Clinical Records | 299 | 11 | DEATH_EVENT | 203 | 96 | 2.115 | [UCI](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) |
| Long Beach VA | 200 | 10 | Diagnosis converted to binary | 149 | 51 | 2.922 | [UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) |

For Z Alizadeh Sani, CAD is mapped to 1 and NORMAL to 0 during data loading. For Heart Failure Clinical Records, DEATH_EVENT is used as the target. For Long Beach VA, diagnosis values greater than zero are assigned to the positive class. Before each outer split, the minority class in the training partition is recoded to 1 and the same mapping is applied to the corresponding test partition.

## Preprocessing

All preprocessing is fitted using the corresponding training partition only.

1. Column names are stripped and blank values are treated as missing.
2. Object columns that are fully numeric are converted to numeric values.
3. Integer like predictors with at most 10 distinct observed values are treated as categorical; the remaining predictors are treated as numerical.
4. Numerical predictors are median imputed and standardized.
5. Categorical predictors are imputed with the most frequent value and one hot encoded. Unseen categories are ignored and binary categories use a single encoded column.
6. The `time` variable is excluded from Heart Failure Clinical Records.
7. The `slope`, `ca`, and `thal` variables are excluded from Long Beach VA because of substantial missingness.
8. No synthetic samples, sample duplication, or sample removal is used by the evaluated methods.

## Compared methods

| Method | Weighting configuration |
|---|---|
| XGB | Unweighted XGBoost |
| Class-XGB | Inverse frequency class weighting, $w_y=n/(2n_y)$ |
| SPW-XGB | Positive class scaling, $\rho=n_0/n_1$ |
| Weighted-XGB | Positive class scaling with $\alpha \in \{1.5,2.0,2.5,3.0,4.0\}$ selected by inner validation |
| CB-XGB | Effective number class weighting with $\beta \in \{0.90,0.99,0.999\}$, normalized to mean weight 1 |
| CMPSW | Inverse-frequency class balancing with normalized same-class neighbourhood support |

For CMPSW, the neighbourhood size is determined as $k=\min(\max(1,\lceil\sqrt{n}\rceil),n-1)$. Euclidean nearest neighbours are used. Local support is the proportion of the $k$ neighbours having the same class label. Support is normalized within each class and combined with the inverse frequency class weight. The resulting weights preserve an aggregate class mass of $n/2$ for each class. The weights are computed from the training partition and remain fixed during model training.

## XGBoost configuration

| Parameter | Setting |
|---|---|
| Objective | `binary:logistic` |
| Evaluation metric | `aucpr` |
| `max_depth` | `{2, 3}` |
| `eta` (learning rate) | `{0.05, 0.10}` |
| Boosting rounds | `200` |
| `min_child_weight` | `1` |
| `subsample` | `0.9` |
| `colsample_bytree` | `0.9` |
| `reg_lambda` | `1.0` |
| L1 regularization (`alpha`) | `0.0` |
| `scale_pos_weight` | `1.0` unless specified by the method |
| `tree_method` | `hist` |
| `nthread` | `-1` |
| `verbosity` | `0` |

The common tree depth and learning rate were selected by inner cross validation. Variant specific weighting parameters were selected within the same procedure where applicable.

## Validation

A repeated nested stratified cross validation design was used.

| Component | Setting |
|---|---|
| Outer validation | 5 folds × 5 repeats |
| Outer evaluations | 25 per dataset |
| Inner validation | 3 folds |
| Selection criterion | AUC PR |
| Base random seed | 42 |
| Threshold for threshold dependent metrics | 0.5 |

The four reported measures are AUC PR, MCC, G Mean, and Sensitivity. Statistical comparisons use matched outer fold differences with the corrected variance approach and Holm adjustment.

## Execution

Open `cmpsw.ipynb` in Google Colab and run the cells sequentially. The notebook downloads the public datasets, performs training partition specific preprocessing and weighting, executes the nested validation procedure, and saves the generated results and figures to:

`/content/drive/MyDrive/CMPSW_CODE_V1`

## Requirements

Python packages required by the notebook are listed in `requirements.txt`.
