# Credit Card Default Prediction

A binary classification project predicting next-month credit card default for 30,000 Taiwanese customers, using LASSO Logistic Regression, Random Forest, and XGBoost.

This repository contains both the **2019 original** (course project) and a **2026 modernized version** that I refactored as a learning exercise — demonstrating how the R ecosystem has evolved (tidyverse, recipes, rsample, ranger, xgboost).

## Project Structure
```
.
├── credit_card_default_modernized.R   # Combined modernized pipeline (start here)
├── sourcecode/                         # Same pipeline split into 10 numbered files
│   ├── 00_load_packages.R
│   ├── 01_data_ingestion.R
│   ├── ...
│   └── 09_model_comparison.R
└── data/                               # Data not committed; see below
```
## Dataset

**Source**: [UCI ML Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

- 30,000 customers, 25 features (demographics, credit limit, 6-month bill/payment history)
- Target: binary indicator of next-month default
- Class imbalance: 22% positive rate

To run the code, download the dataset from the UCI link and place it in `data/UCI_Credit_Card.csv`.

## Results

| Model | AUC | F1 | Sensitivity | Specificity |
|-------|-----|-----|-------------|-------------|
| LASSO Logistic | 0.717 | 0.346 | 0.229 | 0.973 |
| Random Forest  | 0.776 | 0.469 | 0.366 | 0.944 |
| XGBoost        | 0.780 | 0.534 | 0.631 | 0.792 |

**Key findings**:
- `pay_0` (most recent payment status) is the dominant predictor in all three models.
- Tree-based models outperform LASSO by ~0.06 AUC, indicating meaningful non-linear structure.
- XGBoost's higher Sensitivity reflects `scale_pos_weight` class-imbalance correction; AUC and F1 are the fairest cross-model comparators.

## Modernization Highlights

| Aspect | 2019 Original | 2026 Refactor |
|--------|---------------|---------------|
| Style | base R, repeated `df$col <- ...` | tidyverse pipelines with `\|>` |
| Pre-processing | `scale()` on full data (data leakage) | `recipes` pipeline, train-only fit |
| Train/test split | `sample()` (no stratification) | `rsample::initial_split(strata=...)` |
| Random Forest | `randomForest` (single-threaded) | `ranger` (multi-threaded) |
| Third model | SVM (slow on 21k rows) | XGBoost with early stopping |
| Plot composition | `gridExtra::grid.arrange` | `patchwork` |

## Reproducing the Analysis

```r
install.packages(c(
  "tidyverse", "readxl", "janitor", "patchwork", "scales",
  "corrplot", "rsample", "recipes", "glmnet", "ROCR",
  "ranger", "xgboost"
))

source("credit_card_default_modernized.R")
```

## Tech Stack

R 4.5 · tidyverse · ggplot2 · patchwork · recipes · rsample · glmnet · ranger · xgboost

## Author

**Kehan Wang** — Master's student in Information Systems @ NYU
