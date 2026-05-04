# =============================================================================
# Section 0: Load Packages
# =============================================================================

# Install once if not already available:
# install.packages(c("tidyverse", "readxl", "janitor", "patchwork", "scales",
#                    "corrplot", "rsample", "recipes", "glmnet", "ROCR",
#                    "ranger", "xgboost"))


library(tidyverse)   # ggplot2, dplyr, tidyr, readr, purrr, forcats
library(readxl)      # Read Excel files (the source data is .xls)
library(janitor)     # Clean column names (snake_case)
library(patchwork)   # Combine multiple ggplots with + and /
library(scales)      # Formatting helpers (comma, percent, etc.)
library(corrplot)    # Correlation matrix visualization
library(rsample)     # Stratified train/test split
library(recipes)     # Pre-processing pipeline
library(glmnet)      # LASSO / Ridge / Elastic Net
library(ROCR)        # ROC curves and AUC
library(ranger)      # Fast Random Forest (multi-threaded C++ implementation)
library(xgboost)     # Gradient boosting

set.seed(1234)       # Global seed for reproducibility