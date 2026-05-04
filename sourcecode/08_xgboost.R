# =============================================================================
# Section 8: Model 3 — XGBoost
# =============================================================================
# Why XGBoost (replacing the original SVM):
#   - SVM is O(n^2 to n^3) — would take ~30 min on 21K rows
#   - XGBoost dominates tabular ML benchmarks since 2014
#   - Boosting captures non-linearities and feature interactions that LASSO misses

# -----------------------------------------------------------------------------
# 8.1 Build DMatrix (xgboost-native column-store format, faster than data.frame)
# -----------------------------------------------------------------------------
x_train_mat <- train_baked |> select(-default_payment_next_month) |> as.matrix()
x_test_mat  <- test_baked  |> select(-default_payment_next_month) |> as.matrix()
y_train_xgb <- train_baked$default_payment_next_month
y_test_xgb  <- test_baked$default_payment_next_month

dtrain <- xgb.DMatrix(data = x_train_mat, label = y_train_xgb)
dtest  <- xgb.DMatrix(data = x_test_mat,  label = y_test_xgb)


# -----------------------------------------------------------------------------
# 8.2 Hyperparameters
# -----------------------------------------------------------------------------
# Key parameters explained:
#   learning_rate     — step size; 0.1 is standard. Lower = more trees needed
#   max_depth         — tree depth; boosting uses shallow trees (3-6)
#   subsample         — row sub-sampling (anti-overfitting)
#   colsample_bytree  — column sub-sampling (similar to RF mtry)
#   scale_pos_weight  — class imbalance correction = (neg / pos)
#                       For us: 78/22 ≈ 3.5 — defaulters get 3.5x weight

params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  learning_rate    = 0.1,
  max_depth        = 4,
  min_child_weight = 1,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = 78 / 22,
  nthread          = parallel::detectCores() - 1
)


# -----------------------------------------------------------------------------
# 8.3 Train with early stopping
# -----------------------------------------------------------------------------
# Early stopping is essential for boosting: the model overfits if trained too
# long. We monitor test AUC each round and stop after 20 rounds of no
# improvement, keeping the best iteration's model.

set.seed(1234)
watchlist <- list(train = dtrain, test = dtest)

xgb_fit <- xgb.train(
  params                = params,
  data                  = dtrain,
  nrounds               = 500,
  evals                 = watchlist,           # was 'watchlist' in older xgboost; now 'evals'
  early_stopping_rounds = 20,
  print_every_n         = 20,
  verbose               = 1
)

cat("\nBest iteration:", xgb_fit$best_iteration, "\n")
cat("Best test AUC:",   round(as.numeric(xgb_fit$best_score), 4), "\n")


# -----------------------------------------------------------------------------
# 8.4 Predict + evaluate
# -----------------------------------------------------------------------------
pred_prob_xgb <- predict(xgb_fit, dtest)

cat("\n========== XGBoost ==========\n")
result_xgb <- evaluate_model(pred_prob_xgb, y_test_xgb, model_name = "XGBoost")
get_performance(result_xgb$Confusion_Matrix)
cat("\nXGBoost AUC:", round(result_xgb$AUC, 4), "\n")


# -----------------------------------------------------------------------------
# 8.5 Feature importance
# -----------------------------------------------------------------------------
# XGBoost reports three metrics:
#   Gain      — total information gain from splits using this feature (most useful)
#   Cover     — total coverage (sample weight) of splits
#   Frequency — how often this feature is used in splits

importance_xgb <- xgb.importance(model = xgb_fit)
cat("\n--- Top 10 XGBoost Features ---\n")
print(head(importance_xgb, 10))

xgb.plot.importance(importance_xgb, top_n = 10,
                    main = "XGBoost — Top 10 Feature Importance")