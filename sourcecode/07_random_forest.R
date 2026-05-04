# =============================================================================
# Section 7: Model 2 — Random Forest (ranger)
# =============================================================================
# We use ranger instead of the original randomForest package because:
#   - ranger is multi-threaded C++ (5-10x faster)
#   - Native support for permutation importance (more reliable than Gini)
#   - Cleaner output objects

# -----------------------------------------------------------------------------
# 7.1 Convert outcome to factor (ranger uses outcome type to infer task)
# -----------------------------------------------------------------------------
# If outcome is numeric, ranger does regression. We need it as a factor for
# classification. This is a classic ranger pitfall.

train_rf <- train_baked |>
  mutate(default_payment_next_month = factor(default_payment_next_month,
                                             levels = c(0, 1),
                                             labels = c("no_default", "default")))
test_rf <- test_baked |>
  mutate(default_payment_next_month = factor(default_payment_next_month,
                                             levels = c(0, 1),
                                             labels = c("no_default", "default")))


# -----------------------------------------------------------------------------
# 7.2 Train Random Forest
# -----------------------------------------------------------------------------
set.seed(1234)
rf_fit <- ranger(
  default_payment_next_month ~ .,
  data        = train_rf,
  num.trees   = 500,                                # 500 trees is the standard default
  mtry        = floor(sqrt(26)),                    # sqrt(p) = standard for classification
  importance  = "permutation",                       # more reliable than Gini importance
  probability = TRUE,                                # output probabilities for AUC
  num.threads = parallel::detectCores() - 1,
  seed        = 1234
)

print(rf_fit)


# -----------------------------------------------------------------------------
# 7.3 Variable importance
# -----------------------------------------------------------------------------
importance_df <- data.frame(
  variable   = names(rf_fit$variable.importance),
  importance = round(rf_fit$variable.importance, 4)
) |> arrange(desc(importance))

cat("\n--- Top 10 RF Features (Permutation Importance) ---\n")
print(head(importance_df, 10))

importance_df |>
  head(10) |>
  ggplot(aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Random Forest — Top 10 Feature Importance",
       x = NULL, y = "Permutation Importance") +
  theme_minimal(base_size = 12)


# -----------------------------------------------------------------------------
# 7.4 Predict + evaluate
# -----------------------------------------------------------------------------
rf_pred       <- predict(rf_fit, data = test_rf)
pred_prob_rf  <- rf_pred$predictions[, "default"]

# evaluate_model expects 0/1 numeric. as.integer(factor) returns 1/2 (R is
# 1-indexed for factor levels), so subtract 1 to get 0/1.
y_test_numeric <- as.integer(test_rf$default_payment_next_month) - 1

cat("\n========== Random Forest ==========\n")
result_rf <- evaluate_model(pred_prob_rf, y_test_numeric, model_name = "Random Forest")
get_performance(result_rf$Confusion_Matrix)
cat("\nRF AUC:", round(result_rf$AUC, 4), "\n")
