# =============================================================================
# Section 9: Model Comparison
# =============================================================================
# Honest comparison caveat: XGBoost uses scale_pos_weight while LASSO and RF
# do not — that's why XGBoost has much higher Sensitivity but lower Precision.
# AUC and F1 are the most fair comparators since they don't depend on the
# decision threshold strategy.

# -----------------------------------------------------------------------------
# 9.1 Comparison table
# -----------------------------------------------------------------------------
comparison <- tibble(
  Model       = c("LASSO Logistic", "Random Forest", "XGBoost"),
  AUC         = c(result_lr$AUC, result_rf$AUC, result_xgb$AUC),
  Sensitivity = c(0.229, 0.366, 0.631),
  Specificity = c(0.973, 0.944, 0.792),
  Precision   = c(0.704, 0.651, 0.462),
  F1          = c(0.346, 0.469, 0.534)
) |>
  mutate(across(where(is.numeric), \(x) round(x, 3)))

print(comparison)


# -----------------------------------------------------------------------------
# 9.2 Overlaid ROC curves
# -----------------------------------------------------------------------------
plot_roc_data <- function(pred_prob, y_real, model_name) {
  pred_obj <- prediction(pred_prob, y_real)
  perf     <- performance(pred_obj, "tpr", "fpr")
  data.frame(
    fpr   = perf@x.values[[1]],
    tpr   = perf@y.values[[1]],
    model = model_name
  )
}

roc_combined <- bind_rows(
  plot_roc_data(pred_prob,     y_test,         "LASSO (AUC=0.717)"),
  plot_roc_data(pred_prob_rf,  y_test_numeric, "Random Forest (AUC=0.776)"),
  plot_roc_data(pred_prob_xgb, y_test_xgb,     "XGBoost (AUC=0.780)")
)

ggplot(roc_combined, aes(x = fpr, y = tpr, color = model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  scale_color_manual(values = c("LASSO (AUC=0.717)"         = "#E41A1C",
                                "Random Forest (AUC=0.776)" = "#377EB8",
                                "XGBoost (AUC=0.780)"       = "#4DAF4A")) +
  labs(title    = "ROC Curves Comparison",
       subtitle = "Tree-based models clearly outperform the linear baseline",
       x = "False Positive Rate", y = "True Positive Rate", color = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = c(0.7, 0.2),
        plot.title = element_text(face = "bold"))


# -----------------------------------------------------------------------------
# 9.3 Multi-metric bar chart
# -----------------------------------------------------------------------------
comparison |>
  pivot_longer(cols = c(AUC, F1, Sensitivity, Specificity),
               names_to = "Metric", values_to = "Value") |>
  ggplot(aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_text(aes(label = round(Value, 3)),
            position = position_dodge(width = 0.7),
            vjust = -0.4, size = 3) +
  scale_fill_manual(values = c("AUC"         = "#1B9E77",
                               "F1"          = "#D95F02",
                               "Sensitivity" = "#7570B3",
                               "Specificity" = "#E7298A")) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.2)) +
  labs(title    = "Model Comparison Across Multiple Metrics",
       subtitle = "AUC and F1 are the fairest comparators; Sensitivity/Specificity depend on threshold strategy",
       x = NULL, y = "Score") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold"))


# =============================================================================
# End of pipeline
# =============================================================================
# Summary of findings:
#   1. pay_0 (most recent payment status) is the strongest single predictor
#      in all three models — robust signal across linear and tree-based methods.
#   2. Tree models (RF, XGBoost) outperform LASSO by ~0.06 AUC, indicating
#      the data has meaningful non-linear structure and feature interactions
#      that linear models cannot capture.
#   3. Class imbalance (22% defaults) means accuracy is misleading; use
#      AUC + F1 + class-weighted Sensitivity for honest evaluation.
#   4. For production, choose a model + threshold based on the bank's cost
#      matrix: high Sensitivity if missing a default is costly; high
#      Precision if false alarms hurt customer relationships.
#
# Suggested next steps:
#   - Feature engineering on bill_amt sequences (volatility, trend)
#   - Hyperparameter tuning via tidymodels::tune_grid
#   - SHAP values for individual prediction explanation
#   - Production-ize via plumber API + Docker
# =============================================================================
