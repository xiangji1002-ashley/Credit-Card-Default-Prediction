# =============================================================================
# Section 5: Helper Functions for Model Evaluation
# =============================================================================
# Three helpers for consistent evaluation across all models:
#   evaluate_model()   — AUC + ROC curve + confusion matrix
#   get_performance()  — Sensitivity/Specificity/Precision/Accuracy/F1
#   get_lasso_coefs()  — Extract non-zero coefs with odds ratios

# -----------------------------------------------------------------------------
# evaluate_model: compute AUC, plot ROC, return confusion matrix
# -----------------------------------------------------------------------------
evaluate_model <- function(pred_prob, y_real, threshold = 0.5,
                           model_name = "Model") {
  pred_obj <- prediction(pred_prob, y_real)
  auc      <- performance(pred_obj, "auc")@y.values[[1]]
  
  perf <- performance(pred_obj, "tpr", "fpr")
  plot(perf,
       main = paste0(model_name, " ROC Curve (AUC = ", round(auc, 3), ")"),
       col  = "steelblue", lwd = 2)
  abline(a = 0, b = 1, lty = 2, col = "grey60")
  
  pred_class <- as.integer(pred_prob > threshold)
  conf_mat   <- table(predicted = pred_class, actual = y_real)
  
  list(AUC = auc, Confusion_Matrix = conf_mat)
}


# -----------------------------------------------------------------------------
# get_performance: derive 5 classification metrics from confusion matrix
# -----------------------------------------------------------------------------
# IMPORTANT: indexes by character names ("0", "1") rather than positions
# [1,1]/[2,2]. This is defensive against any factor-level reordering and was
# the source of a subtle bug in the 2019 original.
get_performance <- function(conf_mat) {
  tn <- conf_mat["0", "0"]
  fp <- conf_mat["1", "0"]
  fn <- conf_mat["0", "1"]
  tp <- conf_mat["1", "1"]
  
  metrics <- c(
    Sensitivity = tp / (tp + fn),  # Recall: of true defaults, how many caught
    Specificity = tn / (tn + fp),  # Of true non-defaults, how many identified
    Precision   = tp / (tp + fp),  # Of predicted defaults, how many real
    Accuracy    = (tp + tn) / sum(conf_mat),
    F1          = 2 * tp / (2 * tp + fp + fn)
  )
  print(round(metrics, 3))
  invisible(metrics)
}


# -----------------------------------------------------------------------------
# get_lasso_coefs: extract non-zero LASSO coefficients with odds ratios
# -----------------------------------------------------------------------------
# odds_ratio = exp(coefficient): on log-odds scale, this is the multiplicative
# effect of a one-unit increase in the (standardized) feature. odds_ratio > 1
# means feature increases default odds; < 1 means it decreases them.
get_lasso_coefs <- function(fit, lambda) {
  coefs       <- coef(fit, s = lambda)
  active_idx  <- which(coefs != 0)
  
  data.frame(
    variable    = rownames(coefs)[active_idx],
    coefficient = round(as.vector(coefs)[active_idx], 4),
    odds_ratio  = round(exp(as.vector(coefs)[active_idx]), 3)
  ) |> dplyr::arrange(desc(abs(coefficient)))
}

