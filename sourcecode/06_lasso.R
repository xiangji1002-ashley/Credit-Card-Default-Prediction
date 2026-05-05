# =============================================================================
# Section 6: Model 1 — LASSO Logistic Regression
# =============================================================================
# LASSO = L1-regularized logistic regression. Why use it over plain logistic:
#   1. Automatic feature selection (some coefficients become exactly zero)
#   2. Handles multicollinearity (e.g., bill_amt1..6) by selecting one and
#      zeroing the rest
#   3. cv.glmnet() handles cross-validation for lambda automatically
# -----------------------------------------------------------------------------
# 6.0 Plain Logistic Regression — baseline before regularization
# -----------------------------------------------------------------------------
# Run a vanilla glm() first as a sanity-check baseline. Comparing it to LASSO
# tells us how much regularization helps with this multicollinear dataset
# (recall: bill_amt1..6 have correlations of 0.85+).

# Use same data as LASSO (already split + baked)
glm_data_train <- train_baked
glm_data_test  <- test_baked

glm_fit <- glm(
  default_payment_next_month ~ .,
  family = binomial(),
  data   = glm_data_train
)

# Predict probabilities on test set
glm_pred_prob <- predict(glm_fit, newdata = glm_data_test, type = "response")

cat("\n========== Plain GLM Logistic Regression (Baseline) ==========\n")
result_glm <- evaluate_model(glm_pred_prob, y_test, model_name = "GLM Baseline")
get_performance(result_glm$Confusion_Matrix)
cat("\nGLM Baseline AUC:", round(result_glm$AUC, 4), "\n")

# Number of "active" predictors (any with non-zero coefficient — for GLM that's all)
cat("GLM uses all", length(coef(glm_fit)) - 1, "predictors\n")
cat("LASSO will use only the non-zero subset (typically 15-19 of 26)\n")
# -----------------------------------------------------------------------------
# 6.1 Prepare matrix inputs (glmnet requires matrix, not data.frame)
# -----------------------------------------------------------------------------
y_train <- train_baked$default_payment_next_month
y_test  <- test_baked$default_payment_next_month

x_train <- train_baked |> select(-default_payment_next_month) |> as.matrix()
x_test  <- test_baked  |> select(-default_payment_next_month) |> as.matrix()


# -----------------------------------------------------------------------------
# 6.2 Fit cv.glmnet — internally tries 100 lambdas and selects best via 5-fold CV
# -----------------------------------------------------------------------------
set.seed(1234)
lr_cv <- cv.glmnet(
  x_train, y_train,
  family       = "binomial",   # logistic regression
  alpha        = 1,            # alpha=1 = LASSO; alpha=0 = Ridge; in-between = Elastic Net
  type.measure = "auc",        # use AUC to select lambda
  nfolds       = 5
)

plot(lr_cv)
cat("lambda.min:", lr_cv$lambda.min, "  lambda.1se:", lr_cv$lambda.1se, "\n")
# lambda.min — minimizes CV error
# lambda.1se — most regularized model within 1 SE of min (sparser, more robust)


# -----------------------------------------------------------------------------
# 6.3 Predict on test set + evaluate
# -----------------------------------------------------------------------------
pred_prob <- predict(lr_cv, newx = x_test,
                     s = lr_cv$lambda.1se,
                     type = "response") |> as.vector()

cat("\n========== LASSO Logistic Regression ==========\n")
result_lr <- evaluate_model(pred_prob, y_test, model_name = "LASSO LR")
get_performance(result_lr$Confusion_Matrix)

cat("\n--- Non-zero LASSO coefficients ---\n")
print(get_lasso_coefs(lr_cv, lr_cv$lambda.1se))

cat("\nLASSO AUC:", round(result_lr$AUC, 4), "\n")

  
