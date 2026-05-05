# =============================================================================
# Section 4: Train/Test Split + Pre-processing Pipeline
# =============================================================================
# Critical design: split FIRST, then pre-process — this prevents data leakage.
# The 2019 original applied scale() to the entire dataset before splitting,
# meaning test-set statistics leaked into the training transform.

# -----------------------------------------------------------------------------
# 4.1 Stratified split (preserves class balance)
# -----------------------------------------------------------------------------
# Stratified sampling guarantees that train and test have the same default
# rate as the original data (~22%). With pure random sampling on imbalanced
# data, the test set's class ratio could drift, making evaluation unreliable.

set.seed(1234)
split <- initial_split(dataset, prop = 0.7,
                       strata = default_payment_next_month)

dataset_train <- training(split)
dataset_test  <- testing(split)

# Sanity check — both should be ≈ 0.221
cat("Train default rate:", round(mean(dataset_train$default_payment_next_month), 4), "\n")
cat("Test default rate: ", round(mean(dataset_test$default_payment_next_month),  4), "\n")
cat("Train rows:", nrow(dataset_train), " | Test rows:", nrow(dataset_test), "\n")


# -----------------------------------------------------------------------------
# 4.2 Recipes pipeline — three-step workflow
# -----------------------------------------------------------------------------
# recipe()  -> defines what to do (no computation)
# prep()    -> learns parameters (mean/sd) from TRAINING data only
# bake()    -> applies the learned transformations to any dataset
#
# This pattern mirrors sklearn's fit/transform and is the modern standard
# for leakage-free pre-processing in R.

recipe_obj <- recipe(default_payment_next_month ~ ., data = dataset_train) |>
  step_rm(id) |>                               # ID is not a feature
  step_dummy(all_nominal_predictors()) |>      # one-hot encode factors (N-1 cols)
  step_normalize(all_numeric_predictors())     # standardize: mean=0, sd=1

recipe_prepped <- prep(recipe_obj, training = dataset_train)

# new_data = NULL is a convention: use the training data from prep()
train_baked <- bake(recipe_prepped, new_data = NULL)
test_baked  <- bake(recipe_prepped, new_data = dataset_test)

cat("\nTrain baked dim:", dim(train_baked), "\n")
cat("Test baked dim: ", dim(test_baked),  "\n")
glimpse(train_baked)
# -----------------------------------------------------------------------------
# 4.3 Save baked data to disk for downstream debugging / sharing
# -----------------------------------------------------------------------------
# Optional but useful: persist the processed datasets so colleagues (or a
# future "you") can resume from here without re-running the full pipeline.
# These are gitignored to avoid bloating the repo.

write_csv(train_baked, "data/train_baked.csv")
write_csv(test_baked,  "data/test_baked.csv")
cat("\nBaked datasets saved to data/\n")