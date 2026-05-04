# =============================================================================
# Section 1: Data Ingestion
# =============================================================================
# We use read_csv() and skip the first row (which contains placeholder labels
# X1, X2, etc.; real column names are on row 2).
# clean_names() converts "default payment next month" -> "default_payment_next_month".

raw_data <- read_csv("UCI_Credit_Card.csv", skip = 1) |>
  clean_names()

# Sanity checks: should be 30,000 rows × 25 columns, all numeric
glimpse(raw_data)
dim(raw_data)
colSums(is.na(raw_data))   # Expect all zeros — UCI dataset is clean