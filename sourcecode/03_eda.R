# =============================================================================
# Section 3: Exploratory Data Analysis
# =============================================================================

# -----------------------------------------------------------------------------
# 3.1 Target variable distribution
# -----------------------------------------------------------------------------
dataset |>
  mutate(default_label = factor(default_payment_next_month,
                                levels = c(0, 1),
                                labels = c("Not default", "Default"))) |>
  ggplot(aes(x = default_label, fill = default_label)) +
  geom_bar(color = "black", width = 0.6) +
  geom_text(stat = "count", aes(label = scales::comma(after_stat(count))),
            vjust = -0.5, size = 4) +
  scale_fill_manual(values = c("Not default" = "#4DA6FF", "Default" = "#FF8C42")) +
  scale_y_continuous(labels = scales::comma,
                     expand = expansion(mult = c(0, 0.1))) +
  labs(title    = "Target Variable Distribution",
       subtitle = "Default rate: 22.1% (6,636 / 30,000)",
       x = NULL, y = "Number of customers", fill = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold"))


# -----------------------------------------------------------------------------
# 3.2 Three-way facet: education × marriage × sex on default rate
# -----------------------------------------------------------------------------
# Design choices:
#   - position = "fill": normalize each bar to 100% so red height = default rate
#     (vs "stack" which shows absolute counts and is misleading when subgroup
#     sizes differ widely)
#   - filter() inside the pipeline: temporary view, doesn't modify `dataset`
#   - droplevels(): removes empty factor levels left behind by filter, so
#     facet_grid won't show empty panels for "others"

dataset |>
  filter(education != "others", marriage != "others") |>
  mutate(
    education = droplevels(education),
    marriage  = droplevels(marriage),
    default_label = factor(default_payment_next_month,
                           levels = c(0, 1),
                           labels = c("Not default", "Default"))
  ) |>
  ggplot(aes(x = sex, fill = default_label)) +
  geom_bar(position = "fill", color = "white", linewidth = 0.3) +
  facet_grid(marriage ~ education) +
  scale_fill_manual(values = c("Not default" = "#4DA6FF", "Default" = "#FF6B6B")) +
  scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
  labs(title    = "Default Rate by Education × Marriage × Sex",
       subtitle = "Higher red = higher default rate within that subgroup",
       x = "Sex", y = "Proportion", fill = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold"),
        strip.background = element_rect(fill = "grey90", color = NA),
        panel.spacing = unit(1, "lines"))


# -----------------------------------------------------------------------------
# 3.3 Single-factor plots (sex / education / marriage) using a function
# -----------------------------------------------------------------------------
# Function-based DRY refactor: same plot recipe applied to three variables.
# Key technique: .data[[var_name]] dynamically accesses a column by string name
# inside aes() — required when writing reusable ggplot functions.

plot_data <- dataset |>
  mutate(default_label = factor(default_payment_next_month,
                                levels = c(0, 1),
                                labels = c("Not default", "Default")))

make_factor_plot <- function(var_name, title) {
  ggplot(plot_data, aes(x = .data[[var_name]], fill = default_label)) +
    geom_bar(position = "fill", color = "white", linewidth = 0.3) +
    scale_fill_manual(values = c("Not default" = "#4DA6FF",
                                 "Default"     = "#FF6B6B")) +
    scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
    labs(title = title, x = NULL, y = "Proportion", fill = NULL) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold", size = 12),
          axis.text.x = element_text(angle = 30, hjust = 1))
}

p_sex       <- make_factor_plot("sex",       "By Sex")
p_education <- make_factor_plot("education", "By Education")
p_marriage  <- make_factor_plot("marriage",  "By Marriage")

# patchwork: + arranges horizontally, / vertically
# plot_layout(guides = "collect") merges duplicate legends into one
# & (vs +) applies theme to ALL sub-plots, not just the last one
combined <- (p_sex + p_education + p_marriage) +
  plot_annotation(
    title    = "Default Rate by Single Factor",
    subtitle = "Higher red = higher default rate",
    theme    = theme(plot.title = element_text(face = "bold", size = 14))
  ) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

combined


# -----------------------------------------------------------------------------
# 3.4 Density plot — credit limit by default status
# -----------------------------------------------------------------------------
# Density plot is preferred over histogram for group comparison: alpha
# transparency lets overlapping distributions remain readable.

dataset |>
  mutate(default_label = factor(default_payment_next_month,
                                levels = c(0, 1),
                                labels = c("Not default", "Default"))) |>
  ggplot(aes(x = limit_bal, fill = default_label)) +
  geom_density(alpha = 0.5, color = NA) +
  scale_fill_manual(values = c("Not default" = "#4DA6FF",
                               "Default"     = "#FF6B6B")) +
  scale_x_continuous(labels = scales::comma) +
  labs(title    = "Credit Limit Distribution by Default Status",
       subtitle = "Defaulters tend to have lower credit limits",
       x = "Credit Limit (NT$)", y = "Density", fill = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold"))


# -----------------------------------------------------------------------------
# 3.5 Correlation matrix
# -----------------------------------------------------------------------------
# Why we look at this: identify multicollinearity (problematic for linear
# models) and find redundant features. We expect bill_amt1..6 to be highly
# correlated since they are sequential monthly bills.
#
# Modern improvements over base R [c(2, 7:25)] indexing:
#   - select() with column names: robust to column reordering
#   - where(is.numeric): auto-selects numeric columns
#   - use = "complete.obs": defensive against future NAs

numeric_data <- dataset |>
  select(-id, -sex, -education, -marriage) |>
  select(where(is.numeric))

cor_matrix <- cor(numeric_data, use = "complete.obs")

corrplot(cor_matrix,
         method       = "color",       # color squares
         order        = "AOE",         # eigenvector ordering — clusters related vars
         type         = "upper",       # only upper triangle (matrix is symmetric)
         addCoef.col  = "black",
         tl.cex       = 0.7,
         tl.col       = "black",
         number.cex   = 0.6,
         diag         = FALSE)

# =============================================================================
# Section 3.6: Chi-square tests of independence (categorical features vs default)
# =============================================================================
# Goal: quantitatively confirm whether the visual associations seen in the
# faceted bar charts are statistically significant. Each test answers:
#   "Is the joint distribution of (factor, default) different from what we'd
#    expect if they were independent?"
# Small p-value => reject independence => the factor carries signal.
#
# Note: chi-square tests assume sufficient cell counts (typically >= 5 per
# cell). With 30K samples this is comfortably met.

cat("\n========== Chi-square Tests: Categorical Features vs Default ==========\n\n")

# Build cross-tabs (contingency tables) using base R's xtabs()
sex_xtab       <- xtabs(~ sex       + default_payment_next_month, data = dataset)
education_xtab <- xtabs(~ education + default_payment_next_month, data = dataset)
marriage_xtab  <- xtabs(~ marriage  + default_payment_next_month, data = dataset)

# Print the cross-tabs with marginal totals (gives full picture)
cat("--- SEX × Default ---\n")
print(addmargins(sex_xtab))

cat("\n--- EDUCATION × Default ---\n")
print(addmargins(education_xtab))

cat("\n--- MARRIAGE × Default ---\n")
print(addmargins(marriage_xtab))

# Run chi-square tests
cat("\n========== Test Results ==========\n")

cat("\n--- SEX vs Default ---\n")
print(chisq.test(sex_xtab))

cat("\n--- EDUCATION vs Default ---\n")
print(chisq.test(education_xtab))

cat("\n--- MARRIAGE vs Default ---\n")
print(chisq.test(marriage_xtab))

# Tidy summary: extract p-values into one comparison table
chisq_summary <- tibble(
  feature = c("sex", "education", "marriage"),
  p_value = c(
    chisq.test(sex_xtab)$p.value,
    chisq.test(education_xtab)$p.value,
    chisq.test(marriage_xtab)$p.value
  ),
  test_statistic = c(
    chisq.test(sex_xtab)$statistic,
    chisq.test(education_xtab)$statistic,
    chisq.test(marriage_xtab)$statistic
  )
) |>
  mutate(
    significant_at_001 = p_value < 0.001,
    p_value = formatC(p_value, format = "e", digits = 2)   # scientific notation
  ) |>
  arrange(desc(test_statistic))

cat("\n--- Summary ---\n")
print(chisq_summary)

# =============================================================================
# Section 3.7: Credit limit distribution by demographic factors
# =============================================================================
# Three additional density plots — limit_bal sliced by sex / education / marriage
# These reveal demographic patterns in credit allocation that may correlate
# with default risk. Useful for "feature interaction" intuition before modeling.

library(patchwork)

# Helper: build limit_bal density plot grouped by any factor
make_limit_density <- function(group_var, title) {
  ggplot(dataset, aes(x = limit_bal, fill = .data[[group_var]])) +
    geom_density(alpha = 0.5, color = NA) +
    scale_x_continuous(labels = scales::comma) +
    scale_fill_brewer(palette = "Set2") +
    labs(title = title, x = "Credit Limit (NT$)", y = "Density", fill = NULL) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom",
          plot.title = element_text(face = "bold", size = 11))
}

p_limit_sex       <- make_limit_density("sex",       "By Sex")
p_limit_education <- make_limit_density("education", "By Education")
p_limit_marriage  <- make_limit_density("marriage",  "By Marriage")

# Combine into one figure with patchwork
limit_combined <- (p_limit_sex / p_limit_education / p_limit_marriage) +
  plot_annotation(
    title    = "Credit Limit Distribution Across Demographic Factors",
    subtitle = "Higher-education customers receive systematically higher limits",
    theme    = theme(plot.title = element_text(face = "bold", size = 14))
  )

print(limit_combined)
