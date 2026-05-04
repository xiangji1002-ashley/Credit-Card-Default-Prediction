# =============================================================================
# Section 2: Data Pre-processing — Factor Conversion
# =============================================================================
# Why this design:
#   - SEX:        straightforward 1=male, 2=female recode
#   - EDUCATION:  UCI documents only 1/2/3; values 0,4,5,6 appear in data
#                 (~1.5% of rows) and are likely "unknown / other". We collapse
#                 them into "others" rather than dropping (avoids selection bias).
#   - MARRIAGE:   value 0 also appears (undocumented); collapse with 3 -> "others".
#
# We do NOT use ordered factors here, even though high_school < university < grad
# has a natural order. Reason: "others" is "unknown", not "lowest education level",
# so forcing it into the ordinal scale would introduce a false linear assumption.
# Tree models and dummy encoding handle this better.

dataset <- raw_data |>
  mutate(
    sex = factor(sex, levels = c(1, 2), labels = c("male", "female")),
    
    education = case_when(
      education %in% c(0, 4, 5, 6) ~ "others",
      education == 3               ~ "high_school",
      education == 2               ~ "university",
      education == 1               ~ "grad",
      TRUE                         ~ NA_character_
    ) |> factor(levels = c("others", "high_school", "university", "grad")),
    
    marriage = case_when(
      marriage == 1         ~ "married",
      marriage == 2         ~ "single",
      marriage %in% c(0, 3) ~ "others",
      TRUE                  ~ NA_character_
    ) |> factor(levels = c("married", "single", "others"))
  )

# Verify class distributions
dataset |> count(sex)
dataset |> count(education)
dataset |> count(marriage)
dataset |> count(default_payment_next_month)
# Expected: ~22% default rate (6,636 / 30,000)