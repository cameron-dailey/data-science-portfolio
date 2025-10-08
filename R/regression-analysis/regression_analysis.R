# regression_analysis.R
# Predict revenue with linear and polynomial regression, compare with CV
# Libraries
library(tidyverse)
library(modelr)
library(broom)
library(rsample)
library(yardstick)
library(ggrepel)

theme_set(theme_minimal())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# Paths
plots_dir <- "plots"
if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE)

# Load data
df <- readr::read_csv("data/marketing.csv")

# Quick EDA
glimpse(df)

# Pairwise relationships
p1 <- df %>%
  select(revenue, tv_spend, search_spend, social_spend, price) %>%
  pivot_longer(-revenue) %>%
  ggplot(aes(value, revenue)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "loess", se = FALSE) +
  facet_wrap(~name, scales = "free") +
  labs(title = "Revenue vs predictors", x = NULL, y = "Revenue")
ggsave(file.path(plots_dir, "eda_scatter.png"), p1, width = 10, height = 6, dpi = 120)

# Feature engineering
df2 <- df %>%
  mutate(
    log_tv = log1p(tv_spend),
    log_search = log1p(search_spend),
    log_social = log1p(social_spend),
    promo = factor(promo),
    season = factor(season),
    price_c = scale(price, center = TRUE, scale = TRUE)[,1]
  )

# Split
set.seed(123)
split <- initial_split(df2, prop = 0.8, strata = revenue)
train <- training(split)
test  <- testing(split)

# Linear model
m_lin <- lm(revenue ~ log_tv + log_search + log_social + price_c + promo + season, data = train)

# Polynomial terms for price (degree 2) and interactions with promo
m_poly <- lm(revenue ~ log_tv + log_search + log_social +
               poly(price_c, 2, raw = TRUE) * promo + season,
             data = train)

# Cross-validation
set.seed(123)
cv5 <- vfold_cv(train, v = 5, strata = revenue)

rmse_model <- function(mod_formula, data_train, folds) {
  map_dfr(folds$splits, \(sp) {
    analysis <- analysis(sp)
    assess   <- assessment(sp)
    fit <- lm(mod_formula, data = analysis)
    preds <- tibble(
      truth = assess$revenue,
      .pred = predict(fit, newdata = assess)
    )
    yardstick::rmse_vec(truth = preds$truth, estimate = preds$.pred)
  }) %>%
    summarise(mean_rmse = mean(value), sd_rmse = sd(value))
}

cv_lin  <- rmse_model(formula(m_lin), train, cv5) %>% mutate(model = "linear")
cv_poly <- rmse_model(formula(m_poly), train, cv5) %>% mutate(model = "polynomial")

cv_results <- bind_rows(cv_lin, cv_poly)
print(cv_results)
readr::write_csv(cv_results, file.path(plots_dir, "cv_rmse.csv"))

# Diagnostics
aug_lin <- augment(m_lin, data = train)
p_resid_lin <- ggplot(aug_lin, aes(.fitted, .resid)) +
  geom_point(alpha = 0.5) + geom_hline(yintercept = 0, linetype = 2) +
  labs(title = "Linear model residuals", x = "Fitted", y = "Residuals")
ggsave(file.path(plots_dir, "diagnostics_linear_resid.png"), p_resid_lin, width = 7, height = 5, dpi = 120)

aug_poly <- augment(m_poly, data = train)
p_resid_poly <- ggplot(aug_poly, aes(.fitted, .resid)) +
  geom_point(alpha = 0.5) + geom_hline(yintercept = 0, linetype = 2) +
  labs(title = "Polynomial model residuals", x = "Fitted", y = "Residuals")
ggsave(file.path(plots_dir, "diagnostics_poly_resid.png"), p_resid_poly, width = 7, height = 5, dpi = 120)

# Final evaluation on test set
pred_lin  <- predict(m_lin, newdata = test)
pred_poly <- predict(m_poly, newdata = test)

rmse_lin  <- yardstick::rmse_vec(truth = test$revenue, estimate = pred_lin)
rmse_poly <- yardstick::rmse_vec(truth = test$revenue, estimate = pred_poly)

tibble(model = c("linear","polynomial"), rmse = c(rmse_lin, rmse_poly)) %>% print()

# Partial dependence style plot for price by promo for polynomial model
newgrid <- expand_grid(
  price_c = seq(min(train$price_c), max(train$price_c), length.out = 100),
  promo = levels(train$promo),
  log_tv = median(train$log_tv),
  log_search = median(train$log_search),
  log_social = median(train$log_social),
  season = levels(train$season)[1]
) %>%
  mutate(pred = predict(m_poly, newdata = cur_data()))

p_pd <- ggplot(newgrid, aes(price_c, pred, color = promo)) +
  geom_line(size = 1) +
  labs(title = "Effect of standardized price by promo (poly model)",
       x = "Price (z-score)", y = "Predicted revenue")
ggsave(file.path(plots_dir, "partial_price_promo.png"), p_pd, width = 8, height = 5, dpi = 120)

# Coefficients
coef_tbl <- broom::tidy(m_poly) %>% arrange(desc(abs(estimate)))
readr::write_csv(coef_tbl, file.path(plots_dir, "coefficients_poly.csv"))