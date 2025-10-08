# clustering_analysis.R
# K-means segmentation with profiling and diagnostics
library(tidyverse)
library(cluster)
library(factoextra)
library(ggrepel)

theme_set(theme_minimal())

plots_dir <- "plots"
if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE)

df <- readr::read_csv("data/customers.csv")

glimpse(df)

feat <- df %>%
  select(recency_weeks, monetary_90d, engagement_rate)

# Scale
feat_s <- scale(feat)

# Elbow
wss <- map_dbl(1:10, ~kmeans(feat_s, centers = .x, nstart = 25)$tot.withinss)
elbow_df <- tibble(k = 1:10, wss = wss)
p_elbow <- ggplot(elbow_df, aes(k, wss)) +
  geom_line() + geom_point() +
  labs(title = "Elbow method", x = "k", y = "Total within SS")
ggsave(file.path(plots_dir, "elbow.png"), p_elbow, width = 6, height = 4, dpi = 120)

# Choose k = 3 as a good compromise (example)
set.seed(123)
km <- kmeans(feat_s, centers = 3, nstart = 50)

segmented <- df %>%
  mutate(cluster = factor(km$cluster))

# Cluster centers (back-transform approximate by mean per cluster)
profiles <- segmented %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    recency_weeks = mean(recency_weeks),
    monetary_90d = mean(monetary_90d),
    engagement_rate = mean(engagement_rate),
    churn_rate = mean(churned)
  ) %>% arrange(desc(monetary_90d))

print(profiles)
readr::write_csv(profiles, file.path(plots_dir, "cluster_profiles.csv"))

# Visualize in 2D via PCA
pca <- prcomp(feat_s, scale. = FALSE)
p_pca <- as_tibble(pca$x[,1:2]) %>%
  mutate(cluster = segmented$cluster) %>%
  ggplot(aes(PC1, PC2, color = cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "Customer segments (PCA of features)")
ggsave(file.path(plots_dir, "pca_clusters.png"), p_pca, width = 7, height = 5, dpi = 120)

# Churn by cluster barplot
p_churn <- segmented %>%
  group_by(cluster) %>%
  summarise(churn_rate = mean(churned)) %>%
  ggplot(aes(cluster, churn_rate, fill = cluster)) +
  geom_col(show.legend = FALSE) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Churn rate by cluster", x = "Cluster", y = "Churn rate")
ggsave(file.path(plots_dir, "churn_by_cluster.png"), p_churn, width = 6, height = 4, dpi = 120)