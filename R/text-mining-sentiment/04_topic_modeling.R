library(topicmodels)
library(tidytext)
library(tidyverse)

cleaned_reviews <- read_csv("data/cleaned_reviews.csv")

# Create DTM
dtm <- cleaned_reviews %>%
  unnest_tokens(word, review_text) %>%
  count(id = row_number(), word) %>%
  cast_dtm(id, word, n)

# Fit LDA
lda_model <- LDA(dtm, k = 2, control = list(seed = 123))

topics <- tidy(lda_model, matrix = "beta")

top_terms <- topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

ggplot(top_terms, aes(beta, reorder_within(term, beta, topic), fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  labs(title = "Top Terms per Topic")

ggsave("outputs/lda_topics.png")