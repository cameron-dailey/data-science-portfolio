library(tidyverse)
library(tm)          
library(tidytext)

# Load data
reviews <- read_csv("data/amazon_reviews_sample.csv")

# Clean text
cleaned_reviews <- reviews %>%
  mutate(review_text = str_to_lower(review_text)) %>%
  mutate(review_text = str_replace_all(review_text, "[^a-z\\s]", " ")) %>%
  rowwise() %>%
  mutate(review_text = removeWords(review_text, stopwords("en"))) %>%
  ungroup()

# Save cleaned file
write_csv(cleaned_reviews, "data/cleaned_reviews.csv")