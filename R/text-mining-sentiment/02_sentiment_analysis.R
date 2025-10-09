# Sentiment Analysis Script (Final Stable Version)
library(tidyverse)
library(tidytext)
library(syuzhet)
library(wordcloud)
library(RColorBrewer)
library(ggplot2)

# Create outputs folder if missing
if (!dir.exists("outputs")) dir.create("outputs")

# Load cleaned data
cleaned_reviews <- read_csv("data/cleaned_reviews.csv")

# Tokenize
tokens <- cleaned_reviews %>%
  unnest_tokens(word, review_text)

# Join with sentiment lexicon
bing <- get_sentiments("bing")
sentiment_data <- tokens %>%
  inner_join(bing, by = "word")

cat("Matched sentiment words:", nrow(sentiment_data), "\n")

if (nrow(sentiment_data) > 0) {
  # Count and visualize
  sentiment_summary <- sentiment_data %>%
    count(sentiment, sort = TRUE)
  
  p <- ggplot(sentiment_summary, aes(x = sentiment, y = n, fill = sentiment)) +
    geom_col() +
    theme_minimal() +
    labs(title = "Sentiment Distribution", x = "", y = "Count")
  
  ggsave("outputs/sentiment_trend.png", p, width = 6, height = 4)
  
} else {
  cat("No sentiment matches found — showing word frequency plot instead.\n")
  
  freq <- tokens %>%
    count(word, sort = TRUE) %>%
    slice_max(n, n = 10)
  
  p <- ggplot(freq, aes(x = reorder(word, n), y = n)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    theme_minimal() +
    labs(title = "Top Words (No Sentiment Matches)", x = "", y = "Frequency")
  
  ggsave("outputs/sentiment_trend.png", p, width = 6, height = 4)
}

# Generate non-interactive word cloud
set.seed(123)
pal <- brewer.pal(8, "Dark2")

wordcloud(
  words = tokens$word,
  max.words = 100,
  random.order = FALSE,
  colors = pal
)
