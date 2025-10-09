library(tidyverse)
library(tidytext)
library(igraph)
library(ggraph)

# Load data
cleaned_reviews <- read_csv("data/cleaned_reviews.csv")

# Create bigrams
bigrams <- cleaned_reviews %>%
  unnest_tokens(bigram, review_text, token = "ngrams", n = 2)

# Separate and filter
bigrams_separated <- bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word)

# Count bigrams
bigram_counts <- bigrams_separated %>%
  count(word1, word2, sort = TRUE)

# Only keep strong links (adjust threshold if dataset grows)
bigram_graph <- bigram_counts %>%
  filter(n >= 1) %>%
  graph_from_data_frame()

set.seed(123)
p <- ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(width = n), edge_colour = "gray70", alpha = 0.8) +
  geom_node_point(color = "steelblue", size = 5) +
  geom_node_text(aes(label = name), repel = TRUE, size = 4, color = "black") +
  theme_void() +
  labs(title = "Bigram Co-occurrence Network") +
  theme(plot.title = element_text(hjust = 0.5))

# Save plot
if (!dir.exists("outputs")) dir.create("outputs")
ggsave("outputs/bigram_network.png", p, width = 7, height = 5, dpi = 300)