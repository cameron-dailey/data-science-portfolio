# Text Mining & Sentiment Analysis in R

This project demonstrates text mining, sentiment analysis, and topic modeling using real-world text data (e.g., Amazon product reviews or tweets). The workflow covers data cleaning, tokenization, sentiment scoring, visualization, and topic extraction using R’s tidy text ecosystem.

## Project Structure
```
text-mining-sentiment/
├── data/                  # Raw and cleaned text data
├── scripts/               # Modular R scripts for each analysis step
├── outputs/               # Visualizations and result artifacts
└── README.md              # Project overview
```

## Objectives
- Perform text cleaning, tokenization, and word frequency analysis.
- Conduct sentiment analysis using the Bing and NRC lexicons.
- Visualize sentiment trends and top positive/negative words.
- Build word clouds and bigram co-occurrence networks.
- (Optional) Perform topic modeling with Latent Dirichlet Allocation (LDA).

## Key Packages
`tidyverse`, `tidytext`, `tm`, `wordcloud`, `syuzhet`, `igraph`, `ggraph`, `topicmodels`

Install them with:
```R
install.packages(c("tidyverse", "tidytext", "tm", "wordcloud", "syuzhet", "igraph", "ggraph", "topicmodels"))
```

## How to Run
1. Place your dataset (`amazon_reviews_sample.csv`) in the `data/` folder.
2. Run each R script sequentially in the `scripts/` folder:
   - `01_data_cleaning.R`
   - `02_sentiment_analysis.R`
   - `03_bigram_network.R`
   - `04_topic_modeling.R` *(optional)*
3. All visual outputs will be saved to the `outputs/` folder.

## Example Outputs
- **Word Cloud** — most frequent words.
- **Sentiment Distribution** — positive vs. negative sentiment scores.
- **Bigram Network** — common word pairs visualized as a graph.
- **Topic Model** — top terms per latent topic.

## Contact
**Cameron Dailey**  
[LinkedIn](https://www.linkedin.com/in/camerondailey)
