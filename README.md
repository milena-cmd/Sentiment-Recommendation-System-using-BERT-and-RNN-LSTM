# Sentiment Recommendation System using BERT and RNN-LSTM

This repository implements a recommendation system for the e-learning domain by leveraging sentiment analysis through two approaches:
- **BERT-based model**
- **RNN-LSTM-based model**

Additionally, a trust-based recommendation module is integrated, which calculates the similarity between learners, builds a trust network, and generates final course recommendations by combining:
- The utility of courses recommended by trusted learners, and
- The knowledge acquired by the learners (via a performance-based score).

## Repository Structure

├── data/
│   ├── raw/
│   │   ├── oula.csv                      # Raw (OULA) dataset enriched with learners’ reviews
│   │   ├── coursera.csv                  # Raw Coursera dataset
│   │   ├── yelp.csv                       # Raw Yelp Reviews dataset
│   │   ├── tripadvisor.csv                # Raw TripAdvisor Hotel Reviews dataset
│   │   ├── amazon_books.csv               # Raw Amazon Books Reviews dataset
│   │   ├── amazon_products.csv            # Raw Amazon Product Reviews dataset
│   └── processed/                        # Processed data will be stored here
├── docs/
│   ├── README.md                         # Main project description
│   └── installation.md                   # Detailed installation and usage instructions
├── notebooks/                            # Jupyter notebooks for experiments
│   ├── data_preprocessing.ipynb          # Notebook for data cleaning and preprocessing
│   ├── model_training.ipynb              # Notebook for training the models (BERT and RNN-LSTM)
│   └── evaluation.ipynb                  # Notebook for evaluating the models and visualizations
├── results/
│   ├── figures/                          # Graphs and plots generated during evaluation
│   └── tables/                           # Exported tables with performance metrics
├── src/
│   ├── __init__.py
│   ├── preprocess.py                     # Python script for data preprocessing
│   ├── bert_model.py                     # Code for training/fine-tuning the BERT model
│   ├── rnn_lstm_model.py                 # Code for training the RNN-LSTM model
│   ├── train.py                          # Script that calls model training functions
│   ├── evaluate.py                       # Script for model evaluation
│   ├── visualize.py                      # Script to generate visualizations (graphs)
│   ├── trust_module.py                   # Module implementing the trust computations (similarity, confidence matrix, etc.)
│   └── recommend.py                      # Module implementing the recommendation algorithm (Reccl and final recommendation)
├── requirements.txt                      # Python dependencies
└── .gitignore                            # Files/folders to ignore in Git

## Project Overview

- **Data Collection & Preprocessing:** Data is collected from multiple e-learning platforms and preprocessed to clean learner reviews.
- **Modeling:** Two approaches are implemented:
  - A BERT-based model for deep contextual sentiment analysis.
  - A RNN-LSTM-based model for sequential analysis.
- **Trust Module:** Computes similarity between learners (using a sigmoid function and Pearson correlation) and builds a trust network.
- **Recommendation:** Combines the weighted performance of learners (Reccl) with course utility (Ut_c_N) to generate final course recommendations.
- **Evaluation:** Models are evaluated using metrics such as Precision, Recall, F1-Score, RMSE, MAE, and NDCG.

