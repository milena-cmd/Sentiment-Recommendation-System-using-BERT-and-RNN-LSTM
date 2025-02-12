
Hybrid Recommendation System using BERT and RNN-LSTM

This repository implements a recommendation system for the e-learning domain by leveraging sentiment analysis through two approaches:

    BERT-based model
    RNN-LSTM-based model

Additionally, a trust-based recommendation module is integrated, which calculates the similarity between learners, builds a trust network, and generates final course recommendations by combining:

    The utility of courses recommended by trusted learners, and
    The knowledge acquired by the learners (via a performance-based score).

Repository Structure

├── data/ # Raw and processed datasets ├── docs/ # Project documentation and installation instructions ├── notebooks/ # Jupyter notebooks for experiments ├── results/ # Figures and tables from evaluations ├── src/ # Source code for preprocessing, training, evaluation, trust and recommendation modules
├── requirements.txt # Python dependencies └── .gitignore # Git ignore file
Project Overview

    Data Collection & Preprocessing: Data is collected from multiple e-learning platforms and preprocessed to clean learner reviews.
    Modeling: Two approaches are implemented:
        A BERT-based model for deep contextual sentiment analysis.
        A RNN-LSTM-based model for sequential analysis.
    Trust Module: Computes similarity between learners (using a sigmoid function and Pearson correlation) and builds a trust network.
    Recommendation: Combines the weighted performance of learners (Reccl) with course utility (Ut_c_N) to generate final course recommendations.
    Evaluation: Models are evaluated using metrics such as Precision, Recall, F1-Score, RMSE, MAE, and NDCG.

For more details, please refer to installation instructions.
