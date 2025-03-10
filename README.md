# Hybrid Recommendation System using BERT and RNN-LSTM

This repository implements a recommendation system for the e-learning domain by leveraging sentiment analysis through two approaches:
- **BERT-based model**
- **RNN-LSTM-based model**

It also integrates a trust-based recommendation module that calculates the similarity between learners, builds a trust network, and generates final course recommendations by combining:
- The utility of courses recommended by trusted learners, and
- The knowledge acquired by learners (via performance-based scores).

## Datasets

The datasets used in this project are:
- **OULA**: Enriched with learners’ reviews
- **Coursera dataset**
- **Udemy Course Reviews dataset**
- **EdX Course Reviews dataset**
- **Amazon Books Reviews dataset** 
- **Skillshare Course Reviews dataset**

## Repository Structure

├── data/ # Raw and processed datasets ├── docs/ # Project documentation and installation instructions ├── notebooks/ # Jupyter notebooks for experiments ├── results/ # Figures and tables from evaluations ├── src/ # Source code for preprocessing, training, evaluation, trust and recommendation modules, and baselines ├── requirements.txt # Python dependencies └── .gitignore # Git ignore file

For more details, please refer to the [installation instructions](installation.md).


