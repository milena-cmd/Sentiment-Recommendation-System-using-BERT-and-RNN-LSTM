# src/evaluate.py

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, ndcg_score

def evaluate_metrics(y_true, y_pred, y_score=None):
    """
    Compute evaluation metrics: Precision, Recall, F1-Score, RMSE, MAE, and NDCG (if y_score is provided).
    
    Args:
        y_true (list or np.array): Ground truth labels.
        y_pred (np.array): Predicted labels (e.g., from argmax).
        y_score (np.array, optional): Predicted probabilities or scores. Required for NDCG.
    
    Returns:
        tuple: (precision, recall, f1, rmse, mae, ndcg)
    """
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    ndcg = None
    if y_score is not None:
        # Convert y_true into one-hot encoding for NDCG computation
        n_classes = y_score.shape[1]
        y_true_onehot = np.zeros_like(y_score)
        for i, label in enumerate(y_true):
            if label < n_classes:
                y_true_onehot[i, label] = 1
        ndcg = ndcg_score(y_true_onehot, y_score)
    
    return precision, recall, f1, rmse, mae, ndcg

def evaluate_bert():
    from transformers import BertTokenizer, TFBertForSequenceClassification
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_path = os.path.join(os.getcwd(), 'results', 'bert_model')
    model = TFBertForSequenceClassification.from_pretrained(model_path)
    
    # Load processed data (using amazon_books_processed.csv)
    data_file = os.path.join(os.getcwd(), 'data', 'processed', 'amazon_books_processed.csv')
    df = pd.read_csv(data_file)
    texts = df['cleaned_review'].tolist()
    y_true = df['label'].tolist()
    
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='tf')
    # Get logits (predicted scores) from the model
    predictions = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])[0].numpy()
    y_pred = np.argmax(predictions, axis=1)
    
    precision, recall, f1, rmse, mae, ndcg = evaluate_metrics(y_true, y_pred, predictions)
    print('BERT Model Evaluation:')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, NDCG: {ndcg:.4f}')
    
def evaluate_rnn():
    model_path = os.path.join(os.getcwd(), 'results', 'rnn_lstm_model', 'rnn_lstm_model.h5')
    model = tf.keras.models.load_model(model_path)
    
    # Load processed data and prepare tokenizer (using amazon_books_processed.csv)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    data_file = os.path.join(os.getcwd(), 'data', 'processed', 'amazon_books_processed.csv')
    df = pd.read_csv(data_file)
    texts = df['cleaned_review'].tolist()
    y_true = df['label'].tolist()
    
    VOCAB_SIZE = 5000
    MAX_LENGTH = 512
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    predictions = model.predict(X)
    y_pred = np.argmax(predictions, axis=1)
    
    precision, recall, f1, rmse, mae, ndcg = evaluate_metrics(y_true, y_pred, predictions)
    print('RNN-LSTM Model Evaluation:')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, NDCG: {ndcg:.4f}')

def main(model_choice):
    if model_choice == 'bert':
        evaluate_bert()
    elif model_choice == 'rnn':
        evaluate_rnn()
    else:
        print('Invalid model choice. Use \"bert\" or \"rnn\".')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate BERT or RNN-LSTM model.')
    parser.add_argument('--model', type=str, required=True, help='Model to evaluate: bert or rnn')
    args = parser.parse_args()
    main(args.model)
