# src/bert_model.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 4
MAX_LENGTH = 512

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Assumes the processed CSV has columns: 'cleaned_review' and 'label'
    texts = df['cleaned_review'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def encode_data(texts, labels, tokenizer, max_length=MAX_LENGTH):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='tf')
    return inputs, np.array(labels)

def train_bert_model(data_file, model_save_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    texts, labels = load_data(data_file)
    inputs, labels = encode_data(texts, labels, tokenizer)
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Split data into training and testing (80/20)
    X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = train_test_split(
        input_ids, attention_mask, labels, test_size=0.2, random_state=42)
    
    num_labels = len(set(labels))
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    model.fit([X_train_ids, X_train_masks], y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=([X_test_ids, X_test_masks], y_test))
    model.save_pretrained(model_save_path)
    print(f'Model saved to {model_save_path}')
    
if __name__ == '__main__':
    # Example usage: using the processed Amazon Books Reviews dataset
    data_file = os.path.join(os.getcwd(), 'data', 'processed', 'amazon_books_processed.csv')
    model_save_path = os.path.join(os.getcwd(), 'results', 'bert_model')
    os.makedirs(model_save_path, exist_ok=True)
    train_bert_model(data_file, model_save_path)
