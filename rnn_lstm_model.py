# src/rnn_lstm_model.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 4
MAX_LENGTH = 512
VOCAB_SIZE = 5000

def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['cleaned_review'].tolist()
    labels = df['label'].tolist()  # Ensure 'label' column exists
    return texts, labels

def prepare_tokenizer(texts):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    return tokenizer

def encode_texts(tokenizer, texts, max_length=MAX_LENGTH):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded

def train_rnn_lstm_model(data_file, model_save_path):
    texts, labels = load_data(data_file)
    tokenizer = prepare_tokenizer(texts)
    sequences = encode_texts(tokenizer, texts)
    labels = np.array(labels)
    
    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
    
    num_labels = len(set(labels))
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LENGTH),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(num_labels, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    os.makedirs(model_save_path, exist_ok=True)
    model_path = os.path.join(model_save_path, 'rnn_lstm_model.h5')
    model.save(model_path)
    print(f'Model saved to {model_path}')
    
if __name__ == '__main__':
    data_file = os.path.join(os.getcwd(), 'data', 'processed', 'amazon_books_processed.csv')
    model_save_path = os.path.join(os.getcwd(), 'results', 'rnn_lstm_model')
    os.makedirs(model_save_path, exist_ok=True)
    train_rnn_lstm_model(data_file, model_save_path)
