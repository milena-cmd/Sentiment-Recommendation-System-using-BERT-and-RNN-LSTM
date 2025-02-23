# src/bert_model.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers.legacy import Adam

# Hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 4
MAX_LENGTH = 128  

def load_data(file_path):
    """Loads preprocessed dataset (train or test)."""
    df = pd.read_csv(file_path)

    if 'cleaned_review' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'cleaned_review' and 'label' columns")

    texts = df['cleaned_review'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    return texts, np.array(labels)

def encode_data(texts, labels, tokenizer, max_length=MAX_LENGTH):
    """Tokenizes text and converts to tensors."""
    encoded = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors="np")
    return np.array(encoded['input_ids']), np.array(encoded['attention_mask']), labels

def train_bert_model(train_file, test_file, model_save_path):
    """Fine-tunes BERT for sentiment classification."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load pre-split datasets
    train_texts, train_labels = load_data(train_file)
    test_texts, test_labels = load_data(test_file)

    # Encode datasets
    X_train_ids, X_train_masks, y_train = encode_data(train_texts, train_labels, tokenizer)
    X_test_ids, X_test_masks, y_test = encode_data(test_texts, test_labels, tokenizer)

    num_labels = len(set(train_labels))
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    optimizer = Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    # Train the model
    model.fit(
        x=[X_train_ids, X_train_masks], y=y_train,
        validation_data=([X_test_ids, X_test_masks], y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE
    )

    # Save trained model and tokenizer
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to {model_save_path}")

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "data", "processed")
    train_file = os.path.join(data_dir, "train.csv")
    test_file = os.path.join(data_dir, "test.csv")
    model_save_path = os.path.join(os.getcwd(), "results", "bert_model")

    os.makedirs(model_save_path, exist_ok=True)
    train_bert_model(train_file, test_file, model_save_path)

