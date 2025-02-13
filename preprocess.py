# src/preprocess.py

import os
import pandas as pd
import re
from bs4 import BeautifulSoup

def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove punctuation and non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text.lower()

def preprocess_file(input_path, output_path):
    df = pd.read_csv(input_path)
    # Assume the raw CSV has a column named 'review'; adjust if necessary.
    if 'review' in df.columns:
        df['cleaned_review'] = df['review'].apply(clean_text)
    else:
        print(f'Column \"review\" not found in {input_path}')
    # Assume a 'label' column exists for classification tasks. If not, you may add one.
    df.to_csv(output_path, index=False)
    print(f'Processed data saved to {output_path}')

def main():
    raw_dir = os.path.join(os.getcwd(), 'data', 'raw')
    processed_dir = os.path.join(os.getcwd(), 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    for file in os.listdir(raw_dir):
        if file.endswith('.csv'):
            input_path = os.path.join(raw_dir, file)
            output_path = os.path.join(processed_dir, file.replace('.csv', '_processed.csv'))
            preprocess_file(input_path, output_path)

if __name__ == '__main__':
    main()
