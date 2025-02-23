import pandas as pd
import gdown
from sklearn.model_selection import train_test_split

def download_dataset(drive_url, output_path):
    """Download dataset from Google Drive."""
    gdown.download(drive_url, output_path, quiet=False)

def load_and_split_dataset(csv_path, test_size=0.2):
    """Load CSV dataset and split into train/test."""
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df

if __name__ == "__main__":
    DRIVE_URL = "https://drive.google.com/uc?id=1NHCnw7CmEr1ehtQCeuE0iNe58hAsUuYD"
    CSV_PATH = "data/dataset.csv"
    
    print("Downloading dataset...")
    download_dataset(DRIVE_URL, CSV_PATH)
    
    print("Loading and splitting dataset...")
    train_data, test_data = load_and_split_dataset(CSV_PATH)
    
    print("Dataset successfully loaded and split!")
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
"""

# Save load_dataset.py
with open("src/load_dataset.py", "w") as f:
    f.write(LOAD_DATASET_CODE)

print("Dataset loading and splitting script added successfully!")
