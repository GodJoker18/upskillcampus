import sys
import os

sys.path.append(os.path.abspath("src"))

from load_data import load_multiple_datasets
from preprocess import add_rul, clean_and_scale
from sequence import create_sequences
from baseline_model import train_and_evaluate

# Paths for all datasets
dataset_paths = [
    "data/train_FD001.txt",
    "data/train_FD002.txt",
    "data/train_FD003.txt",
    "data/train_FD004.txt"
]

print("Loading all CMAPSS datasets...")
df = load_multiple_datasets(dataset_paths)
print("Total records:", df.shape)

print("Adding RUL labels...")
df = add_rul(df)

print("Cleaning & scaling data...")
df, sensor_cols = clean_and_scale(df)

print("Creating sequences...")
X, y = create_sequences(df, sensor_cols, window_size=20)  
print("Sequence shape:", X.shape)

X = X[:30000]
y = y[:30000]
print("Using subset size:", X.shape[0])

print("Training ML model on combined datasets...")
rmse = train_and_evaluate(X, y)

print("Final RMSE (All datasets):", rmse)
print("Training completed successfully ✅")
