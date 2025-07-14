# Imports
import pandas as pd
import argparse
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(input_path, output_paths):
    # Load raw CSV logs
    print(f"[INFO] Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    # Drop irrelevant columns
    df = df.drop(columns=["Flow ID", "Timestamp"], errors="ignore")

    # Handle missing values
    df = df.fillna(0)

    # Encode categorical features
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Split features/labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    # Normalize numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Reattach labels
    df_processed = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    # Split into train/test and save
    train, test = train_test_split(df_processed, test_size=0.2, random_state=42)
    os.makedirs(output_path, exist_ok=True)
    train.to_csv(os.path.join(output_path, "train.csv"), index=False)
    test.to_csv(os.path.join(output_path, "test.csv"), index=False)

    print(f"[+] Saved: {output_path}/train.csv")
    print(f"[+] Saved: {output_path}/test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV file")
    parser.add_argument(
        "--output", required=True, help="Folder to store processed files"
    )
    args = parser.parse_args()

    preprocess(args.input, args.output)
