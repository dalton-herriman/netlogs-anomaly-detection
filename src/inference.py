import pandas as pd
import argparse
import joblib
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(input_csv):
    df = pd.read_csv(input_csv)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Drop irrelevant columns (if present)
    df = df.drop(columns=["flow id", "timestamp", "label"], errors="ignore")

    # Fill missing and replace infs
    df = df.fillna(0)
    df.replace([float('inf'), float('-inf')], 0, inplace=True)

    # Encode categorical features
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.factorize(df[col])[0]

    # Scale numeric values
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df_scaled

def run_inference(model_path, input_csv):
    print(f"[INFO] Loading model from {model_path}")
    model = joblib.load(model_path)

    print(f"[INFO] Preprocessing input data from {input_csv}")
    data = load_and_preprocess(input_csv)

    print("[INFO] Generating predictions...")
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    results = pd.DataFrame({
        "prediction": predictions,
        "anomaly_score": probabilities
    })

    print("\n=== Inference Results ===")
    print(results.head())

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/xgb_model.joblib", help="Path to saved model file")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    args = parser.parse_args()

    run_inference(args.model, args.input)
