import pandas as pd
import argparse
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_model(train_path, test_path, model_out):
    # Load preprocessed data
    print(f"[INFO] Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # strip preprocessed data column names
    train_df.columns = train_df.columns.str.strip().str.lower()
    test_df.columns = test_df.columns.str.strip().str.lower()


    # Split features and labels
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # Train XGBoost classifier
    print("[INFO] Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=5,  # adjust depending on label imbalance
        use_label_encoder=False,
        eval_metric='auc'
    )

    model.fit(X_train, y_train)

    # Evaluate
    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("=== ROC AUC Score ===")
    print(roc_auc_score(y_test, y_proba))

    # Save model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print(f"[+] Model saved to {model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--model_out", default="models/xgb_model.joblib", help="Path to save trained model")
    args = parser.parse_args()

    train_model(args.train, args.test, args.model_out)
