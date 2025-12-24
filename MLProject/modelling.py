import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn
import argparse
import os

# ============================
# MLflow CONFIG
# ============================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ai4i2020-experiment")
mlflow.autolog()

def main(data_path):
    # ============================
    # Load data
    # ============================
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df  = pd.read_csv(os.path.join(data_path, "test.csv"))

    X_train = train_df.drop(columns=["Machine failure"])
    y_train = train_df["Machine failure"]

    X_test = test_df.drop(columns=["Machine failure"])
    y_test = test_df["Machine failure"]

    # ============================
    # Train model
    # ============================
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ============================
    # Log model (TANPA registry)
    # ============================
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model"
    )

    # ============================
    # Evaluation
    # ============================
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    mlflow.log_metric("roc_auc", auc)
    print("ROC AUC:", auc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="ai4i2020_preprocessed"
    )
    args = parser.parse_args()

    main(args.data_path)
