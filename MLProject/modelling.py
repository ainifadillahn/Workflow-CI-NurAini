import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def main(data_path):
    # =========================
    # Load data
    # =========================
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df  = pd.read_csv(os.path.join(data_path, "test.csv"))

    X_train = train_df.drop(columns=["Machine failure"])
    y_train = train_df["Machine failure"]

    X_test = test_df.drop(columns=["Machine failure"])
    y_test = test_df["Machine failure"]

    # =========================
    # Train model
    # =========================
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # =========================
    # Evaluation
    # =========================
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # =========================
    # Log to MLflow
    # MLflow Projects sudah setup semuanya, tinggal log!
    # =========================
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print(f"✅ Training completed successfully!")
    print(f"ROC AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    main(args.data_path)