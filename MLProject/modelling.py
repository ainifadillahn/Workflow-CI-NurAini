import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn
import argparse

def main(data_path):
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df  = pd.read_csv(f"{data_path}/test.csv")

    X_train = train_df.drop(columns=["Machine failure"])
    y_train = train_df["Machine failure"]

    X_test = test_df.drop(columns=["Machine failure"])
    y_test = test_df["Machine failure"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # LOG PARAM
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    # LOG METRIC
    mlflow.log_metric("roc_auc", auc)

    # LOG MODEL
    mlflow.sklearn.log_model(model, artifact_path="model")

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
