import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn

def main():
    mlflow.set_experiment("Machine_Failure_RF")

    with mlflow.start_run():

        train_df = pd.read_csv("ai4i2020_preprocessed/train.csv")
        test_df  = pd.read_csv("ai4i2020_preprocessed/test.csv")

        X_train = train_df.drop(columns=["Machine failure"])
        y_train = train_df["Machine failure"]

        X_test = test_df.drop(columns=["Machine failure"])
        y_test = test_df["Machine failure"]

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        # log metric
        mlflow.log_metric("roc_auc", auc)

        # log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print("ROC AUC:", auc)


if __name__ == "__main__":
    main()
