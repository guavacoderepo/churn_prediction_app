import mlflow
import numpy as np
from typing import Dict, Any
from mlflow import sklearn
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class ModelingPipeline:
    """Train, evaluate, and log RandomForest model for churn prediction."""

    def __init__(self, data: Any, exp_id:str) -> None:
        """Initialize class, load data, and set hyperparameters."""
        self.data = data
        self.model = None
        self.exp_id = exp_id
    
    def _unpack_data(self):
        """Safely unpack dataset into train/test splits."""
        try:
            X_train, X_test, y_train, y_test = self.data
        except Exception as e:
            raise ValueError("❌ Data must be a tuple: (X_train, X_test, y_train, y_test)") from e
        return X_train, X_test, y_train, y_test

    def train_model(self, params:Dict) -> RandomForestClassifier:
        """Train RandomForestClassifier on training data if not already trained."""
        X_train, _, y_train, _ = self._unpack_data()
        y_train = np.ravel(y_train).astype(int)
        
        if self.model is None:
            self.model = RandomForestClassifier(**params, n_jobs=-1).fit(X_train, y_train.ravel())
            print(self.model)
        return self.model # type: ignore

    def evaluate_model(self, model: RandomForestClassifier) -> dict:
        """Predict on test data and return accuracy, f1, recall, and precision."""
        _, X_test, _, y_test = self._unpack_data()
        y_test = np.ravel(y_test).astype(int)

        pred = model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "precision": precision_score(y_test, pred)
        }

    def log_to_mlflow(
            self,
            sk_model: RandomForestClassifier,
            model_name: str,
            params: Dict,
            run_name: str
        ):
        """Log model, hyperparameters, metrics, and tags to MLflow."""
        X_train, X_test, _, _ = self._unpack_data()
        
        experiment_id = self.exp_id
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
            for key, val in params.items():
                mlflow.log_param(key, val)

            eval_metrics = self.evaluate_model(sk_model)
            for key, val in eval_metrics.items():
                mlflow.log_metric(key, val)

            for key, val in {
                "mlflow.note.content": f"{model_name} training outcome",
                "model_name": model_name
            }.items():
                mlflow.set_tag(key, val)

            sklearn.log_model(
                sk_model=sk_model,
                input_example=X_train[:5],
                registered_model_name=f"{model_name}_models",
                signature=infer_signature(X_test[:5], sk_model.predict(X_test[:5])),
                artifact_path=f"{model_name}_artifact"
            )

        print(f"✅ Done logging {model_name}")
