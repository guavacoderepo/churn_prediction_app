import mlflow
from mlflow import sklearn
from ...config.setting import Settings
from .etl_pipeline import ETLPipeline
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

settings = Settings() # type: ignore

class ModelingPipeline:
    """Train, evaluate, and log RandomForest model for churn prediction."""

    def __init__(self, params: dict = {}) -> None:
        """Initialize class, load data, and set hyperparameters."""
        self.data = ETLPipeline()
        self.params = params or self.default_params()
        self.model = None

    def default_params(self) -> dict:
        """Return default hyperparameters for RandomForestClassifier."""
        return {
            "n_estimators": 50,
            "criterion": "gini",
            "random_state": 42,
            "max_depth": 10,
            "class_weight": "balanced_subsample"
        }

    def train_model(self) -> RandomForestClassifier:
        """Train RandomForestClassifier on training data if not already trained."""
        X_train, y_train, _, _ = self.data.transform_data()
        if self.model is None:
            self.model = RandomForestClassifier(**self.params, n_jobs=-1).fit(X_train, y_train)
        return self.model

    def evaluate_model(self) -> dict:
        """Predict on test data and return accuracy, f1, recall, and precision."""
        _, _, X_test, y_test = self.data.transform_data()
        model = self.train_model()
        pred = model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "precision": precision_score(y_test, pred)
        }

    def mlflow_connect(self) -> str:
        """Connect to MLflow, create experiment if missing, and return experiment ID."""
        mlflow.set_tracking_uri(settings.TRACKING_URI)
        exp = mlflow.get_experiment_by_name(settings.EXPERIMENT_NAME)
        if exp is None:
            mlflow.create_experiment(
                name=settings.EXPERIMENT_NAME,
                tags={
                    "mlflow.note.content": "Customer churn prediction pipeline",
                    "team": "AI engineering team",
                    "project": "churn_prediction_end_to_end"
                },
                artifact_location="src/data/churn_pred_models"
            )
            exp = mlflow.get_experiment_by_name(settings.EXPERIMENT_NAME)
        return exp.experiment_id # type: ignore

    def log_to_mlflow(self, model_name: str, run_name: str):
        """Log model, hyperparameters, metrics, and tags to MLflow."""
        experiment_id = self.mlflow_connect()
        sk_model = self.train_model()
        _, _, X_test, _ = self.data.transform_data()

        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
            for key, val in self.params.items():
                mlflow.log_param(key, val)

            for key, val in self.evaluate_model().items():
                mlflow.log_metric(key, val)

            for key, val in {
                "mlflow.note.content": f"{model_name} training outcome",
                "model_name": model_name
            }.items():
                mlflow.set_tag(key, val)

            sklearn.log_model(
                sk_model=sk_model,
                input_example=X_test[:5],
                registered_model_name=f"{model_name}_models",
                signature=infer_signature(X_test[:5], sk_model.predict(X_test[:5])),
                name=f"{model_name}_artifact"
            )

        print(f"✅ Done logging {model_name}")
