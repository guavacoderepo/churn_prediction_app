import dagshub
import mlflow
from config.setting import Settings

def mlflow_connect():
    """Connect to MLflow, create experiment if missing, and return experiment ID."""
    try:
        settings = Settings()  # type: ignore
        dagshub.init(repo_owner='guavacoderepo', repo_name='churn_prediction_app', mlflow=True) # type: ignore

        mlflow.set_tracking_uri(settings.TRACKING_URI)
        exp = mlflow.get_experiment_by_name(settings.EXPERIMENT_NAME) # type: ignore

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
        if exp is None:
            raise RuntimeError("❌ Failed to create or fetch MLflow experiment.")

        return exp.experiment_id # type: ignore

    except Exception as e:
         raise ConnectionError(f"Failed to initialize DagsHub or connect to MLflow: {e}") from e
